# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from types import SimpleNamespace
import threading
from ray import cloudpickle
import urllib.error
import urllib.request
from typing import Any

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmClient,
    RemoteVllmFleetClient,
    RemoteVllmGeneration,
    RemoteVllmServiceConfig,
    preflight_remote_vllm_service,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.hf_checkpoint_weight_synchronizer import (
    HfCheckpointWeightSynchronizer,
)


class _Response:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.status = 200
        self._payload = b"" if body is None else json.dumps(body).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _config(**updates: Any) -> RemoteVllmServiceConfig:
    values: dict[str, Any] = {
        "base_url": "http://rollout.test:9210/v1",
        "served_model_name": "policy",
        "max_model_len": 4096,
        "refit": {},
    }
    values.update(updates)
    return RemoteVllmServiceConfig.model_validate(values)


def test_remote_service_config_rejects_non_http_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url must start"):
        _config(base_url="rollout.test:9210/v1")


def test_external_backend_reuses_hf_transport_without_ray_inference_cluster(
    tmp_path: Path,
) -> None:
    generation = RemoteVllmGeneration(
        {
            "remote_vllm_cfg": _config().model_dump(),
            "refit_transport": "hf_checkpoint",
            "refit_cfg": {"hf_checkpoint": {"checkpoint_dir": str(tmp_path)}},
        }
    )
    policy = SimpleNamespace(cfg={"megatron_cfg": {"enabled": True}})
    synchronizer = create_weight_synchronizer(policy, generation, "remote_vllm", False)
    assert isinstance(synchronizer, HfCheckpointWeightSynchronizer)


def test_remote_service_config_rejects_unknown_refit_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(refit={"checkpoint_dir": "/shared/hf-exports"})


def test_preflight_uses_stock_vllm_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 5.0
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/health"):
            return _Response()
        if request.full_url.endswith("/v1/models"):
            return _Response({"data": [{"id": "policy"}]})
        return _Response({"model_config": {"max_model_len": 4096}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    info = preflight_remote_vllm_service(_config())

    assert info.model == "policy"
    assert info.dev_mode_control_api is True
    assert requested_urls == [
        "http://rollout.test:9210/health",
        "http://rollout.test:9210/v1/models",
        "http://rollout.test:9210/server_info",
    ]


def test_preflight_rejects_served_model_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _Response(),
            _Response({"data": [{"id": "another-model"}]}),
        ]
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: next(responses),
    )

    with pytest.raises(RuntimeError, match="model mismatch"):
        preflight_remote_vllm_service(_config())


def test_preflight_rejects_multiple_independent_lb_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: _Response(
            {"status": "ok", "healthy_backends": 2, "total_backends": 2}
        ),
    )

    with pytest.raises(RuntimeError, match="control-plane fan-out"):
        preflight_remote_vllm_service(_config())


def test_reload_uses_collective_rpc_with_weights_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 1800.0
        requests.append(request)
        return _Response({"result": [None, None]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    RemoteVllmClient(_config()).reload_weights("/shared/hf-exports/version_00000001")

    assert requests[0].full_url == "http://rollout.test:9210/collective_rpc"
    assert json.loads(requests[0].data or b"") == {
        "method": "reload_weights",
        "kwargs": {"weights_path": "/shared/hf-exports/version_00000001"},
    }


def test_pause_keeps_inflight_requests_and_clears_their_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"status": "paused"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    config = _config()
    RemoteVllmClient(config).pause(
        mode=config.refit.inflight_policy,
        clear_cache=True,
    )

    assert requests[0].full_url == (
        "http://rollout.test:9210/pause?mode=keep&clear_cache=true"
    )


def test_token_capture_configuration_uses_service_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append(request)
        return _Response({"status": "ok", "fanout_backends": 8})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    config = _config()

    RemoteVllmClient(config).configure_token_capture(
        bridge_url="http://10.0.0.1:12345",
        auth_token="secret",
    )

    assert requests[0].full_url == (
        "http://rollout.test:9210/v1/nemo-rl/token-capture/configure"
    )
    assert json.loads(requests[0].data or b"") == {
        "bridge_url": "http://10.0.0.1:12345",
        "auth_token": "secret",
    }


def test_generation_serializes_bridge_coordinates_not_live_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnpicklableBridge:
        def __init__(self) -> None:
            self.lock = threading.Lock()

    generation = RemoteVllmGeneration({"remote_vllm_cfg": _config().model_dump()})
    live_bridge = UnpicklableBridge()
    generation._token_capture_bridge = live_bridge  # type: ignore[assignment]
    generation._token_capture_bridge_url = "http://10.0.0.1:12345"
    generation._token_capture_auth_token = "secret"

    restored = cloudpickle.loads(cloudpickle.dumps(generation))

    assert generation._token_capture_bridge is live_bridge
    assert restored._token_capture_bridge is None
    assert restored._token_capture_bridge_url == "http://10.0.0.1:12345"
    assert restored._token_capture_auth_token == "secret"

    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout == 30.0
        requests.append(request)
        return _Response({"weight_version": 7})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    restored.set_rollout_weight_version(7)

    assert requests[0].full_url == (
        "http://10.0.0.1:12345/v1/nemo-rl/token-capture/weight-version"
    )
    assert requests[0].headers["Authorization"] == "Bearer secret"
    assert json.loads(requests[0].data or b"") == {"weight_version": 7}


_ENGINES = ["http://engine-0.test:8000", "http://engine-1.test:8000"]


def test_remote_service_config_validates_control_urls() -> None:
    assert _config().engine_control_urls == ["http://rollout.test:9210/v1"]
    assert _config(control_urls=_ENGINES).engine_control_urls == _ENGINES
    with pytest.raises(ValidationError, match="control_urls must start"):
        _config(control_urls=["engine-0.test:8000"])
    with pytest.raises(ValidationError, match="duplicates"):
        _config(control_urls=[_ENGINES[0], _ENGINES[0]])


def test_preflight_checks_router_data_path_and_every_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requested_urls.append(request.full_url)
        if request.full_url.endswith("/health"):
            # A router over several engines reports them; each engine reports none.
            if request.full_url.startswith("http://rollout.test"):
                return _Response({"total_backends": 2})
            return _Response()
        if request.full_url.endswith("/v1/models"):
            return _Response({"data": [{"id": "policy"}]})
        return _Response({"model_config": {"max_model_len": 4096}})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    info = preflight_remote_vllm_service(_config(control_urls=_ENGINES))

    assert info.num_engines == 2
    assert requested_urls == [
        "http://rollout.test:9210/health",
        "http://rollout.test:9210/v1/models",
        "http://engine-0.test:8000/health",
        "http://engine-0.test:8000/v1/models",
        "http://engine-0.test:8000/server_info",
        "http://engine-1.test:8000/health",
        "http://engine-1.test:8000/v1/models",
        "http://engine-1.test:8000/server_info",
    ]


def test_preflight_rejects_control_url_that_is_itself_a_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        if request.full_url == "http://engine-1.test:8000/health":
            return _Response({"total_backends": 3})
        if request.full_url.endswith("/v1/models"):
            return _Response({"data": [{"id": "policy"}]})
        if request.full_url.endswith("/server_info"):
            return _Response({"model_config": {"max_model_len": 4096}})
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="engine-1.test:8000"):
        preflight_remote_vllm_service(_config(control_urls=_ENGINES))


def test_fleet_client_fans_refit_out_to_every_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    lock = threading.Lock()

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        with lock:
            requested.append(request.full_url)
        return _Response({"status": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    fleet = RemoteVllmFleetClient(_config(control_urls=_ENGINES))

    fleet.pause(mode="keep", clear_cache=True)
    fleet.reload_weights("/shared/hf-exports/version_00000002")
    fleet.reset_prefix_cache()
    fleet.resume()
    fleet.configure_token_capture(bridge_url="http://10.0.0.1:1", auth_token="t")

    for engine in _ENGINES:
        assert sorted(url for url in requested if url.startswith(engine)) == sorted(
            [
                f"{engine}/pause?mode=keep&clear_cache=true",
                f"{engine}/collective_rpc",
                f"{engine}/reset_prefix_cache",
                f"{engine}/resume",
                f"{engine}/v1/nemo-rl/token-capture/configure",
            ]
        )
    assert not any(url.startswith("http://rollout.test") for url in requested)


def test_fleet_client_attempts_every_engine_and_names_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    lock = threading.Lock()

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        with lock:
            requested.append(request.full_url)
        if request.full_url.startswith(_ENGINES[0]):
            raise urllib.error.URLError("connection refused")
        return _Response({"status": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    fleet = RemoteVllmFleetClient(_config(control_urls=_ENGINES))

    with pytest.raises(RuntimeError, match=r"resume failed on 1 of 2 engines.*engine-0"):
        fleet.resume()
    assert f"{_ENGINES[1]}/resume" in requested


def test_generation_refits_every_engine_but_serves_through_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requested: list[str] = []
    lock = threading.Lock()

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        with lock:
            requested.append(request.full_url)
        return _Response({"status": "ok"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    generation = RemoteVllmGeneration(
        {
            "remote_vllm_cfg": _config(control_urls=_ENGINES).model_dump(),
            "refit_transport": "hf_checkpoint",
            "refit_cfg": {"hf_checkpoint": {"checkpoint_dir": str(tmp_path)}},
        }
    )

    assert generation.dp_openai_server_base_urls == ["http://rollout.test:9210/v1"]
    generation.pause_generation_for_refit(clear_cache=True)
    generation.reload_weights("/shared/hf-exports/version_00000003")
    generation.resume_generation_after_refit()

    for engine in _ENGINES:
        assert f"{engine}/collective_rpc" in requested
        assert f"{engine}/resume" in requested
