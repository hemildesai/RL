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

"""Configuration for an externally managed stock vLLM rollout server."""

from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class RemoteVllmRefitConfig(BaseModel, extra="forbid"):
    """Refit settings for an external vLLM deployment.

    ``transport="hf_checkpoint"`` (default) exports an HF checkpoint and has
    vLLM reload it through ``/collective_rpc`` (requires
    ``policy.generation.refit_transport=hf_checkpoint``).

    ``transport="nccl"`` uses vLLM's native weight-transfer engine instead
    (servers started with ``--weight-transfer-config '{"backend":"nccl"}'``;
    leave ``policy.generation.refit_transport`` unset). Every policy rank and
    every vLLM rank join one NCCL group, policy rank 0 broadcasts the HF-named
    weights in packed buckets, and vLLM loads them with its own loaders. The
    engine ranks start at ``rank_offset = train world size``. When
    ``base_url`` fronts several vLLM servers, the endpoint must fan the
    control and weight-transfer requests out to every server and give server
    ``i`` the rank offset ``rank_offset + i * nccl_engine_world_size``; it
    advertises this with ``"control_fanout": true`` in its ``/health`` JSON.

    The trainer side of the NCCL group must use the same NCCL network plugin
    as the vLLM servers (for example set ``NCCL_NET_PLUGIN`` in the policy
    ``env_vars`` to match), or the rendezvous hangs. MTP draft weights
    (``mtp.*``) are sent but dropped by the main-model loader.
    """

    inflight_policy: Literal["abort", "wait", "keep"] = "keep"
    timeout_s: PositiveFloat = 1800.0
    transport: Literal["hf_checkpoint", "nccl"] = "hf_checkpoint"
    # transport=nccl only. NCCL ranks of one vLLM server (TP x PP x DP).
    nccl_engine_world_size: PositiveInt = 1
    # transport=nccl only. vLLM servers behind base_url; setup waits up to
    # timeout_s until /health reports this many, all healthy. None: use what
    # /health reports (one server when it has no total_backends field).
    nccl_num_engines: PositiveInt | None = None
    # transport=nccl only. Packed broadcast geometry, sent to vLLM in init_info.
    nccl_packed_buffer_size_bytes: PositiveInt = 1024**3
    nccl_packed_num_buffers: PositiveInt = 2


class RemoteVllmServiceConfig(BaseModel, extra="forbid"):
    """Connection and control settings for one external vLLM deployment.

    ``base_url`` may include the OpenAI ``/v1`` prefix. Control paths are
    resolved against the URL origin so the same endpoint can be handed to
    NeMo-Gym and used for vLLM's development endpoints.
    """

    base_url: str = Field(min_length=1)
    served_model_name: str = Field(min_length=1)
    max_model_len: PositiveInt
    health_path: str = "/health"
    models_path: str = "/v1/models"
    server_info_path: str = "/server_info"
    pause_path: str = "/pause"
    resume_path: str = "/resume"
    collective_rpc_path: str = "/collective_rpc"
    reset_prefix_cache_path: str = "/reset_prefix_cache"
    request_timeout_s: PositiveFloat = 600.0
    connect_timeout_s: PositiveFloat = 5.0
    api_key_env_var: str | None = None
    refit: RemoteVllmRefitConfig

    @model_validator(mode="after")
    def validate_urls_and_timeouts(self) -> "RemoteVllmServiceConfig":
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        path_fields = (
            "health_path",
            "models_path",
            "server_info_path",
            "pause_path",
            "resume_path",
            "collective_rpc_path",
            "reset_prefix_cache_path",
        )
        for field_name in path_fields:
            path = getattr(self, field_name)
            if not path.startswith("/"):
                raise ValueError(f"{field_name} must be an absolute HTTP path")
        if self.connect_timeout_s > self.request_timeout_s:
            raise ValueError("connect_timeout_s cannot exceed request_timeout_s")
        return self


class RemoteVllmServiceInfo(BaseModel, extra="forbid"):
    """Non-mutating facts verified during setup-time preflight."""

    model: str
    dev_mode_control_api: bool
    server_info: dict[str, Any]
