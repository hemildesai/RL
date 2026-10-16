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

"""NCCL refit of an external vLLM deployment via vLLM's weight-transfer engine.

Setup builds one NCCL group: ranks ``[0, train_world_size)`` are the policy
workers, the vLLM ranks follow (``rank_offset = train_world_size``). Per refit::

    pause -> /start_weight_update
          -> /update_weights  ||  policy.broadcast_weights_for_collective
          -> /finish_weight_update(version) -> resume

The HTTP calls that pair with a collective run on a side thread. A rejected
request never enters NCCL, so the refit fails as soon as it errors instead of
leaving the policy ranks blocked in the collective.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

_ENGINE_POLL_INTERVAL_S = 5.0


class RemoteVllmNcclWeightSynchronizer(WeightSynchronizer):
    """Broadcast policy weights to an external vLLM deployment over NCCL."""

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        refit_timeout_s: Optional[float] = None,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._refit_timeout_s = refit_timeout_s
        self._refit = generation.remote_config.refit
        self._update_info: Optional[dict[str, list[Any]]] = None
        self._next_version = 1
        self._stale = True

    @property
    def is_stale(self) -> bool:
        return self._stale

    def _num_engines(self) -> int:
        """Wait until the expected vLLM servers are registered and healthy."""
        expected = self._refit.nccl_num_engines
        deadline = time.monotonic() + self._refit.timeout_s
        while True:
            health = self._generation.client.health(
                timeout_s=self._generation.remote_config.connect_timeout_s
            )
            total = (health or {}).get("total_backends")
            if not isinstance(total, int):
                total, healthy = 1, 1  # a single vLLM server
            else:
                healthy = health.get("healthy_backends", total)
            if expected is not None and total > expected:
                raise RuntimeError(
                    f"{total} vLLM servers behind base_url, more than "
                    f"refit.nccl_num_engines={expected}"
                )
            if total > 0 and healthy == total and total == (expected or total):
                return total
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for {expected or 'all'} healthy vLLM "
                    f"servers behind base_url ({healthy}/{total} healthy)"
                )
            time.sleep(_ENGINE_POLL_INTERVAL_S)

    def _run_with_http(
        self, futures: list[ray.ObjectRef], call: Callable[[], None], what: str
    ) -> None:
        """Run ``call`` beside the policy collective; fail fast if it fails."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            http: Future = executor.submit(call)
            pending = list(futures)
            while pending:
                _, pending = ray.wait(pending, num_returns=len(pending), timeout=1.0)
                if http.done() and http.exception() is not None:
                    raise RuntimeError(
                        f"External vLLM {what} failed: {http.exception()}"
                    )
            ray.get(list(futures))
            http.result()
        finally:
            executor.shutdown(wait=False)

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info(
            refit_payload_mode=self._generation.get_refit_payload_mode()
        )
        if not state_dict_info:
            raise RuntimeError("policy.prepare_refit_info returned no tensors")
        # vLLM's NCCLWeightTransferUpdateInfo; "packed" belongs in init_info.
        self._update_info = {
            "names": list(state_dict_info),
            "dtype_names": [
                str(dtype).removeprefix("torch.")
                for _, dtype in state_dict_info.values()
            ],
            "shapes": [list(shape) for shape, _ in state_dict_info.values()],
        }

        engine_world_size = self._refit.nccl_engine_world_size
        train_world_size = self._train_cluster.world_size()
        world_size = train_world_size + self._num_engines() * engine_world_size
        ip, port = self._train_cluster.get_master_address_and_port()
        sender = self._generation.get_collective_sender_spec()
        init_info = {
            "master_address": ip,
            "master_port": port,
            "rank_offset": train_world_size,
            "world_size": world_size,
            "packed": True,
            "packed_buffer_size_bytes": sender.buffer_size_bytes,
            "packed_num_buffers": sender.num_buffers,
        }
        futures = self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer=sender.nccl_peer,
        )
        self._run_with_http(
            futures,
            lambda: self._generation.client.init_weight_transfer_engine(
                init_info, engine_world_size=engine_world_size
            ),
            "init_weight_transfer_engine",
        )

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        if kv_scales is not None:
            raise NotImplementedError(
                "remote_vllm NCCL refit does not send separate KV scales"
            )
        if self._update_info is None:
            raise RuntimeError("init_communicator() must run before sync_weights()")
        update_info = self._update_info

        self._stale = True
        version = self._next_version
        client = self._generation.client
        sender = self._generation.get_collective_sender_spec()
        context = (
            timer.time("prepare_for_generation/nccl_refit_remote_vllm")
            if timer is not None
            else nullcontext()
        )
        with context:
            self._generation.pause_generation_for_refit(clear_cache=True)
            # On failure execution never reaches resume: some servers may hold
            # partially updated weights, so serving from them would be unsafe.
            client.start_weight_update()
            futures = self._policy.broadcast_weights_for_collective(
                refit_timeout_s=self._refit_timeout_s,
                buffer_size_bytes=sender.buffer_size_bytes,
                num_buffers=sender.num_buffers,
            )
            self._run_with_http(
                futures, lambda: client.update_weights(update_info), "update_weights"
            )
            client.finish_weight_update(str(version))
            self._generation.resume_generation_after_refit()

        self._next_version += 1
        self._stale = False

    def shutdown(self) -> None:
        return None
