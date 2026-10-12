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

"""vLLM versions the plugin's renderer and route hooks have been checked against."""

SUPPORTED_VLLM_VERSIONS = ("0.29.0", "0.30.0", "0.31.0")


def is_supported_vllm_version(installed_version: str) -> bool:
    """Compare the release only, so local builds such as ``0.31.0+precompiled`` pass."""
    return installed_version.split("+", 1)[0] in SUPPORTED_VLLM_VERSIONS
