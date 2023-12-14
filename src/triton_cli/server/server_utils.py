#!/usr/bin/env python3

# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class TritonServerUtils:
    @staticmethod
    def mpi_run(world_size: int, model_repo: str) -> str:
        """
        Parameters
        ----------
        world_size : int
            The path to the model repository
        model_repo : str
            The path to the model repository
        Returns
        -------
        The appropriate world size to use to run the tensorrtllm
        engine(s) stored in the model repository
        """
        cmd = ["mpirun", "--allow-run-as-root"]
        for i in range(world_size):
            cmd += ["-n", "1", "/opt/tritonserver/bin/tritonserver"]
            cmd += [
                f"--model-repository={model_repo}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":",
            ]
        return cmd
