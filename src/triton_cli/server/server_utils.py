#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
import logging
import json
from pathlib import Path
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format, text_format

from triton_cli.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonServerUtils:
    """
    A utility class for launching a Triton server.
    """

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._trtllm_model_config_path = self._find_trtllm_model_config_path()
        self._is_trtllm_model = self._trtllm_model_config_path is not None

        if self._is_trtllm_model:
            self._world_size = self._parse_world_size()

    def _find_trtllm_model_config_path(self) -> Path:
        """
        Returns
        -------
            A pathlib.Path object containing the path to the TRT LLM model config folder.
        Assumptions
        ----------
            - Assumes only a single model uses the TRT LLM backend (could have
            multiple engines)
        """
        try:
            match = subprocess.check_output(
                [
                    "grep",
                    "-r",
                    "--include",
                    "*.pbtxt",
                    'backend: "tensorrtllm"',
                    self._model_repo_path,
                ]
            )
            # Example match: b'{PATH}/config.pbtxt:backend: "tensorrtllm"\n'
            return Path(match.decode().split(":")[0])
        except subprocess.CalledProcessError:
            # The 'grep' command will return a non-zero exit code
            # if no matches are found.
            return None

    def _get_engine_path(self, config_path: Path) -> str:
        """
        Parameters
        ----------
        config_path : Path
            A pathlib.Path object containing the path to the TRT LLM model config folder.
        Returns
        -------
            A pathlib.Path object containing the path to the TRT LLM engines.
        """
        try:
            with open(config_path) as config_file:
                config = text_format.Parse(config_file.read(), mc.ModelConfig())
                json_config = json.loads(
                    json_format.MessageToJson(config, preserving_proto_field_name=True)
                )
                return Path(json_config["parameters"]["gpt_model_path"]["string_value"])
        except KeyError as e:
            raise Exception(
                f"Unable to extract engine path from config file {config_path}. Key error: {str(e)}"
            )
        except OSError:
            raise Exception(
                f"Failed to open config file for tensorrt_llm. Searched: {config_path}"
            )

    def _parse_world_size(self) -> int:
        """
        Returns
        -------
            The appropriate world size to use to run the tensorrtllm
            engine(s).
        """
        assert self._is_trtllm_model, "World size cannot be parsed from a model repository that does not contain a TRT LLM model."
        try:
            engine_path = self._get_engine_path(self._trtllm_model_config_path)
            engine_config_path = engine_path / "config.json"
            with open(engine_config_path) as json_data:
                data = json.load(json_data)
                return int(data["builder_config"]["tensor_parallel"])
        except KeyError as e:
            raise Exception(
                f"Unable to extract world size from {engine_config_path}. Key error: {str(e)}"
            )
        except OSError:
            raise Exception(f"Unable to open {engine_config_path}")

    def mpi_run(self) -> str:
        """
        Returns
        -------
            TRT LLM models must be run using MPI. This function constructs
            the appropriate mpi command to run a TRT LLM engine given a
            previously parsed world size.
        """
        cmd = ["mpirun", "--allow-run-as-root"]
        for i in range(self._world_size):
            cmd += ["-n", "1", "/opt/tritonserver/bin/tritonserver"]
            cmd += [
                f"--model-repository={self._model_repo_path}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":",
            ]
        return cmd

    def prepare_command(self, env_cmds: list, server_args: str) -> str:
        """
        Parameters
        ----------
        env_cmds : str
            A list of environment commands to run with the tritonserver (non-trtllm models)
        server_args : str
            Command-line arguments to run tritonserver (non-trtllm models)
        Returns
        -------
            The appropriate command for launching a tritonserver.
        """
        if self._is_trtllm_model:
            return " ".join(self.mpi_run())
        else:
            return " ".join(env_cmds + ["tritonserver", server_args])
