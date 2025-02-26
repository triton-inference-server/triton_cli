#!/usr/bin/env python3

# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Union
from pathlib import Path
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format, text_format

from .server_config import TritonServerConfig

from triton_cli.common import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonServerUtils:
    """
    A utility class for launching a Triton server.
    """

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._trtllm_utils = TRTLLMUtils(self._model_repo_path)

    def get_launch_command(
        self,
        server_config: TritonServerConfig,
        cmd_as_list: bool,
        env_cmds=[],
    ) -> Union[str, list]:
        """
        Parameters
        ----------
        server_config : TritonServerConfig
            A TritonServerConfig object containing command-line arguments to run tritonserver
        cmd_as_list : bool
            Whether the command string needs to be returned as a list of string (local requires list,
            docker requires str)
        env_cmds : list (optional)
            A list of environment commands to run with the tritonserver (non-trtllm models)
        Returns
        -------
            The appropriate command for launching a tritonserver.
        """

        # Don't use mpirun in the world_size==1 case because it obscures
        # errors at runtime, making debugging more difficult.
        if (
            self._trtllm_utils.has_trtllm_model()
            and self._trtllm_utils.get_world_size() > 1
        ):
            logger.info(
                f"Launching server with world size: {self._trtllm_utils.get_world_size()}"
            )
            cmd = self._trtllm_utils.mpi_run(server_config)
        else:
            cmd = (
                env_cmds + [server_config.server_path()] + server_config.to_args_list()
            )

        if cmd_as_list:
            return cmd
        else:
            return " ".join(cmd)


class TRTLLMUtils:
    """
    A utility class for handling TRT LLM-specific models.
    """

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._trtllm_model_config_path = self._find_trtllm_model_config_path()
        self._is_trtllm_model = self._trtllm_model_config_path is not None
        self._supported_args = ["model-repository"]

        self._world_size = -1
        if self._is_trtllm_model:
            self._world_size = self._parse_world_size()

    def has_trtllm_model(self) -> bool:
        """
        Returns
        -------
            A boolean indicating whether a TRT LLM model exists in the model repo
        """
        return self._is_trtllm_model

    def get_world_size(self) -> int:
        """
        Returns
        -------
            An int corresponding to the appropriate world size to use to run the TRT LLM engine(s).
        """
        return self._world_size

    def get_engine_path(self) -> str:
        """
        Returns
        -------
            A string indicating the path where the TRT LLM engines are stored with the tokenizer
        """
        return str(self._get_engine_path(self._trtllm_model_config_path))

    def mpi_run(self, server_config: TritonServerConfig) -> str:
        """
        Parameters
        ----------
        server_config : TritonServerConfig
            A TritonServerConfig object containing command-line arguments to run tritonserver
        Returns
        -------
            TRT LLM models must be run using MPI. This function constructs
            the appropriate mpi command to run a TRT LLM engine given a
            previously parsed world size.
        """
        unsupported_args = server_config.get_unsupported_args(self._supported_args)
        if unsupported_args:
            raise Exception(
                f"The following args are not currently supported by this model: {unsupported_args}"
            )

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

    def _get_engine_path(self, config_path: Path) -> Path:
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
                # FIXME: Revert handling using 'build_config' as the key when gpt migrates to using unified builder
                config = (
                    data.get("builder_config")
                    if data.get("builder_config") is not None
                    else data.get("build_config")
                )
                if not config:
                    raise Exception(f"Unable to parse config from {engine_config_path}")
                tp = int(config.get("tensor_parallel", 1))
                pp = int(config.get("pipeline_parallel", 1))
                return tp * pp
        except OSError:
            raise Exception(f"Unable to open {engine_config_path}")


class VLLMUtils:
    """
    A utility class for handling vLLM specific models.
    """

    def __init__(self, model_path: str):
        self._model_repo_path = model_path
        self._vllm_model_config_path = self._find_vllm_model_config_path()
        self._is_vllm_model = self._vllm_model_config_path is not None

    def has_vllm_model(self) -> bool:
        """
        Returns
        -------
            A boolean indicating whether a vLLM model exists in the model repo
        """
        return self._is_vllm_model

    def get_vllm_model_huggingface_id_or_path(self) -> str:
        """
        Returns
        -------
            The vLLM model's Huggingface Id or path
        """
        return self._find_vllm_model_huggingface_id_or_path()

    def _find_vllm_model_config_path(self) -> Path:
        """
        Returns
        -------
            A pathlib.Path object containing the path to the vLLM model config folder.
        Assumptions
        ----------
            - Assumes only a single model uses the vLLM backend (could have multiple models)
        """
        try:
            match = subprocess.check_output(
                [
                    "grep",
                    "-r",
                    "--include",
                    "*.pbtxt",
                    'backend: "vllm"',
                    self._model_repo_path,
                ]
            )
            # Example match: b'{PATH}/config.pbtxt:backend: "vllm"\n'
            return Path(match.decode().split(":")[0])
        except subprocess.CalledProcessError:
            # The 'grep' command will return a non-zero exit code
            # if no matches are found.
            return None

    def _find_vllm_model_huggingface_id_or_path(self) -> str:
        """
        Returns
        -------
            The vLLM model's Huggingface Id or path
        """
        assert self._is_vllm_model, "model Huggingface Id or path cannot be parsed from a model repository that does not contain a vLLM model."
        try:
            # assume the version is always "1"
            model_version_path = self._vllm_model_config_path.parent / "1"
            model_config_json_file = model_version_path / "model.json"
            with open(model_config_json_file) as json_data:
                data = json.load(json_data)
                model_id = data.get("model")
                if not model_id:
                    raise Exception(
                        f"Unable to parse config from {model_config_json_file}"
                    )
                return model_id
        except OSError:
            raise Exception(f"Unable to open {model_config_json_file}")
