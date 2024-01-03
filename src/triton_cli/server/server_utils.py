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

import logging
import json
from pathlib import Path
import os
import tritonclient.grpc.model_config_pb2 as mc
from google.protobuf import json_format, text_format

from triton_cli.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonServerUtils:
    """
    A utility class for launching a Triton server.
    """

    @staticmethod
    def is_trtllm_model(model_repo: str) -> bool:
        """
        Parameters
        ----------
        model_repo : str
            The path to the model repository
        Returns
        -------
            Whether the model repository contains a model
            using the tensorrtllm backend.
        Assumptions
        ----------
            - Assumes a TRT LLM model would have a tensorrt_llm folder.
        """
        for _, dirs, _ in os.walk(model_repo, topdown=True):
            if "tensorrt_llm" in dirs:
                return True
        return False

    @staticmethod
    def get_engine_path(config_path: str) -> str:
        """
        Parameters
        ----------
        config_path : str
            The path to the tensorrt_llm model's config.pbtxt
            file.
        Returns
        -------
            The path to the TRT LLM engine.
        Assumptions
        ----------
            - Assumes model will be stored at env variable NGC_DEST_DIR
        """
        try:
            config_file = open(config_path)
        except OSError:
            raise Exception(
                f"Failed to open config file for tensorrt_llm. Searched: {config_path}"
            )
        config = text_format.Parse(config_file.read(), mc.ModelConfig())
        json_config = json.loads(
            json_format.MessageToJson(config, preserving_proto_field_name=True)
        )
        try:
            return json_config["parameters"]["gpt_model_path"]["string_value"]
        except KeyError as e:
            raise Exception(
                f"Unable to extract engine path from config file {config_path}. Key error: {str(e)}"
            )

    @staticmethod
    def parse_world_size(model_repo: str) -> int:
        """
        Parameters
        ----------
        config_file_path : str
            The path to the model repository.
        Returns
        -------
            The appropriate world size to use to run the tensorrtllm
            engine(s) stored in the model repository
        Assumptions
        ----------
            - Assumes a TRT LLM model would have a tensorrt_llm folder.
            - Assumes only a single TRT LLM model will be launched (could have
            multiple engines)
        """
        triton_config_path = Path(model_repo) / "tensorrt_llm" / "config.pbtxt"
        # Helper to find model path from triton config file instead
        # of having to specify a model name at the cmdline.
        model_path = TritonServerUtils.get_engine_path(triton_config_path)
        model_config_path = Path(model_path) / "config.json"
        try:
            with open(model_config_path) as json_data:
                data = json.load(json_data)
                try:
                    return int(data["builder_config"]["tensor_parallel"])
                except KeyError as e:
                    raise Exception(
                        f"Unable to extract world size from {model_config_path}. Key error: {str(e)}"
                    )
        except OSError:
            raise Exception(f"Unable to open {model_config_path}")

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
