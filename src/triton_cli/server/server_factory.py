#!/usr/bin/env python3

# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil

from .server_local import TritonServerLocal
from .server_docker import TritonServerDocker
from .server_config import TritonServerConfig, TritonOpenAIServerConfig
from triton_cli.common import (
    DEFAULT_SHM_SIZE,
    LOGGER_NAME,
    TritonCLIException,
)
from .server_utils import TRTLLMUtils, VLLMUtils

logger = logging.getLogger(LOGGER_NAME)


class TritonServerFactory:
    """
    A factory for creating TritonServer instances
    """

    @staticmethod
    def create_server_docker(
        image,
        config,
        gpus=None,
        mounts=None,
        labels=None,
        shm_size=None,
        args=None,
    ):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
            Use ["all"] to include all GPUs
        mounts: list of str
            The volumes to be mounted to the tritonserver container.
        labels: dict
            name-value pairs for label to set metadata for triton docker
            container. (Not the same as environment variables)
        shm-size: str
            The size of /dev/shm for the triton docker container.
        args: dict
            name-value pairs of triton docker args
        Returns
        -------
        TritonServerDocker
        """

        return TritonServerDocker(
            image=image,
            config=config,
            gpus=gpus,
            mounts=mounts,
            labels=labels,
            shm_size=shm_size,
            args=args,
        )

    @staticmethod
    def create_server_local(config, gpus=None):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus: list of str
            List of GPU UUIDs to be made visible to Triton
            Use ["all"] to include all GPUs

        Returns
        -------
        TritonServerLocal
        """

        return TritonServerLocal(config=config, gpus=gpus)

    @staticmethod
    def get_server_handle(config, gpus=None):
        """
        Creates and returns a TritonServer
        with specified arguments

        Parameters
        ----------
        config : namespace
            Arguments parsed from the CLI
        gpus : list of str
            Available, supported, visible requested GPU UUIDs
        Returns
        -------
        TritonServer
            Handle to the Triton Server
        """

        if config.mode == "local":
            server = TritonServerFactory._get_local_server_handle(config, gpus)
        elif config.mode == "docker":
            server = TritonServerFactory._get_docker_server_handle(config, gpus)
        else:
            raise TritonCLIException(f"Unsupported triton-launch-mode : {config.mode}")

        return server

    @staticmethod
    def _get_local_server_handle(config, gpus):
        triton_config = TritonServerFactory._get_triton_server_config(config)
        TritonServerFactory._validate_triton_server_path(triton_config.server_path())

        server = TritonServerFactory.create_server_local(
            config=triton_config,
            gpus=gpus,
        )

        return server

    @staticmethod
    def _get_docker_server_handle(config, gpus):
        triton_config = TritonServerFactory._get_triton_server_config(config)

        server = TritonServerFactory.create_server_docker(
            image=config.image,
            config=triton_config,
            gpus=gpus,
            mounts=None,
            labels=None,
            shm_size=DEFAULT_SHM_SIZE,
            args=None,
        )

        return server

    @staticmethod
    def _validate_triton_server_path(tritonserver_path):
        """
        Raises an exception if tritonserver binary isn't found at path
        """
        # Determine if the binary is valid and executable
        if not shutil.which(tritonserver_path):
            raise TritonCLIException(
                f"Either the binary {tritonserver_path} is invalid, not on the PATH, or does not have the correct permissions."
            )

    @staticmethod
    def _get_triton_server_config(config):
        if config.frontend == "openai":
            triton_config = TritonOpenAIServerConfig()
            triton_config["model-repository"] = config.model_repository

            triton_config["tokenizer"] = (
                TritonServerFactory._get_openai_chat_template_tokenizer(config)
            )

            if config.verbose:
                triton_config["tritonserver-log-verbose-level"] = "1"
        else:
            triton_config = TritonServerConfig()
            triton_config["model-repository"] = config.model_repository
            if config.verbose:
                triton_config["log-verbose"] = "1"

        return triton_config

    @staticmethod
    def _get_openai_chat_template_tokenizer(config):
        """
        Raises an exception if a tokenizer can not be found and is not specified with OpenAI Frontend
        """
        if config.openai_chat_template_tokenizer:
            return config.openai_chat_template_tokenizer

        logger.info(
            "OpenAI frontend's tokenizer for chat template is not specify, searching for an available tokenizer in the model repository."
        )
        trtllm_utils = TRTLLMUtils(config.model_repository)
        vllm_utils = VLLMUtils(config.model_repository)

        if trtllm_utils.has_trtllm_model():
            tokenizer_path = trtllm_utils.get_engine_path()
        elif vllm_utils.has_vllm_model():
            tokenizer_path = vllm_utils.get_vllm_model_huggingface_id_or_path()
        else:
            raise TritonCLIException(
                "Unable to find a tokenizer to start the Triton OpenAI RESTful API, please use '--openai-chat-template-tokenizer' to specify a tokenizer."
            )

        logger.info(
            f"Found tokenizer in '{tokenizer_path}' after searching for the tokenizer in the model repository"
        )
        return tokenizer_path
