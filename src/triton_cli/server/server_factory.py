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

import logging
import os
import shutil

from .server_local import TritonServerLocal
from .server_docker import TritonServerDocker
from .server_config import TritonServerConfig
from triton_cli.constants import (
    LOGGER_NAME,
    DEFAULT_SHM_SIZE,
    DEFAULT_TRITONSERVER_PATH,
)

logger = logging.getLogger(LOGGER_NAME)


class TritonServerFactory:
    """
    A factory for creating TritonServer instances
    """

    @staticmethod
    def create_server_docker(
        image,
        world_size,
        config,
        gpus,
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
        world_size : int
            Number of devices to deploy a tensorrtllm model.
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
            Use ["all"] to include all GPUs
        mounts: list of str
            The volumes to be mounted to the tritonserver container
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
            world_size=world_size,
            config=config,
            gpus=gpus,
            mounts=mounts,
            labels=labels,
            shm_size=shm_size,
            args=args,
        )

    @staticmethod
    def create_server_local(path, config, gpus):
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

        return TritonServerLocal(path=path, config=config, gpus=gpus)

    @staticmethod
    def get_server_handle(config, gpus):
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
            raise Exception(
                f"Unsupported triton-launch-mode : {config.triton_launch_mode}"
            )

        return server

    @staticmethod
    def _get_local_server_handle(config, gpus):
        tritonserver_path = DEFAULT_TRITONSERVER_PATH
        TritonServerFactory._validate_triton_server_path(tritonserver_path)

        triton_config = TritonServerConfig()
        triton_config["model-repository"] = config.model_repository
        logger.info("Starting a Triton Server locally")
        server = TritonServerFactory.create_server_local(
            path=tritonserver_path,
            config=triton_config,
            gpus=gpus,
        )

        return server

    @staticmethod
    def _get_docker_server_handle(config, gpus):
        triton_config = TritonServerConfig()
        triton_config["model-repository"] = os.path.abspath(config.model_repository)
        logger.info("Starting a Triton Server using docker")
        server = TritonServerFactory.create_server_docker(
            image=config.image,
            world_size=config.world_size,
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
            raise Exception(
                f"Either the binary {tritonserver_path} is invalid, not on the PATH, or does not have the correct permissions."
            )
