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

import tempfile
from io import TextIOWrapper
from multiprocessing.pool import ThreadPool
from subprocess import DEVNULL

import docker

from .server import TritonServer

import logging
from constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, image, config, gpus, log_path, mounts, labels, shm_size, args):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
        log_path: str
            Absolute path to the triton log file
        mounts: list of str
            The volumes to be mounted to the tritonserver container
        labels: dict
            name-value pairs for label to set metadata for triton docker
            container. (Not the same as environment variables)
        shm-size: str
            The size of /dev/shm for the triton docker container.
        args: dict
            name-values part for triton docker args
        """

        self._server_config = config
        self._docker_client = docker.from_env()
        self._tritonserver_image = image
        self._tritonserver_container = None
        self._log_path = log_path
        self._log_file = DEVNULL
        self._mounts = mounts
        self._labels = labels if labels else {}
        self._gpus = gpus
        self._shm_size = shm_size
        self._args = args if args else {}

        assert self._server_config[
            "model-repository"
        ], "Triton Server requires --model-repository argument to be set."

        try:
            self._docker_client.images.get(self._tritonserver_image)
        except Exception:
            logger.info(f"Pulling docker image {self._tritonserver_image}")
            self._docker_client.images.pull(self._tritonserver_image)

    def start(self, env=None):
        """
        Starts the tritonserver docker container using docker-py
        """

        # Use "all" gpus by default. Can be more configurable in the future.
        devices = [
            docker.types.DeviceRequest(
                count=-1,  # use all gpus
                capabilities=[["gpu"]],
            )
        ]

        # Set environment inside container.
        env_cmds = []
        # Mount required directories
        volumes = {}
        # Mount model repository at same path in read-only mode for simplicity
        volumes[self._server_config["model-repository"]] = {
            "bind": self._server_config["model-repository"],
            "mode": "ro",
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        server_http_port = 8000
        server_grpc_port = 8001
        server_metrics_port = 8002

        ports = {
            server_http_port: server_http_port,
            server_grpc_port: server_grpc_port,
            server_metrics_port: server_metrics_port,
        }

        # Construct run command
        command = " ".join(
            env_cmds + ["tritonserver", self._server_config.to_cli_string()]
        )
        try:
            # Run the docker container and run the command in the container
            self._tritonserver_container = self._docker_client.containers.run(
                command=f'bash -c "{command}"',
                init=True,
                image=self._tritonserver_image,
                device_requests=devices,
                volumes=volumes,
                labels=self._labels,
                ports=ports,
                publish_all_ports=True,
                tty=False,
                stdin_open=False,
                detach=True,
                shm_size=self._shm_size,
                **self._args,
            )
            logger.info("Triton Server started.")
        except docker.errors.APIError as e:
            if e.explanation.find("port is already allocated") != -1:
                raise Exception(
                    "One of the following port(s) are already allocated: "
                    f"{server_http_port}, {server_grpc_port}, "
                    f"{server_metrics_port}.\n"
                    "Change the Triton server ports using"
                    " --triton-http-endpoint, --triton-grpc-endpoint,"
                    " and --triton-metrics-endpoint flags."
                )
            else:
                raise Exception(e)

        if self._log_path:
            try:
                self._log_file = open(self._log_path, "a+")
                self._log_pool = ThreadPool(processes=1)
                self._log_pool.apply_async(self._logging_worker)
            except OSError as e:
                raise Exception(e)
        else:
            self._log_file = tempfile.NamedTemporaryFile()

    def _logging_worker(self):
        """
        streams logs to
        log file
        """

        for chunk in self._tritonserver_container.logs(stream=True):
            self._log_file.write(chunk.decode("utf-8"))

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """

        if self._tritonserver_container is not None:
            if self._log_path:
                if self._log_pool:
                    self._log_pool.terminate()
                    self._log_pool.close()
                if self._log_file:
                    self._log_file.close()
            self._tritonserver_container.stop()
            self._tritonserver_container.remove(force=True)
            self._tritonserver_container = None
            logger.info("Stopped Triton Server.")
        self._docker_client.close()

    def log_file(self) -> TextIOWrapper:
        return self._log_file

    def logs(self):
        for chunk in self._tritonserver_container.logs(stream=True):
            print(chunk.decode("utf-8").rstrip())
