#!/usr/bin/env python3

# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
import logging
import docker

from .server import TritonServer
from .server_utils import TritonServerUtils
from triton_cli.common import (
    HF_CACHE,
    DEFAULT_TRITONSERVER_IMAGE,
    LOGGER_NAME,
)


logger = logging.getLogger(LOGGER_NAME)


def docker_pull(image):
    logger.info(f"Pulling docker image: {image}")
    cmd = f"docker pull {image}"
    output = subprocess.run(cmd.split())
    if output.returncode:
        err = output.stderr.decode("utf-8")
        raise Exception(f"Failed to pull docker image: {image}:\n{err}")


# TODO: See if there is a way to remove or hide the build output after it
# successfully completes, similar to 'transient' in rich progress bars.
def docker_build(path, image):
    cmd = f"docker build -t {image} {path}"
    logger.debug(f"Running '{cmd}'")
    output = subprocess.run(cmd.split())
    if output.returncode:
        err = output.stderr.decode("utf-8")
        raise Exception(f"Failed to build docker image at: {path}:\n{err}")


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, image, config, gpus, mounts, labels, shm_size, args):
        """
        Parameters
        ----------
        image : str
            The tritonserver docker image to pull and run
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus : list of str
            List of GPU UUIDs to be mounted and used in the container
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
        self._mounts = mounts
        # NOTE: Could use labels to determine containers started/owned by CLI
        self._labels = labels if labels else {}
        self._gpus = gpus
        self._shm_size = shm_size
        self._args = args if args else {}

        assert self._server_config[
            "model-repository"
        ], "Triton Server requires --model-repository argument to be set."

        self._server_utils = TritonServerUtils(self._server_config["model-repository"])

        # TODO: Always build for now as it's unclear how to detect changes.
        # Iterative runs should be cached and quick after first build.
        if not self._tritonserver_image:
            self._tritonserver_image = DEFAULT_TRITONSERVER_IMAGE
            logger.debug(
                f"No image specified, using custom image: {self._tritonserver_image}"
            )
            dir_path = os.path.dirname(os.path.realpath(__file__))
            docker_path = os.path.join(dir_path, "..", "docker")
            docker_build(docker_path, self._tritonserver_image)

        try:
            self._docker_client.images.get(self._tritonserver_image)
        except Exception:
            docker_pull(self._tritonserver_image)

    def start(self, env=None):
        """
        Starts the tritonserver docker container using docker-py
        """

        logger.info(
            f"Starting a Triton Server via docker image '{self._tritonserver_image}' with model repository: {self._server_config['model-repository']}"
        )

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
        volumes[str(self._server_config["model-repository"])] = {
            "bind": str(self._server_config["model-repository"]),
            "mode": "ro",
        }
        # Mount huggingface model cache to save time across runs
        # Use default cache in container for now.
        volumes[str(HF_CACHE)] = {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
        }
        # Mount /tmp for convenience. For example, TRT-LLM or NGC assets
        # may default to living in /tmp.
        volumes["/tmp"] = {
            "bind": "/tmp",
            "mode": "rw",
        }

        # Map ports, use config values but set to server defaults if not
        # specified
        server_http_port = 8000
        server_grpc_port = 8001
        server_metrics_port = 8002
        openai_http_port = 9000

        ports = {
            server_http_port: server_http_port,
            server_grpc_port: server_grpc_port,
            server_metrics_port: server_metrics_port,
            openai_http_port: openai_http_port,
        }
        # Construct run command
        command = self._server_utils.get_launch_command(
            server_config=self._server_config,
            cmd_as_list=False,
            env_cmds=env_cmds,
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
        except docker.errors.APIError as e:
            if e.explanation.find("port is already allocated") != -1:
                raise Exception(
                    "One of the following port(s) are already allocated: "
                    f"{server_http_port}, {server_grpc_port}, "
                    f"{server_metrics_port}."
                )
            else:
                raise Exception(e)

    def stop(self):
        """
        Stops the tritonserver docker container
        and cleans up docker client
        """

        if self._tritonserver_container is not None:
            self._tritonserver_container.stop()
            self._tritonserver_container.remove(force=True)
            self._tritonserver_container = None
            logger.info("Stopped Triton Server.")
        self._docker_client.close()

    def logs(self):
        for chunk in self._tritonserver_container.logs(stream=True):
            print(chunk.decode("utf-8").rstrip())

    def health(self):
        # Local attrs are cached, need to call reload() to update them
        self._tritonserver_container.reload()
        status = self._tritonserver_container.status
        if status not in ["created", "running"]:
            logs = self._tritonserver_container.logs(stream=False).decode("utf-8")
            raise Exception(
                f"Triton server experienced an error. Status: {status}\nLogs: {logs}"
            )
