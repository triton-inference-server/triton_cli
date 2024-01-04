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
import docker
from rich.progress import Progress

from .server import TritonServer
from .server_utils import TritonServerUtils
from triton_cli.constants import LOGGER_NAME, HF_CACHE

logger = logging.getLogger(LOGGER_NAME)


# Rich visualization of docker pull through API
# TODO: Revisit
def show_progress(line, progress, tasks):
    if line["status"] == "Downloading":
        id = f'[red][Downloading {line["id"]}]'
    elif line["status"] == "Extracting":
        id = f'[green][Extracting  {line["id"]}]'
    else:
        # skip other statuses
        return

    if id not in tasks.keys():
        tasks[id] = progress.add_task(f"{id}", total=line["progressDetail"]["total"])
    else:
        progress.update(tasks[id], completed=line["progressDetail"]["current"])


def image_pull(client, image):
    with Progress() as progress:
        status = client.api.pull(image, stream=True, decode=True)
        tasks = {}
        for line in status:
            show_progress(line, progress, tasks)


class TritonServerDocker(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    triton in a docker container.
    """

    def __init__(self, image, world_size, config, gpus, mounts, labels, shm_size, args):
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
        self._world_size = world_size

        assert self._server_config[
            "model-repository"
        ], "Triton Server requires --model-repository argument to be set."

        try:
            self._docker_client.images.get(self._tritonserver_image)
        except Exception:
            logger.info(f"Pulling docker image {self._tritonserver_image}")
            image_pull(self._docker_client, self._tritonserver_image)

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
        # Mount huggingface model cache to save time across runs
        # Use default cache in container for now.
        volumes[HF_CACHE] = {
            "bind": "/root/.cache/huggingface",
            "mode": "rw",
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
        # TRTLLM models require special handling. For now,
        # we will 'spell-out' the command.
        if self._world_size >= 1:
            command = " ".join(
                TritonServerUtils.mpi_run(
                    self._world_size, self._server_config["model-repository"]
                )
            )
        else:
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
