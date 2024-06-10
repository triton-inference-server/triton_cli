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

import os
import logging
from subprocess import STDOUT, PIPE, Popen, TimeoutExpired
from .server_utils import TritonServerUtils

from .server import TritonServer

from triton_cli.common import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
SERVER_OUTPUT_TIMEOUT_SECS = 30


class TritonServerLocal(TritonServer):
    """
    Concrete Implementation of TritonServer interface that runs
    tritonserver locally as as subprocess.
    """

    def __init__(self, path, config, gpus):
        """
        Parameters
        ----------
        path  : str
            The absolute path to the tritonserver executable
        config : TritonServerConfig
            the config object containing arguments for this server instance
        gpus: list of str
            List of GPU UUIDs to be made visible to Triton
        """

        self._tritonserver_process = None
        self._server_config = config
        self._server_path = path
        self._gpus = gpus
        self._is_first_time_starting_server = True

        assert self._server_config[
            "model-repository"
        ], "Triton Server requires --model-repository argument to be set."

        self._server_utils = TritonServerUtils(self._server_config["model-repository"])

    def start(self, env=None):
        """
        Starts the tritonserver container locally
        """

        logger.info(
            f"Starting a Triton Server locally with model repository: {self._server_config['model-repository']}"
        )

        if self._server_path:
            # Get the appropriate server launch command
            cmd = self._server_utils.get_launch_command(
                tritonserver_path=self._server_path,
                server_config=self._server_config,
                cmd_as_list=True,
            )

            # Set environment, update with user config env
            triton_env = os.environ.copy()

            if env:
                # Filter env variables that use env lookups
                for variable, value in env.items():
                    if value.find("$") == -1:
                        triton_env[variable] = value
                    else:
                        # Collect the ones that need lookups to give to the shell
                        triton_env[variable] = os.path.expandvars(value)

            # List GPUs to be used by tritonserver
            if self._gpus:
                raise Exception(
                    "GPUs aren't configurable at this time, leave it unspecified."
                )

            self._is_first_time_starting_server = False

            # Construct Popen command
            try:
                self._tritonserver_process = Popen(
                    cmd,
                    stdout=PIPE,
                    stderr=STDOUT,
                    start_new_session=True,
                    universal_newlines=True,
                    env=triton_env,
                )
            except Exception as e:
                logger.error(e)
                raise Exception(e)

    def stop(self):
        """
        Stops the running tritonserver
        """

        # Terminate process, capture output
        if self._tritonserver_process is not None:
            self._tritonserver_process.terminate()
            try:
                self._tritonserver_process.communicate(
                    timeout=SERVER_OUTPUT_TIMEOUT_SECS
                )
            except TimeoutExpired:
                self._tritonserver_process.kill()
                self._tritonserver_process.communicate()
            self._tritonserver_process = None
            logger.debug("Stopped Triton Server.")

    def logs(self):
        for line in self._tritonserver_process.stdout:
            print(line.rstrip())

    def health(self):
        status = self._tritonserver_process.returncode
        stderr = self._tritonserver_process.stderr
        if status:
            raise Exception(
                f"Triton server experienced an error. Status: {status}\nStderr: {stderr}"
            )
