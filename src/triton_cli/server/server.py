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

from abc import ABC, abstractmethod


class TritonServer(ABC):
    """
    Defines the interface for the objects created by
    TritonServerFactory
    """

    @abstractmethod
    def start(self, env=None):
        """
        Starts the tritonserver

        Parameters
        ----------
        env: dict
            The environment to set for this tritonserver launch
        """

    @abstractmethod
    def stop(self):
        """
        Stops and cleans up after the server
        """

    @abstractmethod
    def logs(self):
        """
        Outputs and follows the server's log file in a blocking manner
        """

    @abstractmethod
    def health(self):
        """
        Checks server health. Raises an Exception on error.
        """

    def update_config(self, params):
        """
        Update the server's arguments

        Parameters
        ----------
        params: dict
            keys are argument names and values are their values.
        """

        self._server_config.update_config(params)

    def config(self):
        """
        Returns
        -------
        TritonServerConfig
            This server's config
        """

        return self._server_config
