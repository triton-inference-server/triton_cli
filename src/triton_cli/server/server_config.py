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

from triton_cli.common import (
    DEFAULT_TRITONSERVER_PATH,
    DEFAULT_TRITONSERVER_OPENAI_FRONTEND_PATH,
)


class TritonServerConfig:
    """
    A config class to set arguments to the Triton Inference
    Server. An argument set to None will use the server default.
    """

    server_arg_keys = [
        # Logging
        "log-verbose",
        "log-info",
        "log-warning",
        "log-error",
        "id",
        # Model Repository
        "model-store",
        "model-repository",
        # Exit
        "exit-timeout-secs",
        "exit-on-error",
        # Strictness
        "strict-model-config",
        "strict-readiness",
        # API Servers
        "allow-http",
        "http-port",
        "http-thread-count",
        "allow-grpc",
        "grpc-port",
        "grpc-infer-allocation-pool-size",
        "grpc-use-ssl",
        "grpc-use-ssl-mutual",
        "grpc-server-cert",
        "grpc-server-key",
        "grpc-root-cert",
        "allow-metrics",
        "allow-gpu-metrics",
        "metrics-interval-ms",
        "metrics-port",
        # Tracing
        "trace-file",
        "trace-level",
        "trace-rate",
        # Model control
        "model-control-mode",
        "repository-poll-secs",
        "load-model",
        # Memory and GPU
        "pinned-memory-pool-byte-size",
        "cuda-memory-pool-byte-size",
        "min-supported-compute-capability",
        # Backend config
        "backend-directory",
        "backend-config",
        "allow-soft-placement",
        "gpu-memory-fraction",
        "tensorflow-version",
    ]

    def __init__(self, server_path=None):
        """
        Construct TritonServerConfig

        Parameters
        ----------
        server_path: string
            path to the triton server binary. Default is "tritonserver" if unset.
        """

        self._server_args = {k: None for k in self.server_arg_keys}
        self._server_path = server_path if server_path else DEFAULT_TRITONSERVER_PATH
        self._server_name = "Triton Inference Server"

    @classmethod
    def allowed_keys(cls):
        """
        Returns
        -------
        list of str
            The keys that can be used to configure tritonserver instance
        """

        snake_cased_keys = [key.replace("-", "_") for key in cls.server_arg_keys]
        return cls.server_arg_keys + snake_cased_keys

    def update_config(self, params=None):
        """
        Allows setting values from a
        params dict

        Parameters
        ----------
        params: dict
            keys are allowed args to tritonserver
        """

        if params:
            for key in params:
                self[key.strip().replace("_", "-")] = params[key]

    def to_cli_string(self):
        """
        Utility function to convert a config into a
        string of arguments to the server with CLI.

        Returns
        -------
        str
            the command consisting of all set arguments to
            the tritonserver.
            e.g. '--model-repository=/models --log-verbose=True'
        """

        return " ".join(
            [f"--{key}={val}" for key, val in self._server_args.items() if val]
        )

    def to_args_list(self):
        """
        Utility function to convert a cli string into a list of arguments while
        taking into account "smart" delimiters.  Notice in the example below
        that only the first equals sign is used as split delimiter.

        Returns
        -------
        list
            the list of arguments consisting of all set arguments to
            the tritonserver.

            Example:
            input cli_string: "--model-control-mode=explicit
                --backend-config=tensorflow,version=2"

            output: ['--model-control-mode', 'explicit',
                '--backend-config', 'tensorflow,version=2']
        """
        args_list = []
        args = self.to_cli_string().split()
        for arg in args:
            args_list += arg.split("=", 1)
        return args_list

    def copy(self):
        """
        Returns
        -------
        TritonServerConfig
            object that has the same args as this one
        """

        config_copy = TritonServerConfig()
        config_copy.update_config(params=self._server_args)
        return config_copy

    def server_args(self):
        """
        Returns
        -------
        dict
            keys are server arguments
            values are their values
        """

        return self._server_args

    def server_path(self) -> str:
        """
        Returns
        -------
        str
            A path to the triton server binary or script
        """

        return self._server_path

    # TODO: Investigate what parameters are supported with TRT LLM's launching style.
    # For example, explicit launch mode is not. See the TRTLLMUtils class for a list of
    # supported args.
    def get_unsupported_args(self, supported_args: list) -> list:
        """
        Parameters
        -------
        supported_args : list
            A list of supported config arguments
        Returns
        -------
            A list of specified args that are unsupported
        """
        unsupported_args = []
        for key, val in self._server_args.items():
            if val and key not in supported_args:
                unsupported_args.append(key)
        return unsupported_args

    def __getitem__(self, key):
        """
        Gets an arguments value in config

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver

        Returns
        -------
            The value that the argument is set to in this config
        """

        return self._server_args[key.strip().replace("_", "-")]

    def __setitem__(self, key, value):
        """
        Sets an arguments value in config
        after checking if defined/supported.

        Parameters
        ----------
        key : str
            The name of the argument to the tritonserver
        value : (any)
            The value to which the argument is being set

        Raises
        ------
        Exception
            If key is unsupported or undefined in the
            config class
        """

        kebab_cased_key = key.strip().replace("_", "-")
        if kebab_cased_key in self._server_args:
            self._server_args[kebab_cased_key] = value
        else:
            raise Exception(
                f"The argument '{key}' to the {self._server_name}"
                " is not currently supported."
            )


class TritonOpenAIServerConfig(TritonServerConfig):
    """
    A config class to set arguments to the Triton Inference
    Server with OpenAI RESTful API. An argument set to None will use the server default.
    """

    server_arg_keys = [
        # triton server args
        "tritonserver-log-verbose-level",
        "host",
        "backend",
        "tokenizer",
        "model-repository",
        # uvicorn args
        "openai-port",
        "uvicorn-log-level",
        # kserve frontend args
        "enable-kserve-frontends",
        "kserve-http-port",
        "kserve-grpc-port",
    ]

    def __init__(self, server_path=None):
        """
        Construct TritonOpenAIServerConfig

        Parameters
        ----------
        server_path: string
            path to the Triton OpenAI Server python script. Default is "/opt/tritonserver/python/openai/openai_frontend/main.py" if unset.
        """

        self._server_args = {k: None for k in self.server_arg_keys}
        self._server_path = (
            server_path if server_path else DEFAULT_TRITONSERVER_OPENAI_FRONTEND_PATH
        )
        self._server_name = "Triton Inference Server with OpenAI RESTful API"
