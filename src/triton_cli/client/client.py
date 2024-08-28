# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import queue
import logging
import numpy as np
from functools import partial

import tritonclient.http
import tritonclient.grpc
from tritonclient.utils import triton_to_np_dtype, InferenceServerException

from triton_cli.common import TritonCLIException, LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class TritonClient:
    def __init__(self, url="localhost", port=None, protocol="grpc"):
        self.protocol = protocol
        if not url:
            url = "localhost"

        if self.protocol == "grpc":
            if not port:
                port = 8001
            self.url = f"{url}:{port}"
            self.client = tritonclient.grpc.InferenceServerClient(self.url)
            self.kwargs = {"as_json": True}
        elif self.protocol == "http":
            if not port:
                port = 8000
            self.url = f"{url}:{port}"
            self.client = tritonclient.http.InferenceServerClient(self.url)
            self.kwargs = {}
        else:
            raise TritonCLIException(
                f"Unsupported protocol passed to TritonClient: '{protocol}'"
            )

    def get_model_config(self, model: str):
        if self.protocol == "grpc":
            config = self.client.get_model_config(model_name=model, **self.kwargs)[
                "config"
            ]
        else:
            config = self.client.get_model_config(model_name=model, **self.kwargs)
        return config

    def get_model_backend(self, model: str):
        config = self.get_model_config(model)
        return config.get("backend")

    def get_server_metadata(self):
        return self.client.get_server_metadata(**self.kwargs)

    def is_server_ready(self):
        return self.client.is_server_ready()

    def is_server_live(self):
        return self.client.is_server_live()

    def ready_for_inference(self):
        return self.client.is_server_live() and self.client.is_server_ready()

    def get_server_health(self):
        live = self.is_server_live()
        ready = self.is_server_ready()
        return {"live": live, "ready": ready}

    def __parse_input(self, _input: dict):
        name = _input.get("name", "")
        dtype = _input.get("data_type", "")
        dims = _input.get("dims", [])
        optional = _input.get("optional", False)
        # TODO: Expose configurable shapes on CLI
        shape = [int(dim) if int(dim) > 1 else 1 for dim in dims]
        return name, dtype, shape, optional

    def __generate_scalar_data(self, shape, np_dtype):
        # TODO: Expose as CLI option
        scalar_value = 0
        data = np.ones(shape, dtype=np_dtype) * scalar_value
        return data

    def __generate_random_data(self, shape, np_dtype):
        if np_dtype in [np.float16, np.float32, np.float64]:
            data = np.random.random(shape).astype(np_dtype)
        elif np_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            data = np.random.randint(0, 255, size=shape, dtype=np_dtype)
        elif np_dtype in [np.int8, np.int16, np.int32, np.int64]:
            # WAR: TRT-LLM uses some signed integers for inputs that don't
            # support negative values, so clamp it to non-negative values.
            # Can be set back to (-127, 128) if types are updated to UINT.
            data = np.random.randint(0, 128, size=shape, dtype=np_dtype)
        else:  # bool or object
            data = np.random.randint(0, 2, size=shape, dtype=np_dtype)
        return data

    def __generate_llm_data(self, name, shape, np_dtype):
        if name.lower() in ["prompt", "text", "text_input"]:
            if not self.prompt:
                raise TritonCLIException(
                    f"LLM input '{name}' detected, but no prompt provided. Please pass '--prompt' to specify this input."
                )
            data = np.full(shape, self.prompt, dtype=np_dtype)
        # vLLM fields
        elif name.lower() == "sampling_parameters":
            parameters = {}
            data = np.full(shape, json.dumps(parameters), dtype=np_dtype)
        elif name.lower() == "stream":
            data = np.zeros(shape, dtype=np_dtype)  # False
        # TRT-LLM fields
        elif name.lower() == "max_tokens":
            default_max_tokens = 128
            data = np.full(shape, default_max_tokens, dtype=np_dtype)
        elif name.lower() in ["bad_words", "stop_words"]:
            data = np.full(shape, "", dtype=np_dtype)
        else:
            data = None

        return data

    def __create_triton_input(self, name, shape, dtype, data):
        if self.protocol == "grpc":
            _input = tritonclient.grpc.InferInput(name, shape, dtype)
        else:
            _input = tritonclient.http.InferInput(name, shape, dtype)
        _input.set_data_from_numpy(data)
        return _input

    def generate_data(self, config: dict, data_mode: str):
        logger.debug("Generating input data...")
        inputs = [i for i in config["input"]]

        infer_inputs = []
        for i in inputs:
            name, dtype, shape, optional = self.__parse_input(i)
            # NOTE: Config returns BYTES, Metadata returns STRING. We should
            # make this easier or more consistent.
            triton_dtype = dtype.replace("TYPE_", "").replace("STRING", "BYTES")
            np_dtype = triton_to_np_dtype(triton_dtype)
            if not np_dtype:
                raise TritonCLIException(
                    f"Failed to convert {triton_dtype=} to numpy equivalent"
                )

            # Skip optional inputs for now
            if optional:
                logger.warning(f"Skipping optional input '{name}'")
                continue

            # If batching is enabled, pad front of shape with batch dimension
            if config.get("max_batch_size", 0) > 0:
                shape = (1, *shape)

            # LLM convenience WAR
            # TODO: Move to customized 'triton llm infer' subcommand or similar
            data = self.__generate_llm_data(name, shape, np_dtype)

            # Standard flow
            if not data:
                if data_mode == "random":
                    data = self.__generate_random_data(shape, np_dtype)
                elif data_mode == "scalar":
                    data = self.__generate_scalar_data(shape, np_dtype)
                else:
                    raise TritonCLIException(
                        f"Unsupported data mode for infer: {data_mode}"
                    )

            # Form tritonclient input
            infer_inputs.append(
                self.__create_triton_input(name, shape, triton_dtype, data)
            )
            json_input = {
                "name": name,
                "shape": str(data.shape),
                "dtype": triton_dtype,
                "value": np.array_str(data),
            }
            logger.info(f"Input:\n{json.dumps(json_input, indent=4)}")

        return infer_inputs

    # TODO: Add specialized 'triton llm infer' subcommand for LLM handling
    # TODO: enum for data mode
    def infer(self, model: str, data_mode: str = "random", prompt: str = None):
        self.prompt = prompt
        config = self.get_model_config(model)
        inputs = self.generate_data(config, data_mode)
        logger.info("Sending inference request...")
        self.__async_infer(model, inputs)

    def __async_infer(self, model: str, inputs):
        if self.protocol == "grpc":
            self.__grpc_async_infer(model, inputs)
        else:
            self.__http_async_infer(model, inputs)

    def __http_async_infer(self, model: str, inputs):
        infer = partial(
            self.client.async_infer,
            model,
            inputs,
        )
        future = infer()
        result = future.get_result()
        self.__process_infer_result(result)

    def __grpc_async_infer(self, model: str, inputs):
        assert self.protocol == "grpc"

        class UserData:
            def __init__(self):
                self._completed_requests = queue.Queue()

        def callback(user_data, result, error):
            if error:
                user_data._completed_requests.put(error)
            else:
                user_data._completed_requests.put(result)

        user_data = UserData()
        logger.debug("Starting stream...")
        self.client.start_stream(callback=partial(callback, user_data))

        # Enable empty final response to always to get final flags for tracking
        infer = partial(
            self.client.async_stream_infer,
            model_name=model,
            inputs=inputs,
            enable_empty_final_response=True,
        )
        infer()

        num_requests = 1
        completed_requests = 0
        try:
            while completed_requests != num_requests:
                result = user_data._completed_requests.get()
                if isinstance(result, InferenceServerException):
                    if result.status() == "StatusCode.CANCELLED":
                        is_final_response = True
                        logger.warning(
                            "Request cancelled, marking this request complete"
                        )
                    else:
                        raise result
                else:
                    is_final_response = self.__process_infer_result(result)

                logger.debug(f"{is_final_response=}")
                if is_final_response:
                    completed_requests += 1
        # Gracefully handle cancelling inference and stopping stream
        except KeyboardInterrupt:
            pass

        logger.debug("Stopping stream...")
        self.client.stop_stream(cancel_requests=True)

    def __process_infer_result(self, result):
        response = result.get_response(**self.kwargs)
        # Detect final response. Parameters are oneof and we expect bool_param
        params = response.get("parameters")
        # Final response by default for non-decoupled models
        is_final_response = True
        # Otherwise, decoupled models will return a response parameter indicating finality
        if params:
            is_final_response = params.get("triton_final_response", {}).get(
                "bool_param", True
            )

        # Process response outputs
        if response.get("outputs"):
            for output in response["outputs"]:
                name = output["name"]
                # TODO: Need special logic for string/bytes type
                data = result.as_numpy(name)
                # WAR for LLMs
                if data.dtype == np.object_:
                    # Assume 2D-output (batch_size, texts)
                    texts = data.flatten()
                    data = np.array([text.decode("utf-8") for text in texts])

                output_data_str = np.array_str(data)
                json_output = {
                    "name": name,
                    "shape": str(data.shape),
                    "dtype": output["datatype"],
                    "value": output_data_str,
                }
                logger.info(f"Output:\n{json.dumps(json_output, indent=4)}")

        # Used for decoupled purposes to determine when requests are finished
        # Only applicable to GRPC streaming at this time
        return is_final_response
