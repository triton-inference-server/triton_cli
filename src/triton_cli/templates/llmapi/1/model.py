# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import gc
import json
import os
import queue
import threading
from contextlib import asynccontextmanager

import numpy as np
import triton_python_backend_utils as pb_utils
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.llmapi import LLM

_TRTLLM_ENGINE_ARGS_FILENAME = "model.json"


class TritonPythonModel:
    # Define the expected keys for each config
    # TODO: Add more keys as needed
    PYTORCH_CONFIG_KEYS = {
        "use_cuda_graph",
        "cuda_graph_batch_sizes",
        "cuda_graph_max_batch_size",
        "cuda_graph_padding_enabled",
        "enable_overlap_scheduler",
        "kv_cache_dtype",
        "torch_compile_enabled",
        "torch_compile_fullgraph",
        "torch_compile_inductor_enabled",
    }

    LLM_ENGINE_KEYS = {
        "model",
        "tokenizer",
        "tokenizer_mode",
        "skip_tokenizer_init",
        "trust_remote_code",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "dtype",
        "revision",
        "tokenizer_revision",
        "speculative_model",
        "enable_chunked_prefill",
    }

    def _get_input_scalar_by_name(self, request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return None

        tensor = tensor.as_numpy()
        if tensor.size == 0:
            return None

        return tensor.item(0)

    def _get_string_list_by_name(self, request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            return None

        tensor = tensor.as_numpy()
        if tensor.size == 0:
            return None

        # Convert to list and handle bytes conversion
        if isinstance(tensor, np.ndarray):
            if tensor.ndim == 0:
                item = tensor.item()
                return [item.decode("utf-8") if isinstance(item, bytes) else str(item)]

            return [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in tensor.flatten()
            ]

        # Fallback case
        if isinstance(tensor, bytes):
            return [tensor.decode("utf-8")]
        return [str(tensor)]

    def _get_sampling_config_from_request(self, request):
        # TODO: Add more sampling parameters as needed
        kwargs = {
            "beam_width": self._get_input_scalar_by_name(request, "beam_width") or 1,
            "temperature": self._get_input_scalar_by_name(request, "temperature"),
            "top_k": self._get_input_scalar_by_name(request, "top_k"),
            "top_p": self._get_input_scalar_by_name(request, "top_p"),
            "frequency_penalty": self._get_input_scalar_by_name(
                request, "frequency_penalty"
            ),
            "presence_penalty": self._get_input_scalar_by_name(
                request, "presence_penalty"
            ),
            "max_tokens": self._get_input_scalar_by_name(request, "max_tokens"),
            # stop_words is deprecated. Should use stop instead.
            "stop": (
                self._get_string_list_by_name(request, "stop")
                or self._get_string_list_by_name(request, "stop_words")
            ),
            # random_seed is deprecated. Should use seed instead.
            "seed": (
                self._get_input_scalar_by_name(request, "seed")
                or self._get_input_scalar_by_name(request, "random_seed")
            ),
        }

        # Adjust top_p if it's not valid
        kwargs["top_p"] = (
            None if kwargs["top_p"] is None or kwargs["top_p"] <= 0 else kwargs["top_p"]
        )

        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return kwargs

    @classmethod
    def auto_complete_config(cls, auto_complete_model_config):
        # Add inputs/outputs to the model config.
        cls._auto_complete_inputs_and_outputs(auto_complete_model_config)

        # Get the max batch size and decoupled model transaction policy from the json file.
        engine_args_filepath = os.path.join(
            pb_utils.get_model_dir(), _TRTLLM_ENGINE_ARGS_FILENAME
        )
        assert os.path.isfile(
            engine_args_filepath
        ), f"'{_TRTLLM_ENGINE_ARGS_FILENAME}' containing TRT-LLM engine args must be provided in '{pb_utils.get_model_dir()}'"
        with open(engine_args_filepath) as file:
            # The Python interpreter used to invoke this function will be destroyed upon returning from this function and as a result none of the objects created here will be available in the initialize, execute, or finalize functions.
            trtllm_engine_config = json.load(file)

        model_config_keys = {"max_batch_size", "decoupled"}
        auto_complete_config = {
            k: v for k, v in trtllm_engine_config.items() if k in model_config_keys
        }

        # Set the max batch size and decoupled model transaction policy in the model config.
        is_decoupled = auto_complete_config.get("decoupled", False)
        auto_complete_model_config.set_model_transaction_policy(
            dict(decoupled=is_decoupled)
        )
        max_batch_size = auto_complete_config.get("max_batch_size", 64)
        auto_complete_model_config.set_max_batch_size(int(max_batch_size))

        return auto_complete_model_config

    @staticmethod
    def _auto_complete_inputs_and_outputs(auto_complete_model_config):
        # Inputs expected by the backend.
        inputs = [
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
            {
                "name": "stream",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "exclude_input_in_output",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_finish_reason",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "return_stop_reason",
                "data_type": "TYPE_BOOL",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "temperature",
                "data_type": "TYPE_FP32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "beam_width",
                "data_type": "TYPE_INT32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "top_k",
                "data_type": "TYPE_INT32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "top_p",
                "data_type": "TYPE_FP32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "frequency_penalty",
                "data_type": "TYPE_FP32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "presence_penalty",
                "data_type": "TYPE_FP32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "max_tokens",
                "data_type": "TYPE_INT32",
                "dims": [1],
                "optional": True,
            },
            {
                "name": "stop",
                "data_type": "TYPE_STRING",
                "dims": [-1],
                "optional": True,
            },
            {
                # stop_words is deprecated. Should use stop instead.
                "name": "stop_words",
                "data_type": "TYPE_STRING",
                "dims": [-1],
                "optional": True,
            },
            {
                "name": "seed",
                "data_type": "TYPE_UINT64",
                "dims": [1],
                "optional": True,
            },
            {
                # random_seed is deprecated. Should use seed instead.
                "name": "random_seed",
                "data_type": "TYPE_UINT64",
                "dims": [1],
                "optional": True,
            },
        ]
        # Outputs expected by the backend.
        outputs = [
            {"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "finish_reason", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "stop_reason", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "cumulative_logprob", "data_type": "TYPE_FP32", "dims": [-1]},
        ]

        # Collect input and output names from the provided model config.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        # Add missing inputs and outputs to the model config.
        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        self.params = self.model_config["parameters"]
        self.logger = pb_utils.Logger

        output_config = pb_utils.get_output_config_by_name(
            self.model_config, "text_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        if global_mpi_rank() == 0:
            # Initialize engine arguments
            self._init_engine_args()
            self.logger.log_info(
                f"[trtllm] rank{global_mpi_rank()} is starting trtllm engine with args: {self.llm_engine_args}"
            )

            # Starting the TRT-LLM engine with LLM API and its event thread running the AsyncIO event loop.
            self._init_engine()

            # Starting the response thread. It allows TRT-LLM to keep making progress while
            # response sender(s) are sending responses to server frontend.
            self._response_queue = queue.Queue()
            self._response_thread = threading.Thread(target=self._response_loop)
            self._response_thread.start()
        else:
            self.logger.log_info(
                f"[trtllm] rank{global_mpi_rank()} is waiting for the leader node..."
            )
            with MPICommExecutor(COMM_WORLD) as executor:
                if executor is not None:
                    raise RuntimeError(
                        f"[trtllm] rank{COMM_WORLD.rank} should not have executor"
                    )
            return

    def _get_llm_args(self, args_dict):
        pytorch_config_args = {
            k: v
            for k, v in args_dict.items()
            if k in self.PYTORCH_CONFIG_KEYS and v is not None
        }
        llm_engine_args = {
            k: v
            for k, v in args_dict.items()
            if k in self.LLM_ENGINE_KEYS and v is not None
        }
        if "model" not in llm_engine_args:
            raise pb_utils.TritonModelException(
                "Model name is required in the TRT-LLM engine config."
            )

        return pytorch_config_args, llm_engine_args

    def _init_engine_args(self):
        """Initialize engine arguments from config file."""
        engine_args_filepath = os.path.join(
            pb_utils.get_model_dir(), _TRTLLM_ENGINE_ARGS_FILENAME
        )
        if not os.path.isfile(engine_args_filepath):
            raise pb_utils.TritonModelException(
                f"'{_TRTLLM_ENGINE_ARGS_FILENAME}' containing TRT-LLM engine args must be provided in '{pb_utils.get_model_dir()}'"
            )

        try:
            with open(engine_args_filepath) as file:
                self.trtllm_engine_config = json.load(file)
        except json.JSONDecodeError as e:
            raise pb_utils.TritonModelException(f"Failed to parse engine config: {e}")

        self.pytorch_config_args, self.llm_engine_args = self._get_llm_args(
            self.trtllm_engine_config
        )

    def _init_engine(self):
        # Run the engine in a separate thread running the AsyncIO event loop.
        self._llm_engine = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(
            target=asyncio.run, args=(self._run_llm_engine(),)
        )
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            self.logger.log_error(f"[trtllm] Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

    async def _run_llm_engine(self):
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                pytorch_config = PyTorchConfig(**self.pytorch_config_args)
                llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        **self.llm_engine_args,
                        backend="pytorch",
                        pytorch_backend_config=pytorch_config,
                    ),
                )
                yield llm
            finally:
                if "llm" in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    self.logger.log_info(
                        "[trtllm] Awaiting remaining {} requests".format(
                            self._ongoing_request_count
                        )
                    )
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        self.logger.log_info("[trtllm] Shutdown complete")

    def _response_loop(self):
        while True:
            item = self._response_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            response_state, response, response_flag = item
            response_sender = response_state["response_sender"]
            try:
                response_sender.send(response, response_flag)
                # Stop checking for cancellation if the last response is generated.
                if not response_state["last_response_generated"]:
                    response_state["is_cancelled"] = response_sender.is_cancelled()
            except Exception as e:
                self.logger.log_error(
                    f"An error occurred while sending a response: {e}"
                )
            finally:
                if response_flag == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
                    self._ongoing_request_count -= 1

    def execute(self, requests):
        # TODO: Add health check here?
        for request in requests:
            # TODO : Verify Lora
            if request is not None:
                assert (
                    self._llm_engine_shutdown_event.is_set() is False
                ), "Cannot create tasks after shutdown has been requested"
                coro = self._generate(request)
                asyncio.run_coroutine_threadsafe(coro, self._event_loop)

        return None

    async def _generate(self, request):
        response_sender = request.get_response_sender()
        response_state = {
            "response_sender": response_sender,
            "is_cancelled": False,
            "last_response_generated": False,  # last response ready but not yet sent
        }
        self._ongoing_request_count += 1
        decrement_ongoing_request_count = True
        try:
            (
                prompt,
                stream,
                prepend_input,
                sampling_config,
                additional_outputs,
            ) = self._get_input_tensors(request)

            sampling_params = SamplingParams(**sampling_config)

            # Generate the response.
            response_iterator = self._llm_engine.generate_async(
                prompt, sampling_params, streaming=stream
            )

            request_output_state = {}
            async for request_output in response_iterator:
                # TODO: Add request cancellation check here
                # Send each response if streaming.
                if stream:
                    response = self._create_response(
                        request_output_state,
                        request_output,
                        prepend_input=False,
                        additional_outputs=additional_outputs,
                    )
                    flags = 0
                    if request_output.finished:
                        response_state["last_response_generated"] = True
                        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        decrement_ongoing_request_count = False
                    self._response_queue.put_nowait((response_state, response, flags))

            # Send the last response which contains all the outputs if not streaming.
            if not stream:
                response_sender.send(
                    self._create_response(
                        request_output_state={},
                        request_output=request_output,
                        prepend_input=prepend_input,
                        additional_outputs=additional_outputs,
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

        except Exception as e:
            self.logger.log_error(f"[trtllm] Error generating stream: {e}")
            error = pb_utils.TritonError(f"Error generating stream: {e}")
            text_output_tensor = pb_utils.Tensor(
                "text_output", np.asarray(["N/A"], dtype=self.output_dtype)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[text_output_tensor], error=error
            )
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
            raise e

        finally:
            if decrement_ongoing_request_count:
                self._ongoing_request_count -= 1

    def _get_input_tensors(self, request):
        # Parse the prompt based on the batch size.
        text_input = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()
        prompt = (
            text_input[0][0]
            if self.model_config["max_batch_size"] > 0
            else text_input[0]
        )

        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        # stream
        stream = pb_utils.get_input_tensor_by_name(request, "stream")
        if stream and not self.decoupled:
            raise pb_utils.TritonModelException(
                "Streaming is only supported in decoupled mode."
            )
        if stream:
            stream = stream.as_numpy()[0]
        else:
            stream = False

        # prepend_input / exclude_input_in_output
        prepend_input = pb_utils.get_input_tensor_by_name(
            request, "exclude_input_in_output"
        )
        if prepend_input:
            # When `exclude_input_in_output` is False, we want to prepend input prompt
            # to output, thus prepend_input should be True, and vice versa.
            prepend_input = not prepend_input.as_numpy()[0]
        elif prepend_input is None and stream:
            prepend_input = False
        else:
            # Default to False if not specified
            prepend_input = False
        if prepend_input and stream:
            raise pb_utils.TritonModelException(
                "When streaming, `exclude_input_in_output` = False is not allowed."
            )

        # Sampling parameters
        sampling_config = self._get_sampling_config_from_request(request)

        # additional outputs
        additional_outputs = {
            "return_finish_reason": None,
            "return_stop_reason": None,
        }
        for tensor_name in additional_outputs.keys():
            tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
            if tensor:
                tensor = bool(tensor.as_numpy()[0])
            else:
                tensor = False
            additional_outputs[tensor_name] = tensor

        return prompt, stream, prepend_input, sampling_config, additional_outputs

    def _create_response(
        self, request_output_state, request_output, prepend_input, additional_outputs
    ):
        # TODO: Check if request_output has_error and handle it
        output_tensors = []

        # text_output
        prepend_prompt = ""
        if "prev_lens_text_output" not in request_output_state:
            # this is the first response
            if prepend_input:
                prepend_prompt = request_output.prompt
            request_output_state["prev_lens_text_output"] = [0] * len(
                request_output.outputs
            )
        prev_lens = request_output_state["prev_lens_text_output"]
        text_output = [
            (prepend_prompt + output.text[prev_len:]).encode("utf-8")
            for output, prev_len in zip(request_output.outputs, prev_lens)
        ]
        request_output_state["prev_lens_text_output"] = [
            len(output.text) for output in request_output.outputs
        ]

        # finish_reason
        if additional_outputs["return_finish_reason"]:
            finish_reason = [
                str(output.finish_reason) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "finish_reason", np.asarray(finish_reason, dtype=np.object_)
                )
            )

        # stop_reason
        if additional_outputs["return_stop_reason"]:
            stop_reason = [
                str(output.finish_reason) for output in request_output.outputs
            ]
            output_tensors.append(
                pb_utils.Tensor(
                    "stop_reason", np.asarray(stop_reason, dtype=np.object_)
                )
            )

        output_tensors.append(
            pb_utils.Tensor(
                "text_output", np.asarray(text_output, dtype=self.output_dtype)
            )
        )

        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        self.logger.log_info("[trtllm] Issuing finalize to trtllm backend")
        self._event_loop.call_soon_threadsafe(self._llm_engine_shutdown_event.set)

        # Shutdown the event thread.
        if self._event_thread is not None:
            self._event_thread.join()
            self._event_thread = None

        # # Shutdown the response thread.
        self._response_queue.put(None)
        if self._response_thread is not None:
            self._response_thread.join()
            self._response_thread = None

        # When using parallel tensors, the stub process may not shutdown due to
        # unreleased references, so manually run the garbage collector once.
        self.logger.log_info("[trtllm] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[trtllm] Garbage Collector on finalize... done")
