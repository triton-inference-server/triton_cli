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

import logging
import subprocess
from pathlib import Path

from triton_cli.common import LOGGER_NAME

from triton_cli.common import TritonCLIException


logger = logging.getLogger(LOGGER_NAME)


class TRTLLMBuilder:
    def __init__(self, huggingface_id, hf_download_path, engine_output_path, config):
        import tensorrt_llm

        if tensorrt_llm.__version__ != "0.9.0":
            raise TritonCLIException(
                f"tensorrt_llm version {tensorrt_llm.__version__} not supported by triton_cli."
                "Please use tensorrt_llm version 0.9.0."
            )
        self.checkpoint_id = config["tensorrtllm"]["convert_checkpoint_type"]
        self.hf_download_path = hf_download_path
        self.converted_weights_path = self.hf_download_path + "/converted_weights"
        self.engine_output_path = engine_output_path
        self.config = config

    def _make_arg(self, arg_name, arg_value):
        if arg_value is None:  # Boolean Argument
            return f"--{arg_name}"
        else:
            return f"--{arg_name}={arg_value}"

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self):
        self._convert_checkpoint()
        self._trtllm_build()

    def _convert_checkpoint(self):
        if Path(self.converted_weights_path).exists():
            logger.info(
                f"Converted weights path {self.converted_weights_path} already exists, skipping checkpoint conversion."
            )
            return

        weight_conversion_args = [
            "--model_dir",
            self.hf_download_path,
            "--output_dir",
            self.converted_weights_path,
        ]

        if self.config and isinstance(
            self.config["tensorrtllm"]["convert_checkpoint_args"], dict
        ):
            weight_conversion_args += [
                self._make_arg(arg_name, arg_value)
                for arg_name, arg_value in self.config["tensorrtllm"][
                    "convert_checkpoint_args"
                ].items()
            ]

        # Need to specify gpt variant for gpt models
        if self.checkpoint_id in ["gpt2"]:
            weight_conversion_args += ["--gpt_variant", self.checkpoint_id]

        ckpt_script = (
            Path(__file__).resolve().parent
            / "checkpoint_scripts"
            / self.checkpoint_id
            / "convert_checkpoint.py"
        )
        cmd = ["python3", str(ckpt_script)] + weight_conversion_args
        cmd_str = " ".join(cmd)
        logger.info(f"Running '{cmd_str}'")
        subprocess.run(cmd, check=True)

    def _trtllm_build(self):
        # TODO: Move towards config-driven build args per-model
        build_args = [
            f"--checkpoint_dir={self.converted_weights_path}",
            f"--output_dir={self.engine_output_path}",
        ]

        if self.config and isinstance(
            self.config["tensorrtllm"]["trtllm_build_args"], dict
        ):
            build_args += [
                self._make_arg(arg_name, arg_value)
                for arg_name, arg_value in self.config["tensorrtllm"][
                    "trtllm_build_args"
                ].items()
            ]

        cmd = ["trtllm-build"] + build_args
        cmd_str = " ".join(cmd)
        logger.info(f"Running '{cmd_str}'")
        subprocess.run(cmd, check=True)
