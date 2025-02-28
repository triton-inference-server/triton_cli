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

import os
import logging
import shutil
import multiprocessing
from pathlib import Path
from .repository_helper import RepositoryHelper
from .model_source.ngc import NGCWrapper
from triton_cli.trt_llm.engine_config_parser import parse_and_substitute

from triton_cli.common import LOGGER_NAME, TritonCLIException

TRTLLM_TEMPLATES_PATH = Path(__file__).parent.parent / "trt_llm"

# Support changing destination dynamically to point at
# pre-downloaded checkpoints in various circumstances
ENGINE_DEST_PATH = os.environ.get("ENGINE_DEST_PATH", "/tmp/engines")

logger = logging.getLogger(LOGGER_NAME)


class TRTLLMRepositoryHelper(RepositoryHelper):
    def create_model(self, source, source_type, model_name, version=1):
        model_dir = self.repo / model_name
        version_dir = model_dir / str(version)

        # Create the model and version directory
        try:
            # Don't allow existing files for TRT-LLM for now in case we delete large engine files
            if model_dir.exists():
                raise TritonCLIException(
                    f"Found existing model at {version_dir}, skipping repo add."
                )

            shutil.copytree(
                TRTLLM_TEMPLATES_PATH,
                self.repo,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            bls_model = self.repo / "tensorrt_llm_bls"
            bls_model.rename(self.repo / model_name)
            logger.debug(f"Adding TensorRT-LLM models at: {self.repo}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        # generate trtllm engines
        if source_type == "ngc":
            ngc = NGCWrapper()
            # NOTE: Assuming that `llama2_13b_trt_a100:0.1` from source
            #       transforms into llama2_13b_trt_a100_v0.1 folder when
            #       downloaded from NGC CLI.
            ngc_model_name = source.split("/")[-1].replace(":", "_v")
            ngc.download_model(source, ngc_model_name, dest=ENGINE_DEST_PATH)
            # TODO: grab downloaded config files,
            #       point to downloaded engines, etc.
            self.__generate_ngc_model(model_name, ngc_model_name)
        else:
            self.__generate_trt_llm_engine(version_dir, source)

    def __generate_trt_llm_engine(self, model_name, huggingface_id):
        # TODO: Refactor the cleanup flow, move it to a higher level
        try:
            self.__generate_trtllm_model(model_name, huggingface_id)
        except Exception as e:
            # If generating TRLTLM model fails, clean up the draft models
            # added to the model repository.
            logger.warning(f"TRT-LLM model creation failed: {e}. Cleaning up...")
            for model in [
                model_name,
                "preprocessing",
                "tensorrt_llm",
                "postprocessing",
            ]:
                model_dir = self.repo / model
                if not model_dir.exists():
                    continue
                logger.info(f"Removing model {model} at {model_dir}...")
                shutil.rmtree(model_dir)
            # Let detailed traceback be reported for TRT-LLM errors for debugging
            raise e

    def __generate_trtllm_model(self, name: str, huggingface_id: str):
        engines_path = ENGINE_DEST_PATH + "/" + name
        engines = [engine for engine in Path(engines_path).glob("*.engine")]
        if engines:
            logger.warning(
                f"Found existing engine(s) at {engines_path}, skipping build."
            )
        else:
            # Run TRT-LLM build in a separate process to make sure it definitely
            # cleans up any GPU memory used when done.
            p = multiprocessing.Process(
                target=self.__build_trtllm_engine, args=(huggingface_id, engines_path)
            )
            p.start()
            p.join()

        # NOTE: In every case, the TRT LLM template should be filled in with values.
        # If the model exists, the CLI will raise an exception when creating the model repo.
        # If a user clears the model repo, they won't need to re-build the engines,
        # but they will still need to modify the TRT LLM template.
        parse_and_substitute(
            triton_model_dir=str(self.repo),
            bls_model_name=name,
            engine_dir=engines_path,
            token_dir=engines_path,
            token_type="auto",
            dry_run=False,
        )

    def __build_trtllm_engine(self, huggingface_id: str, engines_path: Path):
        from tensorrt_llm import LLM, BuildConfig

        # NOTE: Given config.json, can read from 'build_config' section and from_dict
        config = BuildConfig()
        # TODO: Expose more build args to user
        # TODO: Discuss LLM API BuildConfig defaults
        # config.max_input_len = 1024
        # config.max_seq_len = 8192
        # config.max_batch_size = 256

        engine = LLM(huggingface_id, build_config=config)
        # TODO: Investigate if LLM is internally saving a copy to a temp dir
        engine.save(str(engines_path))

        # The new trtllm(v0.17.0+) requires explicit calling shutdown to shutdown
        # the mpi blocking thread, or the engine process won't exit
        engine.shutdown()

    def __generate_ngc_model(self, name: str, source: str):
        engines_path = ENGINE_DEST_PATH + "/" + source
        parse_and_substitute(
            str(self.repo), name, engines_path, engines_path, "auto", dry_run=False
        )
