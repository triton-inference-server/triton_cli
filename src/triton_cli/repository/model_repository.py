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
from vllm_repository_helper import VLLMRepositoryHelper
from trtllm_repository_helper import TRTLLMRepositoryHelper
from llmapi_repository_helper import LLMAPIRepositoryHelper
from pathlib import Path
from directory_tree import display_tree
from triton_cli.common import (
    DEFAULT_MODEL_REPO,
    SUPPORTED_BACKENDS,
    LOGGER_NAME,
    TritonCLIException,
)

SOURCE_PREFIX_HUGGINGFACE = "hf:"
SOURCE_PREFIX_NGC = "ngc:"
SOURCE_PREFIX_LOCAL = "local:"

logger = logging.getLogger(LOGGER_NAME)


class ModelRepository:
    def __init__(self, path: str = None):
        self.repo = DEFAULT_MODEL_REPO
        if path:
            self.repo = Path(path)

        # OK if model repo already exists, support adding multiple models
        try:
            self.repo.mkdir(parents=True, exist_ok=False)
            logger.debug(f"Created new model repository: {self.repo}")
        except FileExistsError:
            logger.debug(f"Using existing model repository: {self.repo}")

    def list(self):
        logger.info(f"Current repo at {self.repo}:")
        display_tree(self.repo)

    def add(
        self,
        name: str,
        version: int = 1,
        source: str = None,
        backend: str = "vllm",
        verbose=True,
    ):
        if not source:
            raise TritonCLIException("Non-empty model source must be provided")

        if backend and backend not in SUPPORTED_BACKENDS:
            raise TritonCLIException(
                f"The specified backend is not currently supported. Please choose from the following backends {SUPPORTED_BACKENDS}"
            )

        # HuggingFace models
        if source.startswith(SOURCE_PREFIX_HUGGINGFACE):
            logger.debug("HuggingFace prefix detected, parsing HuggingFace ID")
            source_type = "huggingface"
            source = source.replace(SOURCE_PREFIX_HUGGINGFACE, "")
        # NGC models
        # TODO: Improve backend detection/assumption for NGC models in future
        elif source.startswith(SOURCE_PREFIX_NGC):
            logger.debug("NGC prefix detected, parsing NGC ID")
            source_type = "ngc"
            backend = "tensorrtllm"
            source = source.replace(SOURCE_PREFIX_NGC, "")
        # Local model path
        else:
            if source.startswith(SOURCE_PREFIX_LOCAL):
                logger.debug("Local prefix detected, parsing local file path")
            else:
                logger.info(
                    "No supported --source prefix detected, assuming local path"
                )

            source_type = "local"
            source = source.replace(SOURCE_PREFIX_LOCAL, "")
            model_path = Path(source)
            if not model_path.exists():
                raise TritonCLIException(
                    f"Local file path '{model_path}' provided by --source does not exist"
                )

        repo_helper = self.__get_repository_helper(backend)
        repo_helper.create_model(source, source_type, name, version)

        if verbose:
            self.list()

    def clear(self):
        logger.info(f"Clearing all contents from {self.repo}...")

        # Loops through folders in self.repo (default: /root/models)
        # and deletes each model directory individually.
        for models in os.listdir(self.repo):
            shutil.rmtree(self.repo / models)

    # No support for removing individual versions for now
    # TODO: remove doesn't support removing groups of models like TRT LLM at this time
    # Use "clear" instead to clean up the repo as a WAR.
    def remove(self, name: str, verbose=True):
        if name.lower() == "all":
            return self.clear()

        model_dir = self.repo / name
        if not model_dir.exists():
            raise TritonCLIException(f"No model folder exists at {model_dir}")
        logger.info(f"Removing model {name} at {model_dir}...")
        shutil.rmtree(model_dir)
        if verbose:
            self.list()

    def __get_repository_helper(self, backend):
        if backend == "llmapi":
            return LLMAPIRepositoryHelper(self.repo)
        elif backend == "vllm":
            return VLLMRepositoryHelper(self.repo)
        else:
            return TRTLLMRepositoryHelper(self.repo)
