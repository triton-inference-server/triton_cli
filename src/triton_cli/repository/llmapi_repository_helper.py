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

import json
import logging
import shutil
from pathlib import Path
from .repository_helper import RepositoryHelper

from triton_cli.common import (
    LOGGER_NAME,
)

LLMAPI_TEMPLATES_PATH = Path(__file__).parent.parent / "llmapi"

logger = logging.getLogger(LOGGER_NAME)


class LLMAPIRepositoryHelper(RepositoryHelper):
    def create_model(self, source, source_type, model_name, version=1):
        model_dir = self.repo / model_name
        version_dir = model_dir / str(version)

        # Create the model and version directory
        try:
            version_dir.mkdir(parents=True, exist_ok=False)
            shutil.copytree(
                LLMAPI_TEMPLATES_PATH / "1",
                version_dir,
                dirs_exist_ok=False,
                ignore=shutil.ignore_patterns("__pycache__"),
            )

            shutil.copy(
                LLMAPI_TEMPLATES_PATH / "config.pbtxt",
                model_dir,
            )
            logger.debug(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        # Fill in the model.json for llmapi
        self.__fill_llmapi_args(version_dir, source)

    def __fill_llmapi_args(self, version_dir, model_id: str):
        # load the model.json from llmapi template
        model_config_file = version_dir / "model.json"
        with open(model_config_file) as f:
            model_config_str = f.read()

        model_config_json = json.loads(model_config_str)

        # change the model id as the huggingface_id
        model_config_json["model"] = model_id

        # write back the model.json
        with open(model_config_file, "w") as f:
            f.write(json.dumps(model_config_json))
