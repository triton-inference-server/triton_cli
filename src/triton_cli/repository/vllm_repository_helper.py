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
from pathlib import Path
from .repository_helper import RepositoryHelper

from triton_cli.common import (
    LOGGER_NAME,
)

MODEL_CONFIG_TEMPLATE = """
backend: "{backend}"
instance_group {instance_group}
"""

logger = logging.getLogger(LOGGER_NAME)


class VLLMRepositoryHelper(RepositoryHelper):
    def create_model(self, source, source_type, model_name, version=1):
        model_dir = self.repo / model_name
        version_dir = model_dir / str(version)

        # Create the model and version directory
        try:
            version_dir.mkdir(parents=True, exist_ok=False)
            logger.debug(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        # Write the config.pbtxt and model.json for vllm model
        self.__generate_vllm_model_config(
            model_id=source, version_dir=version_dir, model_dir=model_dir
        )

    def __generate_vllm_model_config(
        self, model_id: str, version_dir: Path, model_dir: Path
    ):
        backend = "vllm"
        instance_group = "[{kind: KIND_MODEL}]"
        model_config = MODEL_CONFIG_TEMPLATE.format(
            backend=backend, instance_group=instance_group
        ).strip()

        model_contents = json.dumps(
            {
                "model": model_id,
                "disable_log_requests": True,
                "gpu_memory_utilization": 0.85,
            }
        )

        config_file = model_dir / "config.pbtxt"
        config_file.write_text(model_config)
        model_file = version_dir / "model.json"
        model_file.write_text(model_contents)
