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

# NOTE: Thin wrapper around NGC CLI is a WAR for now.
# TODO: Move out to generic files/interface for remote model stores

import os
import logging
import subprocess
from pathlib import Path

from triton_cli.common import (
    LOGGER_NAME,
    TritonCLIException,
)

NGC_CONFIG_TEMPLATE = """
[CURRENT]
apikey = {api_key}
format_type = {format_type}
org = {org}
team = {team}
"""

logger = logging.getLogger(LOGGER_NAME)


class NGCWrapper:
    def __init__(self):
        api_key = os.environ.get("NGC_API_KEY", "")

        # TODO: revisit default org/team
        self.__generate_config(
            org="nvidia",
            team="",
            api_key=api_key,
            # For interactive output to see download progress
            format_type="ascii",
        )

    # To avoid having to interact with NGC CLI interactively,
    # just generate config file to skip auth step.
    def __generate_config(self, org="", team="", api_key="", format_type="ascii"):
        config_dir = Path.home() / ".ngc"
        config_file = config_dir / "config"
        if config_file.exists():
            logger.debug("Found existing NGC config, skipping config generation")
            return

        if not config_dir.exists():
            config_dir.mkdir(exist_ok=True)

        logger.debug("Generating NGC config")
        config = NGC_CONFIG_TEMPLATE.format(
            api_key=api_key, format_type=format_type, org=org, team=team
        )
        config_file.write_text(config)

    # TODO: Remove default model after demo
    # Update model with correct string if running on non-A100 GPU
    def download_model(self, model, ngc_model_name, dest):
        logger.info(f"Downloading NGC model: {model} to {dest}...")
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        model_dir = dest_path / ngc_model_name
        if model_dir.exists():
            logger.warning(
                f"Found existing directory for {model} at {model_dir}, skipping download."
            )
            return

        cmd = f"ngc registry model download-version {model} --dest {dest}"
        logger.debug(f"Running '{cmd}'")
        output = subprocess.run(cmd.split())
        if output.returncode:
            err = output.stderr.decode("utf-8")
            raise TritonCLIException(f"Failed to download {model} from NGC:\n{err}")
