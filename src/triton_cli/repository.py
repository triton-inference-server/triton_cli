# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import shutil
import logging
import subprocess
import multiprocessing
from pathlib import Path

from directory_tree import display_tree

from triton_cli.common import (
    DEFAULT_MODEL_REPO,
    SUPPORTED_BACKENDS,
    LOGGER_NAME,
    TritonCLIException,
)
from triton_cli.trt_llm.engine_config_parser import parse_and_substitute

from huggingface_hub import snapshot_download
from huggingface_hub import utils as hf_utils


logger = logging.getLogger(LOGGER_NAME)

# For now, generated model configs will be limited to only backends
# that can be fully autocompleted for a simple deployment.
MODEL_CONFIG_TEMPLATE = """
backend: "{backend}"
instance_group {instance_group}
"""

NGC_CONFIG_TEMPLATE = """
[CURRENT]
apikey = {api_key}
format_type = {format_type}
org = {org}
team = {team}
"""

SOURCE_PREFIX_HUGGINGFACE = "hf:"
SOURCE_PREFIX_NGC = "ngc:"
SOURCE_PREFIX_LOCAL = "local:"

TEMPALTES_PATH = Path(__file__).parent / "templates"
TRT_TEMPLATES_PATH = TEMPALTES_PATH / "trt_llm"
LLMAPI_TEMPLATES_PATH = TEMPALTES_PATH / "llmapi"

# Support changing destination dynamically to point at
# pre-downloaded checkpoints in various circumstances
ENGINE_DEST_PATH = os.environ.get("ENGINE_DEST_PATH", "/tmp/engines")

HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


# NOTE: Thin wrapper around NGC CLI is a WAR for now.
# TODO: Move out to generic files/interface for remote model stores
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


# Can eventually be an interface and have implementations
# for remote stores or similar, but keeping it simple for now.
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
        backend: str = None,
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
        # NGC models
        # TODO: Improve backend detection/assumption for NGC models in future
        elif source.startswith(SOURCE_PREFIX_NGC):
            logger.debug("NGC prefix detected, parsing NGC ID")
            source_type = "ngc"
            backend = "tensorrtllm"
        # Local model path
        else:
            if source.startswith(SOURCE_PREFIX_LOCAL):
                logger.debug("Local prefix detected, parsing local file path")
            else:
                logger.info(
                    "No supported --source prefix detected, assuming local path"
                )

            source_type = "local"
            model_path = Path(source)
            if not model_path.exists():
                raise TritonCLIException(
                    f"Local file path '{model_path}' provided by --source does not exist"
                )
        model_dir, version_dir = self.__create_model_repository(name, version, backend)

        # Note it's a bit redundant right now, but we check prefix above first
        # to avoid creating model repository files in case that local source
        # path is invalid. This should be cleaned up.
        if source_type == "huggingface":
            hf_id = source.split(":")[1]
            self.__add_huggingface_model(model_dir, version_dir, hf_id, name, backend)
        elif source_type == "ngc":
            # NOTE: NGC models likely to contain colons
            ngc_id = source.replace(SOURCE_PREFIX_NGC, "")
            ngc = NGCWrapper()
            # NOTE: Assuming that `llama2_13b_trt_a100:0.1` from source
            #       transforms into llama2_13b_trt_a100_v0.1 folder when
            #       downloaded from NGC CLI.
            ngc_model_name = source.split("/")[-1].replace(":", "_v")
            ngc.download_model(ngc_id, ngc_model_name, dest=ENGINE_DEST_PATH)
            # TODO: grab downloaded config files,
            #       point to downloaded engines, etc.
            self.__generate_ngc_model(name, ngc_model_name)
        else:
            logger.debug(f"Copying {model_path} to {version_dir}")
            shutil.copy(model_path, version_dir)

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

    def __add_huggingface_model(
        self,
        model_dir: Path,
        version_dir: Path,
        huggingface_id: str,
        name: str,
        backend: str,
    ):
        if not model_dir or not model_dir.exists():
            raise TritonCLIException("Model directory must be provided and exist")
        if not huggingface_id:
            raise TritonCLIException("HuggingFace ID must be non-empty")

        if backend == "tensorrtllm":
            # TODO: Refactor the cleanup flow, move it to a higher level
            try:
                self.__generate_trtllm_model(name, huggingface_id)
            except Exception as e:
                # If generating TRLTLM model fails, clean up the draft models
                # added to the model repository.
                logger.warning(f"TRT-LLM model creation failed: {e}. Cleaning up...")
                for model in [name, "preprocessing", "tensorrt_llm", "postprocessing"]:
                    self.remove(model, verbose=False)
                # Let detailed traceback be reported for TRT-LLM errors for debugging
                raise e
        elif backend == "llmapi":
            self.__generate_llmapi_model(version_dir, huggingface_id)
        else:
            # TODO: Add generic support for HuggingFace models with HF API.
            # For now, use vLLM as a means of deploying HuggingFace Transformers
            # NOTE: Only transformer models are supported at this time.
            config, files = self.__generate_vllm_model(huggingface_id)
            config_file = model_dir / "config.pbtxt"
            config_file.write_text(config)
            for file, contents in files.items():
                model_file = version_dir / file
                model_file.write_text(contents)

    def __download_hf_model(
        self,
        huggingface_id: str,
        hf_download_path: str,
        allow_patterns: list = [],
        ignore_patterns: list = [],
    ):
        # Shouldn't require the user to authenticate with HF unless
        # necessary (i.e., the model exists in a gated repo)
        try:
            snapshot_download(
                huggingface_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=hf_download_path,
            )
        except hf_utils.GatedRepoError:
            if not HF_TOKEN_PATH.exists():
                raise TritonCLIException(
                    "Please authenticate using 'huggingface-cli login' to download this model"
                )
            snapshot_download(
                huggingface_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=hf_download_path,
                use_auth_token=True,  # for gated repos like llama
            )

    def __generate_llmapi_model(self, version_dir, huggingface_id: str):
        # load the model.json from llmapi template
        model_config_file = version_dir / "model.json"
        with open(model_config_file) as f:
            model_config_str = f.read()

        model_config_json = json.loads(model_config_str)

        # change the model id as the huggingface_id
        model_config_json["model"] = huggingface_id

        # write back the model.json
        with open(model_config_file, "w") as f:
            f.write(json.dumps(model_config_json))

    def __generate_vllm_model(self, huggingface_id: str):
        backend = "vllm"
        instance_group = "[{kind: KIND_MODEL}]"
        model_config = MODEL_CONFIG_TEMPLATE.format(
            backend=backend, instance_group=instance_group
        ).strip()
        model_contents = json.dumps(
            {
                "model": huggingface_id,
                "disable_log_requests": True,
                "gpu_memory_utilization": 0.85,
            }
        )
        model_files = {"model.json": model_contents}
        return model_config, model_files

    def __generate_ngc_model(self, name: str, source: str):
        engines_path = ENGINE_DEST_PATH + "/" + source
        parse_and_substitute(
            str(self.repo), name, engines_path, engines_path, "auto", dry_run=False
        )

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

    def __create_model_repository(
        self, name: str, version: int = 1, backend: str = None
    ):
        # Create model directory in repo with name, raise error if
        # repo doesn't exist, or model directory already exists.
        model_dir = self.repo / name
        version_dir = model_dir / str(version)
        try:
            if backend == "tensorrtllm":
                # Don't allow existing files for TRT-LLM for now in case we delete large engine files
                if model_dir.exists():
                    raise TritonCLIException(
                        f"Found existing model at {version_dir}, skipping repo add."
                    )

                shutil.copytree(
                    TRT_TEMPLATES_PATH,
                    self.repo,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
                bls_model = self.repo / "tensorrt_llm_bls"
                bls_model.rename(self.repo / name)
                logger.debug(f"Adding TensorRT-LLM models at: {self.repo}")
            else:
                version_dir.mkdir(parents=True, exist_ok=False)

                if backend == "llmapi":
                    shutil.copytree(
                        LLMAPI_TEMPLATES_PATH / "1",
                        version_dir,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                    shutil.copy(
                        LLMAPI_TEMPLATES_PATH / "config.pbtxt",
                        model_dir,
                    )
                logger.debug(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        return model_dir, version_dir
