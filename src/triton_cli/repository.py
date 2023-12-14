import os
import json
import shutil
import logging
import subprocess
from pathlib import Path

from directory_tree import display_tree

from triton_cli.constants import DEFAULT_MODEL_REPO

logger = logging.getLogger("triton")

# For now, generated model configs will be limited to only backends
# that can be fully autocompleted for a simple deployment.
MODEL_CONFIG_TEMPLATE = """
backend: "{backend}"
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

# NOTE: Thin wrapper around NGC CLI is a WAR for now.
# TODO: Move out to generic files/interface for remote model stores
class NGCWrapper:
    def __init__(self):
        api_key = os.environ.get("NGC_API_KEY", "")

        # Hard-coded for demo purposes
        self.__generate_config(
            org="whw3rcpsilnj",
            team="playground",
            api_key=api_key,
            # For interactive output to see download progress
            format_type="ascii"
        )

    # To avoid having to interact with NGC CLI interactively,
    # just generate config file to skip auth step.
    def __generate_config(self, org="", team="", api_key="", format_type="ascii"):
        config_file = Path.home() / ".ngc" / "config"
        if config_file.exists():
            logger.debug("Found existing NGC config")
            return

        logger.debug("Generating NGC config")
        config = NGC_CONFIG_TEMPLATE.format(
            api_key=api_key,
            format_type=format_type,
            org=org,
            team=team
        )
        config_file.write_text(config)

    # TODO: Remove default model after demo
    # Update model with correct string if running on non-A100 GPU
    def download_model(self, model, dest):
        logger.info(f"Downloading NGC model: {model} to {dest}...")
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        cmd = f"ngc registry model download-version {model} --dest {dest}"
        logger.debug(f"Running '{cmd}'")
        subprocess.run(cmd.split())


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
            logger.info(f"Created new model repository: {self.repo}")
        except FileExistsError:
            logger.info(f"Using existing model repository: {self.repo}")

    def list(self):
        logger.info(f"Current repo at {self.repo}:")
        display_tree(self.repo)

    def add(
        self,
        name: str,
        version: int = 1,
        source: str = None,
        backend: str = None,
    ):
        if not source:
            raise Exception("Non-empty model source must be provided")

        # HuggingFace models
        if source.startswith(SOURCE_PREFIX_HUGGINGFACE):
            logger.info("HuggingFace prefix detected, parsing HuggingFace ID")
            source_type = "huggingface"
        # NGC models
        elif source.startswith(SOURCE_PREFIX_NGC):
            logger.info("NGC prefix detected, parsing NGC ID")
            source_type = "ngc"
        # Local model path
        else:
            logger.info("No supported prefix detected, assuming local path")
            source_type = "local"
            model_path = Path(source)
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} does not exist")

        # Create model directory in repo with name, raise error if
        # repo doesn't exist, or model directory already exists.
        model_dir = self.repo / name
        version_dir = model_dir / str(version)
        try:
            version_dir.mkdir(parents=True, exist_ok=False)
            logger.info(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        if backend:
            raise NotImplementedError(
                "No support for manually specifying backend at this time."
            )

        # Note it's a bit redundant right now, but we check prefix above first
        # to avoid creating model repository files in case that local source
        # path is invalid. This should be cleaned up.
        if source_type == "huggingface":
            hf_id = source.split(":")[1]
            self.__add_huggingface_model(model_dir, version_dir, hf_id)
        elif source_type == "ngc":
            # NOTE: NGC models likely to contain colons
            ngc_id = source.replace(SOURCE_PREFIX_NGC, "")
            ngc = NGCWrapper()
            ngc.download_model(ngc_id, dest="/tmp/engines")
            # TODO: Copy TRT LLM template, grab downloaded config files,
            #       point to downloaded engines, etc.
            self.__generate_trtllm_model()
        else:
            logger.info(f"Copying {model_path} to {version_dir}")
            shutil.copy(model_path, version_dir)

        self.list()

    def clear(self):
        logger.info(f"Clearing all contents from {self.repo}...")
        shutil.rmtree(self.repo)

    # No support for removing individual versions for now
    def remove(self, name: str):
        model_dir = self.repo / name
        if not model_dir.exists():
            raise FileNotFoundError(f"No model folder exists at {model_dir}")
        logger.info(f"Removing model {name} at {model_dir}...")
        shutil.rmtree(model_dir)
        self.list()

    def __add_huggingface_model(
        self, model_dir: Path, version_dir: Path, huggingface_id: str
    ):
        if not model_dir or not model_dir.exists():
            raise Exception("Model directory must be provided and exist")
        if not huggingface_id:
            raise Exception("HuggingFace ID must be non-empty")

        # TODO: Add generic support for HuggingFace models with HF API.
        # For now, use vLLM as a means of deploying HuggingFace Transformers
        # NOTE: Only transformer models are supported at this time.
        config, files = self.__generate_vllm_model(huggingface_id)
        config_file = model_dir / "config.pbtxt"
        config_file.write_text(config)
        for file, contents in files.items():
            model_file = version_dir / file
            model_file.write_text(contents)

    def __generate_vllm_model(self, huggingface_id: str):
        backend = "vllm"
        model_config = MODEL_CONFIG_TEMPLATE.format(backend=backend)
        model_contents = json.dumps(
            {
                "model": huggingface_id,
                "disable_log_requests": True,
                "gpu_memory_utilization": 0.5,
            }
        )
        model_files = {"model.json": model_contents}
        return model_config, model_files

    def __generate_trtllm_model(self):
        pass
