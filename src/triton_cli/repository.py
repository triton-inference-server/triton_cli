import os
import json
import shutil
import logging
import subprocess
from pathlib import Path

from directory_tree import display_tree

from triton_cli.constants import DEFAULT_MODEL_REPO, LOGGER_NAME
from triton_cli.json_parser import parse_and_substitute

logger = logging.getLogger(LOGGER_NAME)

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

TRT_TEMPLATES_PATH = Path(__file__).parent / "templates" / "trtllm"
NGC_ENGINES_PATH = "/tmp/engines"


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
            format_type="ascii",
        )

    # To avoid having to interact with NGC CLI interactively,
    # just generate config file to skip auth step.
    def __generate_config(self, org="", team="", api_key="", format_type="ascii"):
        config_file = Path.home() / ".ngc" / "config"
        if config_file.exists():
            logger.debug("Found existing NGC config, skipping config generation")
            return

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
                "Found existing directory for {model} at {model_dir}, skipping download."
            )
            return

        cmd = f"ngc registry model download-version {model} --dest {dest}"
        logger.debug(f"Running '{cmd}'")
        output = subprocess.run(cmd.split())
        if output.returncode:
            err = output.stderr.decode("utf-8")
            raise Exception(f"Failed to download {model} from NGC:\n{err}")


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
            raise Exception("Non-empty model source must be provided")

        if backend:
            raise NotImplementedError(
                "No support for manually specifying backend at this time."
            )

        # HuggingFace models
        if source.startswith(SOURCE_PREFIX_HUGGINGFACE):
            logger.debug("HuggingFace prefix detected, parsing HuggingFace ID")
            source_type = "huggingface"
        # NGC models
        elif source.startswith(SOURCE_PREFIX_NGC):
            logger.debug("NGC prefix detected, parsing NGC ID")
            source_type = "ngc"
            backend = "tensorrtllm"
        # Local model path
        else:
            logger.debug("No supported prefix detected, assuming local path")
            source_type = "local"
            model_path = Path(source)
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} does not exist")

        model_dir, version_dir = self.__create_model_repository(name, version, backend)

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
            # NOTE: Assuming that `llama2_13b_trt_a100:0.1` from source
            #       transforms into llama2_13b_trt_a100_v0.1 folder when
            #       downloaded from NGC CLI.
            ngc_model_name = source.split("/")[-1].replace(":", "_v")
            ngc.download_model(ngc_id, ngc_model_name, dest=NGC_ENGINES_PATH)
            # TODO: grab downloaded config files,
            #       point to downloaded engines, etc.
            self.__generate_trtllm_model(name, ngc_model_name)
        else:
            logger.debug(f"Copying {model_path} to {version_dir}")
            shutil.copy(model_path, version_dir)

        if verbose:
            self.list()

    def clear(self):
        logger.info(f"Clearing all contents from {self.repo}...")
        shutil.rmtree(self.repo)

    # No support for removing individual versions for now
    # TODO: remove doesn't support removing groups of models like TRT LLM at this time
    # Use "clear" instead to clean up the repo as a WAR.
    def remove(self, name: str, verbose=True):
        model_dir = self.repo / name
        if not model_dir.exists():
            raise FileNotFoundError(f"No model folder exists at {model_dir}")
        logger.info(f"Removing model {name} at {model_dir}...")
        shutil.rmtree(model_dir)
        if verbose:
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

    def __generate_trtllm_model(self, name: str, source: str):
        engines_path = NGC_ENGINES_PATH + "/" + source
        parse_and_substitute(
            str(self.repo), engines_path, engines_path, "llama", dry_run=False
        )
        bls_model = self.repo / "tensorrt_llm_bls"
        bls_model.rename(self.repo / name)

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
                    raise Exception(
                        f"Found existing model at {version_dir}, skipping repo add."
                    )

                shutil.copytree(
                    TRT_TEMPLATES_PATH,
                    self.repo,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
                logger.debug(f"Adding TensorRT-LLM models at: {self.repo}")
            else:
                version_dir.mkdir(parents=True, exist_ok=False)
                logger.debug(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        return model_dir, version_dir
