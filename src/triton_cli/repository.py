import json
import shutil
import logging
from pathlib import Path

from directory_tree import display_tree

from triton_cli.constants import DEFAULT_MODEL_REPO

logger = logging.getLogger("triton")

# For now, generated model configs will be limited to only backends
# that can be fully autocompleted for a simple deployment.
MODEL_CONFIG_TEMPLATE = """
backend: "{backend}"
"""


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
        model_path: str = None,
        huggingface_id: str = None,
        backend: str = None,
    ):
        if all([model_path, huggingface_id]):
            raise Exception(
                "Must specify oneof (model_path, huggingface_id), but not both"
            )
        if not any([model_path, huggingface_id]):
            raise Exception("Must specify oneof (model_path, huggingface_id)")
        if "/" in name or ".." in name:
            raise Exception(
                f"Name argument must not contain any path-like elements like '/' or '..': {name}"
            )
        if backend:
            raise NotImplementedError

        # Create model directory in repo with name, raise error if
        # repo doesn't exist, or model directory already exists.
        model_dir = self.repo / name
        version_dir = model_dir / str(version)
        try:
            version_dir.mkdir(parents=True, exist_ok=False)
            logger.info(f"Adding new model to repo at: {version_dir}")
        except FileExistsError:
            logger.warning(f"Overwriting existing model in repo at: {version_dir}")

        if model_path:
            raise NotImplementedError

        if huggingface_id:
            self.__add_huggingface_model(model_dir, version_dir, huggingface_id)

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

        # For now, use vLLM as a means of deploying HuggingFace Transformers
        # NOTE: Non-transformer models are not supported at this time.
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
