#!/usr/bin/env python3

import os
from pathlib import Path

LOGGER_NAME: str = "triton"

# Server
DEFAULT_TRITONSERVER_PATH: str = "tritonserver"
## Server Docker
DEFAULT_SHM_SIZE: str = "1G"

# Model Repository
DEFAULT_MODEL_REPO: Path = Path.home() / "models"
DEFAULT_HF_CACHE: Path = Path.home() / ".cache" / "huggingface"
HF_CACHE: Path = Path(os.environ.get("TRANSFORMERS_CACHE", DEFAULT_HF_CACHE))
