#!/usr/bin/env python3

from pathlib import Path

LOGGER_NAME: str = "triton"

# Server
DEFAULT_SHM_SIZE: str = "1G"
DEFAULT_TRITONSERVER_PATH: str = "tritonserver"

# Model Repository
DEFAULT_MODEL_REPO: Path = Path.home() / "models"
