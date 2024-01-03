#!/usr/bin/env python3

from pathlib import Path

LOGGER_NAME: str = "triton"

# Server
DEFAULT_TRITONSERVER_PATH: str = "tritonserver"
## Server Docker
DEFAULT_SHM_SIZE: str = "1G"

# Model Repository
DEFAULT_MODEL_REPO: Path = Path.home() / "models"
