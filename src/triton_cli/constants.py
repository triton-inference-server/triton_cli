#!/usr/bin/env python3

from pathlib import Path
import os

LOGGER_NAME: str = "triton"

# Server
DEFAULT_TRITONSERVER_PATH: str = "tritonserver"
## Server Docker
DEFAULT_SHM_SIZE: str = "1G"

# Model Repository
DEFAULT_MODEL_REPO: Path = Path.home() / "models"

# Support changing destination dynamically to point at
# pre-downloaded checkpoints in various circumstances
NGC_ENGINES_PATH = os.environ.get("NGC_DEST_DIR", "/tmp/engines")
