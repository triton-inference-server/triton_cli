import os
import sys
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).resolve().parent, ".."))
import pytest
from tests.utils import ScopedTritonServer


@pytest.fixture(scope="function")
def llm_server():
    llm_repo = None

    # Give ample startup timeout for possible downloading of models
    server = ScopedTritonServer(repo=llm_repo, timeout=600)
    yield server
    # Ensure server is cleaned up after each test
    server.stop()


@pytest.fixture(scope="function")
def simple_server():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    simple_repo = os.path.join(test_dir, "test_models")

    server = ScopedTritonServer(repo=simple_repo)
    yield server
    # Ensure server is cleaned up after each test
    server.stop()
