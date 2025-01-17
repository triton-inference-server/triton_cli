import os
import sys
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).resolve().parent, ".."))
import pytest
from tests.utils import ScopedTritonServer


@pytest.fixture(scope="function")
def trtllm_server():
    llm_repo = None

    # TRT-LLM models should be pre-built offline, and only need to be read
    # from disk at server startup time, so they should generally load faster
    # than vLLM models, but still give some room for long startup.
    server = ScopedTritonServer(repo=llm_repo, timeout=600)
    yield server
    # Ensure server is cleaned up after each test
    server.stop()


@pytest.fixture(scope="function")
def vllm_server():
    llm_repo = None

    # vLLM models are downloaded on the fly during model loading as part of
    # server startup, so give even more room for timeout in case of slow network
    #     TODO: Consider one of the following
    #     (a) Pre-download and mount larger models in test environment
    #     (b) Download model from HF for vLLM at import step to remove burden
    #         from server startup step.
    server = ScopedTritonServer(repo=llm_repo, timeout=1800)
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
