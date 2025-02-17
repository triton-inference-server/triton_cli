# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def llmapi_server():
    llm_repo = None

    # llmapi models might need to be downloaded on the fly during model loading,
    # so leave longer timeout in case of slow network.
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
