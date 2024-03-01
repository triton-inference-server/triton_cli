# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys


import os
import pytest
from triton_cli.main import run
from triton_cli.parser import KNOWN_MODEL_SOURCES
import utils

sys.path.append("../")
KNOWN_MODELS = KNOWN_MODEL_SOURCES.keys()
KNOWN_SOURCES = KNOWN_MODEL_SOURCES.values()
TRTLLM_MODELS = ["gpt2"]
VLLM_MODELS = ["gpt2"]

PROMPT = "machine learning is"


class TestE2E:
    def repo_clear(self, repo=None):
        args = ["repo", "clear"]
        if repo:
            args += ["--repo", repo]
        run(args)

    def repo_add(self, model, source=None, backend=None, repo=None):
        args = ["repo", "add", "-m", model]
        if source:
            args += ["--source", source]
        if backend:
            args += ["--backend", backend]
        if repo:
            args += ["--repo", repo]
        run(args)

    def model_infer(self, model, prompt=None, protocol=None):
        args = ["model", "infer", "-m", model]
        if prompt:
            args += ["--prompt", prompt]
        if protocol:
            args += ["-i", protocol]
        run(args)

    def model_profile(self, model, protocol=None, backend=None):
        args = ["model", "profile", "-m", model]
        if protocol:
            args += ["-i", protocol]
        if backend:
            args += ["--backend", backend]
        run(args)

    class KillServerByPid:
        def __init__(self):
            self.pid = None

        def kill_server(self):
            if self.pid is not None:
                utils.kill_server(self.pid)

    @pytest.fixture
    def setup_and_teardown(self):
        # Setup before the test case is run.
        kill_server = self.KillServerByPid()
        self.repo_clear()

        yield kill_server

        # Teardown after the test case is done.
        kill_server.kill_server()
        self.repo_clear()

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.parametrize(
        "protocol",
        [
            "grpc",
            pytest.param(
                "http",
                marks=pytest.mark.xfail(
                    reason="http does not support model infer and model profile for decoupled models"
                ),
            ),
        ],
    )
    def test_tensorrtllm_e2e(self, protocol, setup_and_teardown):
        for model in TRTLLM_MODELS:
            self.repo_add(model, backend="tensorrtllm")
        pid = utils.run_server()
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        for model in TRTLLM_MODELS:
            self.model_infer(model, prompt=PROMPT, protocol=protocol)
            self.model_profile(model, backend="tensorrtllm", protocol=protocol)

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "VLLM", reason="Only run for VLLM image"
    )
    @pytest.mark.parametrize(
        "protocol",
        [
            "grpc",
            pytest.param(
                "http",
                marks=pytest.mark.xfail(
                    reason="http not supported decoupled models and model profiling yet"
                ),
            ),
        ],
    )
    def test_vllm_e2e(self, protocol, setup_and_teardown):
        for model in VLLM_MODELS:
            self.repo_add(model)
        pid = utils.run_server()
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        for model in VLLM_MODELS:
            self.model_infer(model, prompt=PROMPT, protocol=protocol)
            self.model_profile(model, protocol=protocol)

    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_non_llm(self, protocol, setup_and_teardown):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        model_repo = "test_models"
        pid = utils.run_server(repo=model_repo)
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        model = "add_sub"
        # infer should work without a prompt for non-LLM models
        self.model_infer(model, protocol=protocol)
        # profile should fail for non-LLM models
        with pytest.raises(Exception):
            if protocol == "http":
                pytest.xfail("Profiler does not support http protocol at this time")
            self.model_profile(model, protocol=protocol)

    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_mock_llm(self, protocol, setup_and_teardown):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        model_repo = "test_models"
        pid = utils.run_server(repo=model_repo)
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        model = "mock_llm"
        # infer should work with a prompt for LLM models
        self.model_infer(model, prompt=PROMPT, protocol=protocol)
        # infer should fail without a prompt for LLM models
        with pytest.raises(Exception):
            self.model_profile(model, protocol=protocol)
