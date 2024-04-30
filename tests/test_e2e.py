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

import os
import pytest
from triton_cli.main import run
import utils


PROMPT = "machine learning is"

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_REPO = os.path.join(TEST_DIR, "test_models")


class TestE2E:
    def _clear(self):
        args = ["remove", "-m", "all"]
        run(args)

    def _import(self, model, source=None, backend=None, repo=None):
        args = ["import", "-m", model]
        if source:
            args += ["--source", source]
        if backend:
            args += ["--backend", backend]
        if repo:
            args += ["--repo", repo]
        run(args)

    def _infer(self, model, prompt=None, protocol=None):
        args = ["infer", "-m", model]
        if prompt:
            args += ["--prompt", prompt]
        if protocol:
            args += ["-i", protocol]
        run(args)

    def _profile(self, model, protocol=None, backend=None, is_llm=False):
        args = ["profile", "-m", model]
        if backend:
            args += ["--backend", backend]
        if is_llm:
            args += ["--task", "llm"]
        else:
            if protocol:
                args += ["-i", protocol]
        run(args)

    def _optimize(self, model):
        args = [
            "optimize",
            "profile",
            "--profile-models",
            model,
            "--triton-launch-mode",
            "local",
            "--model-repository",
            "test_models",
            "--override-output-model-repository",
        ]
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
        self._clear()

        yield kill_server

        # Teardown after the test case is done.
        kill_server.kill_server()
        self._clear()

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.parametrize(
        "protocol",
        [
            "grpc",
            pytest.param(
                "http",
                # NOTE: skip because xfail was causing server to not get cleaned up by test in background
                marks=pytest.mark.skip(
                    reason="http does not support model infer and model profile for decoupled models"
                ),
            ),
        ],
    )
    @pytest.mark.timeout(600)
    def test_tensorrtllm_e2e(self, protocol, setup_and_teardown):
        # NOTE: TRTLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("TRTLLM_MODEL")
        assert model is not None, "TRTLLM_MODEL env var must be set!"
        self._import(model, backend="tensorrtllm")
        pid = utils.run_server()
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        self._infer(model, prompt=PROMPT, protocol=protocol)
        self._profile(model, backend="tensorrtllm", is_llm=True)

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "VLLM", reason="Only run for VLLM image"
    )
    @pytest.mark.parametrize(
        "protocol",
        [
            "grpc",
            pytest.param(
                "http",
                # NOTE: skip because xfail was causing server to not get cleaned up by test in background
                marks=pytest.mark.skip(
                    reason="http not supported decoupled models and model profiling yet"
                ),
            ),
        ],
    )
    @pytest.mark.timeout(600)
    def test_vllm_e2e(self, protocol, setup_and_teardown):
        # NOTE: VLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("VLLM_MODEL")
        assert model is not None, "VLLM_MODEL env var must be set!"
        self._import(model)
        pid = utils.run_server()
        setup_and_teardown.pid = pid
        # vLLM will download the model on the fly, so give it a big timeout
        utils.wait_for_server_ready(timeout=300)

        self._infer(model, prompt=PROMPT, protocol=protocol)
        self._profile(model, backend="vllm", is_llm=True)

    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_non_llm(self, protocol, setup_and_teardown):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        pid = utils.run_server(repo=MODEL_REPO)
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        model = "add_sub"
        # infer should work without a prompt for non-LLM models
        self._infer(model, protocol=protocol)
        self._profile(model, protocol=protocol)

        # Model Analyzer will start a new instance of Triton, so shut down the previous one.
        utils.kill_server(pid)
        setup_and_teardown.pid = None
        self._optimize(model)

    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_mock_llm(self, protocol, setup_and_teardown):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        pid = utils.run_server(repo=MODEL_REPO)
        setup_and_teardown.pid = pid
        utils.wait_for_server_ready()

        model = "mock_llm"
        # infer should work with a prompt for LLM models
        self._infer(model, prompt=PROMPT, protocol=protocol)
        # infer should fail without a prompt for LLM models
        with pytest.raises(Exception):
            self._infer(model, protocol=protocol)
        # profile should work without a prompt for LLM models
        self._profile(model, protocol=protocol, backend="tensorrtllm", is_llm=True)
