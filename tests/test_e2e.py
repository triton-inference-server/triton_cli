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

    def model_infer(self, model, prompt, protocol=None):
        args = ["model", "infer", "-m", model, "--prompt", prompt]
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

    @pytest.fixture
    def setup_and_teardown(self):
        pids = []
        self.repo_clear()
        yield pids
        for pid in pids:
            utils.kill_server(pid)
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
        setup_and_teardown.append(pid)
        utils.wait_for_server_ready()

        for model in TRTLLM_MODELS:
            self.model_infer(model, PROMPT, protocol=protocol)
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
        setup_and_teardown.append(pid)
        utils.wait_for_server_ready()

        for model in VLLM_MODELS:
            self.model_infer(model, PROMPT, protocol=protocol)
            self.model_profile(model, protocol=protocol)
