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
import pytest
from utils import TritonCommands


PROMPT = "machine learning is"

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_REPO = os.path.join(TEST_DIR, "test_models")


# Give ample 30min timeout for tests that download models from huggingface
# where network speed can be intermittent, for test consistency.
LLM_TIMEOUT_SECS = 1800


class TestE2E:
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
    @pytest.mark.timeout(LLM_TIMEOUT_SECS)
    def test_tensorrtllm_e2e(self, trtllm_server, protocol):
        # NOTE: TRTLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("TRTLLM_MODEL")
        assert model is not None, "TRTLLM_MODEL env var must be set!"
        # Source is optional if using a "known: model"
        source = os.environ.get("MODEL_SOURCE")
        TritonCommands._clear()
        TritonCommands._import(model, source=source, backend="tensorrtllm")
        trtllm_server.start()
        TritonCommands._infer(model, prompt=PROMPT, protocol=protocol)
        TritonCommands._profile(model, backend="tensorrtllm")

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.skipif(
        os.environ.get("TRTLLM_MODEL") == "gpt2",
        reason="gpt2's tokenizer doesn't have a chat template defined",
    )
    @pytest.mark.timeout(LLM_TIMEOUT_SECS)
    def test_tensorrtllm_openai_e2e(self, trtllm_openai_server):
        # NOTE: TRTLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("TRTLLM_MODEL")
        assert model is not None, "TRTLLM_MODEL env var must be set!"
        # Source is optional if using a "known: model"
        source = os.environ.get("MODEL_SOURCE")
        TritonCommands._clear()
        TritonCommands._import(model, source=source, backend="tensorrtllm")
        trtllm_openai_server.start()
        TritonCommands._profile(
            model, service_kind="openai", endpoint_type="chat", url="localhost:9000"
        )

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
    @pytest.mark.timeout(LLM_TIMEOUT_SECS)
    def test_vllm_e2e(self, vllm_server, protocol):
        # NOTE: VLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("VLLM_MODEL")
        assert model is not None, "VLLM_MODEL env var must be set!"
        # Source is optional if using a "known: model"
        source = os.environ.get("MODEL_SOURCE")
        TritonCommands._clear()
        TritonCommands._import(model, source=source)
        vllm_server.start()
        TritonCommands._infer(model, prompt=PROMPT, protocol=protocol)
        TritonCommands._profile(model, backend="vllm")

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "VLLM", reason="Only run for VLLM image"
    )
    @pytest.mark.skipif(
        os.environ.get("VLLM_MODEL") == "gpt2",
        reason="gpt2's tokenizer doesn't have a chat template defined",
    )
    @pytest.mark.timeout(LLM_TIMEOUT_SECS)
    def test_vllm_openai_e2e(self, vllm_openai_server):
        # NOTE: VLLM test models will be passed by the testing infrastructure.
        # Only a single model will be passed per test to enable tests to run concurrently.
        model = os.environ.get("VLLM_MODEL")
        assert model is not None, "VLLM_MODEL env var must be set!"
        # Source is optional if using a "known: model"
        source = os.environ.get("MODEL_SOURCE")
        TritonCommands._clear()
        TritonCommands._import(model, source=source)
        vllm_openai_server.start()
        TritonCommands._profile(
            model, service_kind="openai", endpoint_type="chat", url="localhost:9000"
        )

    @pytest.mark.skipif(
        os.environ.get("CI_PIPELINE") == "GITHUB_ACTIONS",
        reason="bandage/temporary fix",
    )
    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_non_llm(self, simple_server, protocol):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        simple_server.start()
        model = "add_sub"
        # infer should work without a prompt for non-LLM models
        TritonCommands._infer(model, protocol=protocol)

    @pytest.mark.skipif(
        os.environ.get("CI_PIPELINE") == "GITHUB_ACTIONS",
        reason="bandage/temporary fix",
    )
    @pytest.mark.parametrize("protocol", ["grpc", "http"])
    def test_mock_llm(self, simple_server, protocol):
        # This test runs on the default Triton image, as well as on both TRT-LLM and VLLM images.
        # Use the existing models.
        simple_server.start()
        model = "mock_llm"
        # infer should work with a prompt for LLM models
        TritonCommands._infer(model, prompt=PROMPT, protocol=protocol)
        # infer should fail without a prompt for LLM models
        with pytest.raises(Exception):
            TritonCommands._infer(model, protocol=protocol)

        # profile for triton endpoints only supports grpc protocol currently
        if protocol == "grpc":
            # profile should work without a prompt for LLM models
            TritonCommands._profile(model, backend="tensorrtllm")
