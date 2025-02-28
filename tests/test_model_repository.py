# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from triton_cli.server.server_utils import TRTLLMUtils, VLLMUtils, LLMAPIUtils
from triton_cli.common import DEFAULT_MODEL_REPO

# Give ample 30min timeout for tests that download models from huggingface
# where network speed can be intermittent, for test consistency.
DOWNLOAD_TIMEOUT_SECS = 1800


class TestModelRepository:
    def setup_method(self):
        TritonCommands._clear()

    def test_can_not_find_models(self):
        trtllm_utils = TRTLLMUtils(DEFAULT_MODEL_REPO)
        vllm_utils = VLLMUtils(DEFAULT_MODEL_REPO)
        llmapi_utils = LLMAPIUtils(DEFAULT_MODEL_REPO)
        assert not trtllm_utils.has_trtllm_model(), f"tensorrtllm model found in model repository: '{DEFAULT_MODEL_REPO}', but the test expect the tensorrtllm model not found"
        assert not vllm_utils.has_vllm_model(), f"vllm model found in model repository: '{DEFAULT_MODEL_REPO}', but the test expect the vllm model not found"
        assert not llmapi_utils.has_llmapi_model(), f"llmapi model found in model repository: '{DEFAULT_MODEL_REPO}', but the test expect the llmapi model not found"

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.timeout(DOWNLOAD_TIMEOUT_SECS)
    def test_can_get_tensorrtllm_engine_folder_from_model_repository(self):
        model = "llama-2-7b-chat"
        expected_engine_path = "/tmp/engines/llama-2-7b-chat"
        TritonCommands._import(model, backend="tensorrtllm")
        trtllm_utils = TRTLLMUtils(DEFAULT_MODEL_REPO)
        assert (
            trtllm_utils.has_trtllm_model()
        ), f"no tensorrtllm model found in model repository: '{DEFAULT_MODEL_REPO}'."
        assert (
            expected_engine_path == trtllm_utils.get_engine_path()
        ), f"engine path found is not as expected. Expected: {expected_engine_path}. Found: {trtllm_utils.get_engine_path()}"

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for VLLM image"
    )
    @pytest.mark.timeout(DOWNLOAD_TIMEOUT_SECS)
    def test_can_get_llmapi_model_id_from_model_repository(self):
        model = "llama-2-7b-chat"
        expected_model_id = "meta-llama/Llama-2-7b-chat-hf"
        TritonCommands._import(model, backend="llmapi")
        llmapi_utils = LLMAPIUtils(DEFAULT_MODEL_REPO)
        assert (
            llmapi_utils.has_llmapi_model()
        ), f"no LLM API model found in model repository: '{DEFAULT_MODEL_REPO}'."
        assert (
            expected_model_id == llmapi_utils.get_llmapi_model_huggingface_id_or_path()
        ), f"model id found is not as expected. Expected: {expected_model_id}. Found: {llmapi_utils.get_llmapi_model_huggingface_id_or_path()}"

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "VLLM", reason="Only run for VLLM image"
    )
    @pytest.mark.timeout(DOWNLOAD_TIMEOUT_SECS)
    def test_can_get_vllm_model_id_from_model_repository(self):
        model = "llama-2-7b-chat"
        expected_model_id = "meta-llama/Llama-2-7b-chat-hf"
        TritonCommands._import(model, backend="vllm")
        vllm_utils = VLLMUtils(DEFAULT_MODEL_REPO)
        assert (
            vllm_utils.has_vllm_model()
        ), f"no vllm model found in model repository: '{DEFAULT_MODEL_REPO}'."
        assert (
            expected_model_id == vllm_utils.get_vllm_model_huggingface_id_or_path()
        ), f"model id found is not as expected. Expected: {expected_model_id}. Found: {vllm_utils.get_vllm_model_huggingface_id_or_path()}"
