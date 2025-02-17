# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from triton_cli.parser import KNOWN_MODEL_SOURCES, parse_args
from triton_cli.common import TritonCLIException
from utils import TritonCommands, ScopedTritonServer

KNOWN_MODELS = KNOWN_MODEL_SOURCES.keys()
KNOWN_SOURCES = KNOWN_MODEL_SOURCES.values()
TEST_REPOS = [None, os.path.join("tmp", "models")]

CUSTOM_VLLM_MODEL_SOURCES = [("vllm-model", "hf:gpt2")]

CUSTOM_TRTLLM_MODEL_SOURCES = [("trtllm-model", "hf:gpt2")]

CUSTOM_LLMAPI_MODEL_SOURCES = [("llmapi-model", "hf:gpt2")]

# TODO: Add public NGC model for testing
CUSTOM_NGC_MODEL_SOURCES = [("my-llm", "ngc:does-not-exist")]


PROMPT = "machine learning is"

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_REPO = os.path.join(TEST_DIR, "test_models")


class TestRepo:
    def test_noargs(self):
        # Test that running 'triton' with no args will fail with a parsing error
        # NOTE: pytest needs to be run with `--capture=sys` to read the output,
        #       so just assert the error code and exception type for now.
        args = []
        with pytest.raises(SystemExit) as excinfo:
            TritonCommands._run_and_capture_stdout(args)

        # argparse returns code 2 for error / missing arguments
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_clear(self, repo):
        TritonCommands._clear(repo)

    # TODO: Add pre/post repo clear to a fixture for setup/teardown
    @pytest.mark.parametrize("model", KNOWN_MODELS)
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_import_known_model(self, model, repo):
        TritonCommands._clear(repo)
        TritonCommands._import(model, repo=repo)
        TritonCommands._clear(repo)

    @pytest.mark.parametrize("source", KNOWN_SOURCES)
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_import_known_source(self, source, repo):
        TritonCommands._clear(repo)
        TritonCommands._import("known_source", source=source, repo=repo)
        TritonCommands._clear(repo)

    @pytest.mark.parametrize("model,source", CUSTOM_VLLM_MODEL_SOURCES)
    def test_import_vllm(self, model, source):
        TritonCommands._clear()
        TritonCommands._import(model, source=source)
        # TODO: Parse repo to find model, with vllm backend in config
        TritonCommands._clear()

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.parametrize("model,source", CUSTOM_TRTLLM_MODEL_SOURCES)
    def test_repo_add_trtllm_build(self, model, source):
        # TODO: Parse repo to find TRT-LLM models and backend in config
        TritonCommands._clear()
        TritonCommands._import(model, source=source, backend="tensorrtllm")
        TritonCommands._clear()

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "LLMAPI", reason="Only run for LLMAPI image"
    )
    @pytest.mark.parametrize("model,source", CUSTOM_TRTLLM_MODEL_SOURCES)
    def test_repo_add_llmapi_build(self, model, source):
        # TODO: Parse repo to find TRT-LLM models and backend in config
        TritonCommands._clear()
        TritonCommands._import(model, source=source, backend="llmapi")
        TritonCommands._clear()

    @pytest.mark.skip(reason="Pre-built TRT-LLM engines not available")
    def test_import_trtllm_prebuilt(self, model, source):
        # TODO: Parse repo to find TRT-LLM models and backend in config
        pass

    def test_import_no_source(self):
        # TODO: Investigate idiomatic way to assert failures for CLIs
        with pytest.raises(
            Exception, match="Please use a known model, or provide a --source"
        ):
            TritonCommands._import("no_source", source=None)

    def test_remove(self):
        TritonCommands._import("gpt2", source="hf:gpt2")
        TritonCommands._remove("gpt2")

    def test_remove_nonexistent(self):
        with pytest.raises(TritonCLIException, match="No model folder exists"):
            TritonCommands._remove("does-not-exist")

    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_list(self, repo):
        TritonCommands._list(repo)

    # This test uses mock system args and a mock subprocess call
    # to ensure that the correct subprocess call is made for profile.
    def test_triton_profile(self, mocker, monkeypatch):
        test_args = ["triton", "profile", "-m", "add_sub"]
        mock_run = mocker.patch("subprocess.run")
        monkeypatch.setattr("sys.argv", test_args)
        args = parse_args()
        args.func(args)
        mock_run.assert_called_once_with(
            ["genai-perf", "profile", "-m", "add_sub"], check=True
        )

    @pytest.mark.parametrize("model", ["mock_llm"])
    def test_triton_metrics(self, model):
        # Import the Model Repo
        with ScopedTritonServer(repo=MODEL_REPO):
            metrics_before = TritonCommands._metrics()

            # Before Inference, Verifying Inference Count == 0
            for loaded_models in metrics_before["nv_inference_request_success"][
                "metrics"
            ]:
                if loaded_models["labels"]["model"] == model:  # If mock_llm
                    assert loaded_models["value"] == 0

            # Model Inference
            TritonCommands._infer(model, prompt=PROMPT)

            metrics_after = TritonCommands._metrics()

            # After Inference, Verifying Inference Count == 1
            for loaded_models in metrics_after["nv_inference_request_success"][
                "metrics"
            ]:
                if loaded_models["labels"]["model"] == model:  # If mock_llm
                    assert loaded_models["value"] == 1

    @pytest.mark.parametrize("model", ["mock_llm"])
    def test_triton_config(self, model):
        # Import the Model
        with ScopedTritonServer(repo=MODEL_REPO):
            config = TritonCommands._config(model)
            # Checks if correct model is loaded
            assert config["name"] == model

    @pytest.mark.parametrize("model", ["mock_llm"])
    def test_triton_status(self, model):
        # Import the Model
        with ScopedTritonServer(repo=MODEL_REPO):
            status = TritonCommands._status(protocol="grpc")
            # Checks if model(s) are live and ready
            assert status["live"] and status["ready"]
