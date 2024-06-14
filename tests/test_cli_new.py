import os
import pytest
from triton_cli.parser import KNOWN_MODEL_CONFIGS, parse_args

# from triton_cli.common import TritonCLIException
from triton_cli.repository import ImportConfig
from utils import TritonCommands, ScopedTritonServer
# from pathlib import Path


TEST_REPOS = [None, os.path.join("tmp", "models")]
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_REPO = os.path.join(TEST_DIR, "test_models")
TEST_MODEL_CONFIG_DIR = os.path.join(TEST_DIR, "test_configs")

KNOWN_MODELS = list(KNOWN_MODEL_CONFIGS.keys())
# VLLM_CONFIG_FILEPATH = TEST_MODEL_CONFIG_DIR / "config-gpt2-vllm.yaml"
# TRTLLM_CONFIG_FILEPATH = TEST_MODEL_CONFIG_DIR / "config-gpt2-trtllm.yaml"

# CUSTOM_VLLM_CONFIG = ImportConfig(VLLM_CONFIG_FILEPATH, None)
# CUSTOM_TRTLLM_CONFIG = ImportConfig(TRTLLM_CONFIG_FILEPATH, None)

CUSTOM_VLLM_CONFIG = os.path.join(TEST_MODEL_CONFIG_DIR, "config-gpt2-vllm.yaml")
CUSTOM_TRTLLM_CONFIG = os.path.join(TEST_MODEL_CONFIG_DIR, "config-gpt2-trtllm.yaml")

# TODO: Add public NGC model for testing
CUSTOM_NGC_MODEL_SOURCES = [("my-llm", "ngc:does-not-exist")]

PROMPT = "machine learning is"


class TestRepo:
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_clear(self, repo):
        TritonCommands._clear(repo)

    @pytest.mark.parametrize("model", KNOWN_MODELS)
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_import_known_model(self, model, repo):
        TritonCommands._clear(repo)  # Ensuring clean /root/models/

        TritonCommands._import(model=model, repo=repo)

        TritonCommands._clear(repo)  # Removing imported models

    @pytest.mark.parametrize("config_filepath", [CUSTOM_VLLM_CONFIG])
    def test_import_vllm(self, config_filepath):
        TritonCommands._clear()
        config = ImportConfig(config_filepath)
        TritonCommands._import(config["model"])

        # TODO: Parse repo to find model, with vllm backend in config
        TritonCommands._clear()

    @pytest.mark.skipif(
        os.environ.get("IMAGE_KIND") != "TRTLLM", reason="Only run for TRT-LLM image"
    )
    @pytest.mark.parametrize("config_filepath", [CUSTOM_TRTLLM_CONFIG])
    def test_repo_add_trtllm_build(self, config_filepath):
        TritonCommands._clear()
        config = ImportConfig(config_filepath)
        TritonCommands._import(config["model"])
        TritonCommands._clear()

    @pytest.mark.skip(reason="Pre-built TRT-LLM engines not available")
    def test_import_trtllm_prebuilt(self, model, source):
        with pytest.raises(
            Exception, match="Please use a known model, or provide a --source"
        ):
            TritonCommands._import("no_source", source=None)

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
        mock_run.assert_called_once_with(["genai-perf", "-m", "add_sub"], check=True)

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
