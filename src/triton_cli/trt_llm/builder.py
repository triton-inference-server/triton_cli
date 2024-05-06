import logging
import subprocess
from pathlib import Path

from triton_cli.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

CHECKPOINT_MODULE_MAP = {
    "meta-llama/Llama-2-7b-hf": "llama",
    "meta-llama/Llama-2-7b-chat-hf": "llama",
    "meta-llama/Meta-Llama-3-8B": "llama",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama",
    "facebook/opt-125m": "opt",
    "gpt2": "gpt2",
}


class TRTLLMBuilder:
    def __init__(self, huggingface_id, hf_download_path, engine_output_path):
        self.checkpoint_id = CHECKPOINT_MODULE_MAP[huggingface_id]
        self.hf_download_path = hf_download_path
        self.converted_weights_path = self.hf_download_path + "/converted_weights"
        self.engine_output_path = engine_output_path

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self):
        self._convert_checkpoint()
        self._trtllm_build()

    # NOTE: This function should be removed once 'trtllm-build' is
    # capable of converting the weights internally.
    def _convert_checkpoint(self):
        if Path(self.converted_weights_path).exists():
            logger.info(
                f"Converted weights path {self.converted_weights_path} already exists, skipping checkpoint conversion."
            )
            return

        weight_conversion_args = [
            "--model_dir",
            self.hf_download_path,
            "--output_dir",
            self.converted_weights_path,
            "--dtype=float16",
        ]

        ckpt_script = (
            Path(__file__).resolve().parent
            / "checkpoint_scripts"
            / self.checkpoint_id
            / "convert_checkpoint.py"
        )
        cmd = ["python3", str(ckpt_script)] + weight_conversion_args
        cmd_str = " ".join(cmd)
        logger.info(f"Running '{cmd_str}'")
        subprocess.run(cmd, check=True)

    def _trtllm_build(self):
        # TODO: Move towards config-driven build args per-model
        build_args = [
            f"--checkpoint_dir={self.converted_weights_path}",
            f"--output_dir={self.engine_output_path}",
            "--gpt_attention_plugin=float16",
            "--gemm_plugin=float16",
        ]

        cmd = ["trtllm-build"] + build_args
        cmd_str = " ".join(cmd)
        logger.info(f"Running '{cmd_str}'")
        subprocess.run(cmd, check=True)
