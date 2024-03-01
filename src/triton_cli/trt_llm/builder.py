import importlib
import subprocess

CHECKPOINT_MODULE_MAP = {
    "meta-llama/Llama-2-7b-hf": "llama",
    "facebook/opt-125m": "opt",
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
        self.__convert_weights()

        # NOTE: These are for IFB. Omit these args for V1 engines.
        ifb_args = [
            "--gpt_attention_plugin=float16",
            "--paged_kv_cache=enable",
            "--remove_input_padding=enable",
        ]

        # TODO: Expose configurability
        int8_args = [
            "--weight_only_precision=int8",
            # INT8 KV Cache requires calibration data (scaling factors)
            # "--int8_kv_cache",
        ]

        build_args = [
            f"--checkpoint_dir={self.converted_weights_path}",
            "--max_batch_size=1",
            "--context_fmha=enable",
            *ifb_args,
            *int8_args,
            f"--output_dir={self.engine_output_path}",
        ]
        subprocess.run(["trtllm-build"] + build_args)

    # NOTE: This function should be removed once 'trtllm-build' is
    # capable of converting the weights internally.
    def __convert_weights(self):
        weight_conversion_args = [
            "--model_dir",
            self.hf_download_path,
            "--output_dir",
            self.converted_weights_path,
            "--dtype=float16",
        ]
        convert_weights_fn = importlib.import_module(
            f"triton_cli.trt_llm.checkpoint_scripts.{self.checkpoint_id}.convert_checkpoint"
        ).main
        convert_weights_fn(weight_conversion_args)
