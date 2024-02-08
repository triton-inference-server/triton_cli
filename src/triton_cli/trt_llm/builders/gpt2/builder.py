import os

import logging
from triton_cli.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class GPTBuilder:
    def __init__(self, tokenizer_path: str, engine_output_path: str):
        self.tokenizer_path = tokenizer_path
        self.engine_output_path = engine_output_path
        self.converted_weights_path = engine_output_path + "/c-model/gpt2"

        # NOTE: Converted weights will not actually be placed at the path above. Instead, they
        # will be placed at the path above + {tensor_parallelism}-gpu/.
        self.tensor_parallelism = 1
        self.final_converted_weights_path = (
            self.converted_weights_path + f"/{self.tensor_parallelism}-gpu"
        )

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self):
        # NOTE: Moving imports internally to allow top-level modules to freely import builders
        from .scripts import hf_gpt_convert
        from .scripts import build

        # NOTE: Due to multiprocess logic in the conversion script, 'hf_gpt_convert.py' must
        # be called as if called from the command line.
        weight_conversion_script = hf_gpt_convert.__file__
        weight_conversion_args = [
            " -i",
            "gpt2",
            "-o",
            self.converted_weights_path,
            "--tensor-parallelism",
            str(self.tensor_parallelism),
            "--storage-type float16",
        ]
        os.system(
            "python3 " + weight_conversion_script + " ".join(weight_conversion_args)
        )

        logger.debug("GPT builder has successfully converted weights from HF to FT")

        # NOTE: These are for IFB. Omit these args for V1 engines.
        ifb_args = [
            "--use_gpt_attention_plugin=float16",
            "--paged_kv_cache",
            "--remove_input_padding",
        ]

        # TODO: Expose configurability
        # int8_args = [
        #    "--use_weight_only",
        #    "--weight_only_precision=int8",
        #    # INT8 KV Cache requires calibration data (scaling factors)
        #    # "--int8_kv_cache",
        # ]

        args = [
            "--model_dir",
            self.final_converted_weights_path,
            # Spawning 1 instance of bls/pre/post models for each batch size
            # significantly increases startup time. Keep GPT2 as batch size 1
            # for simpler demo output and speed purposes.
            "--max_batch_size=1",
            "--dtype=float16",
            *ifb_args,
            # NOTE: GPT2 emits a lot of warnings for INT8 build when it finds
            # invalid kernel configurations. This should be fixed in a later
            # release. Disabling int8 for GPT2 for easier demo output purposes.
            # *int8_args,
            "--output_dir",
            self.engine_output_path,
        ]
        build.run_build(args)
