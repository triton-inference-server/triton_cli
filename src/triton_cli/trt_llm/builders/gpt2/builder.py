import os
from .scripts import hf_gpt_convert
from .scripts import build

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
        # FIXME: Due to ongoing issues with IFB, max_batch_size must also
        # be set to 1 in order to avoid the input dimension error.
        args = [
            "--model_dir",
            self.final_converted_weights_path,
            "--dtype",
            "float16",
            "--max_batch_size",
            "1",
            "--output_dir",
            self.engine_output_path,
        ]
        build.run_build(args)
