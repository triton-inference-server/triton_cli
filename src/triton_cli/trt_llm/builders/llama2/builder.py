from .scripts import build


class LlamaBuilder:
    def __init__(self):
        self.tokenizer_path = tokenizer_path
        self.engine_output_path = engine_output_path

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self, tokenizer_path, engine_output_path):
        # FIXME: Due to ongoing issues with IFB, max_batch_size must also
        # be set to 1 in order to avoid the input dimension error.
        args = [
            "--model_dir",
            self.tokenizer_path,
            "--dtype",
            "float16",
            "--enable_context_fmha",
            "--max_batch_size",
            "1",
            "--use_gemm_plugin",
            "float16",
            "--output_dir",
            self.engine_output_path,
        ]
        build.run_build(args)
