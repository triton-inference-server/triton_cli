from .scripts import build


class GPTBuilder:
    def __init__(self, tokenizer_path, engine_output_path):
        self.tokenizer_path = tokenizer_path
        self.engine_output_path = engine_output_path

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self):
        # FIXME: Due to ongoing issues with IFB, max_batch_size must also
        # be set to 1 in order to avoid the input dimension error.
        args = [
            "--model_dir",
            self.tokenizer_path,
            "--dtype",
            "float16",
            "--max_batch_size",
            "1",
            "--output_dir",
            self.engine_output_path,
        ]
        build.run_build(args)
