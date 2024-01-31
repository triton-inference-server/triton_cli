class LlamaBuilder:
    def __init__(self, tokenizer_path, engine_output_path):
        self.tokenizer_path = tokenizer_path
        self.engine_output_path = engine_output_path

    # TODO: User should be able to specify a what parameters they want to use to build a
    # TRT LLM engine. A input JSON should be suitable for this goal.
    def build(self):
        from .scripts import build

        # NOTE: These are for IFB. Omit these args for V1 engines.
        ifb_args = [
            "--use_gpt_attention_plugin=float16",
            "--paged_kv_cache",
            "--remove_input_padding",
        ]

        # TODO: Expose configurability
        int8_args = [
            "--use_weight_only",
            "--weight_only_precision=int8",
            # INT8 KV Cache requires calibration data (scaling factors)
            # "--int8_kv_cache",
        ]

        args = [
            "--model_dir",
            self.tokenizer_path,
            "--max_batch_size=64",
            "--dtype=float16",
            "--enable_context_fmha",
            *ifb_args,
            *int8_args,
            "--output_dir",
            self.engine_output_path,
        ]
        build.run_build(args)
