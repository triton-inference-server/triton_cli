import argparse
from triton_cli.common import TritonCLIException


def build_engine(huggingface_id, engines_path):
    print("[DEBUG] STARTING ENGINE BUILD")
    from tensorrt_llm import BuildConfig, LLM

    # NOTE: Given config.json, can read from 'build_config' section and from_dict
    config = BuildConfig()
    # TODO: Expose more build args to user
    # TODO: Discuss LLM API BuildConfig defaults
    # NOTE: Using some defaults from trtllm-build because LLM API defaults are too low
    # config.max_input_len = 1024
    # config.max_seq_len = 8192
    # config.max_batch_size = 128

    engine = LLM(huggingface_id, build_config=config)
    # TODO: Investigate if LLM is internally saving a copy to a temp dir
    # Currently, models are being saved to /root/.cache/huggingface/hub/models--<model_name>
    engine.save(str(engines_path))

    del config, engine


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--huggingface_id", type=str)
    parser.add_argument("--engines_path", type=str)

    args = parser.parse_args()

    if not (args.huggingface_id or args.engines_path):
        raise TritonCLIException("Need value for --huggingface_id and --engines_path")

    build_engine(args.huggingface_id, args.engines_path)


if __name__ == "__main__":
    main()
