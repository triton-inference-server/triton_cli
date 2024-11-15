from tensorrt_llm import LLM, BuildConfig
from psutil import Popen

# NOTE: Given config.json, can read from 'build_config' section and from_dict
config = BuildConfig()
# TODO: Expose more build args to user
# TODO: Discuss LLM API BuildConfig defaults
# NOTE: Using some defaults from trtllm-build because LLM API defaults are too low
config.max_input_len = 1024
config.max_seq_len = 8192
config.max_batch_size = 128

engine = LLM("hf:gpt2", build_config=config)
# TODO: Investigate if LLM is internally saving a copy to a temp dir
# Currently, models are being saved to /root/.cache/huggingface/hub/models--<model_name>
engine.save("/tmp/engines")

Popen(["tritonserver", "--model-repo", "/root/models"])

# Hope for no Cuda OOM Issues
