# triton_cli

## Quickstart

```
# Clone repo, git mode for development/contribution
git clone git@github.com:triton-inference-server/triton_cli.git
cd triton_cli

# Should be pointing at directory containing pyproject.toml
pip install .

# Try it out
triton -h
```

## Examples

### Serving a vLLM Model

```
# Generate a Triton model repository containing a vLLM model config
triton repo clear
triton repo add -m gpt2 --backend vllm

# Start Triton pointing at the default model repository
triton server start

# Interact with model
triton model infer -m gpt2 --prompt "machine learning is"

# Profile model with Perf Analyzer
triton model profile -m gpt2
```

### Serving a TRT-LLM Model

**NOTE**: By default, TRT-LLM engines are generated in `/tmp/engines/{model_name}`,
such as `/tmp/engines/gpt2`. They are intentionally kept outside of the model
repository to improve re-usability across examples and repositories.

```
# Install TRT LLM building dependencies
pip install \
  "psutil" \
  "pynvml>=11.5.0" \
  "torch==2.1.2" \
  "tensorrt_llm==0.7.1" --extra-index-url https://pypi.nvidia.com/

# Build TRT LLM engine and generate a Triton model repository pointing at it
triton repo clear
triton repo add -m gpt2 --backend tensorrtllm

# Start Triton pointing at the default model repository
triton server start

# Interact with model
triton model infer -m gpt2 --prompt "machine learning is"

# Profile model with Perf Analyzer
triton model profile -m gpt2 --backend tensorrtllm
```
