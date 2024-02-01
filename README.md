# Triton CLI

## Pre-requisites

> [!NOTE]
> When using Triton and related tools on your host, there are some system
> dependencies that may be required for various workflows. Most system dependency
> issues can be resolved by installing and running the CLI from within the latest
> corresponding `tritonserver` container image, which should have all necessary
> system dependencies installed.
>
> For vLLM and TRT-LLM, you can use their respective image:
> - `nvcr.io/nvidia/tritonserver:24.01-vllm-python-py3`
> - `nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3`
>
> If you decide to run the CLI on the host or in a custom image, you
> may encounter the following system dependency issues:
>
> 1. If you encounter an error related to `libb64.so` from `triton model profile`
> or `perf_analyzer` such as:
> ```
> perf_analyzer: error while loading shared libraries: libb64.so.0d
> ```
>
> Then you likely need to install this system dependency. For example:
> ```
> apt install libb64-dev
> ```
>
> 2. If you encounter an error related to `libcudart.so` from `triton model profile`
> or `perf_analyzer` such as:
> ```
> perf_analyzer: error while loading shared libraries: libcudart.so
> ```
>
> Then you likely need to install the CUDA toolkit or set your `LD_LIBRARY_PATH`
> correctly. Refer to: https://developer.nvidia.com/cuda-downloads.

## Installation

### Install from Pip

```
pip install triton_cli -U --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-dl-triton-pypi-local/simple
```

### Install from Source

```bash
# Clone repo, git mode for development/contribution
git clone git@github.com:triton-inference-server/triton_cli.git
cd triton_cli

# Should be pointing at directory containing pyproject.toml
pip install .
```

## Quickstart

```
# Explore the commands
triton -h

# Interact with a local model repository
triton repo -h

# Add a vLLM model to the model repository, downloaded from HuggingFace
triton repo add -m gpt2

# Start server pointing at the default model repository
triton server start

# Infer with CLI
triton model infer -m gpt2 --prompt "machine learning is"

# Infer with curl using the generate endpoint
curl -X POST localhost:8000/v2/models/gpt2/generate -d '{"text_input": "machine learning is", "max_tokens": 128}'

# Profile model with Perf Analyzer
triton model profile -m gpt2
```

## Examples

> [!NOTE]
> Usage of `llama-2-7b` requires authentication in Huggingface through either
> `huggingface-login` or setting the `HF_TOKEN` environment variable.

### Serving a vLLM Model

The following models have currently been tested for vLLM through the CLI:
- `gpt2`
- `llama-2-7b`
- `opt125m`
- `mistral-7b`
- `falcon-7b`

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

> [!NOTE]
> By default, TRT-LLM engines are generated in `/tmp/engines/{model_name}`,
> such as `/tmp/engines/gpt2`. They are intentionally kept outside of the model
> repository to improve re-usability across examples and repositories. This
> destination can be customized with the `ENGINE_DEST_PATH` environment variable.

#### Pre-requisites

(optional) If you don't want to install TRT-LLM dependencies on the host, you
can also run the following instructions inside of a container that is launched
with the following command:
```
# NOTE: Mounting the huggingface cache is optional, but will allow saving and
# re-using downloaded huggingface models across different runs and containers.
docker run -ti \
  --gpus all \
  --network=host \
  --shm-size=1g --ulimit memlock=-1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3
```

Install the TRT-LLM dependencies:
```
# Install TRT LLM building dependencies
pip install \
  "psutil" \
  "pynvml>=11.5.0" \
  "torch==2.1.2" \
  "tensorrt_llm==0.7.1" --extra-index-url https://pypi.nvidia.com/
```

#### Example

The following models are currently supported for automating TRT-LLM
engine builds through the CLI:
- `gpt2`
- `llama-2-7b`

```
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
