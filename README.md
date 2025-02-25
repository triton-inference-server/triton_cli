<!--
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. -->

# Triton Command Line Interface (Triton CLI)
> [!NOTE]
> Triton CLI is currently in BETA. Its features and functionality are likely
> to change as we collect feedback. We're excited to hear any thoughts you
> have (especially if you find the tool useful) and what features you'd like
> to see!

Triton CLI is an open source command line interface that enables users to
create, deploy, and profile models served by the Triton Inference
Server.

## Table of Contents

| [Pre-requisites](#pre-requisites) | [Installation](#installation) | [Quickstart](#quickstart) | [Serving LLM Models](#serving-llm-models) | [Serving a vLLM Model](#serving-a-vllm-model) | [Serving a TRT-LLM Model](#serving-a-trt-llm-model) | [Serving a LLM model with OpenAI API](#serving-a-llm-model-with-openai-api) | [Additional Dependencies for Custom Environments](#additional-dependencies-for-custom-environments) | [Known Limitations](#known-limitations) |

## Pre-requisites

When using Triton and related tools on your host (outside of a Triton container
image), there are a number of additional dependencies that may be required for
various workflows. Most system dependency issues can be resolved by installing
and running the CLI from within the latest corresponding `tritonserver`
container image, which should have all necessary system dependencies installed.

For vLLM and TRT-LLM, you can use their respective images:
- `nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3`
- `nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3`

If you decide to run the CLI on the host or in a custom image, please
see this list of [additional dependencies](#additional-dependencies-for-custom-environments)
you may need to install.


## Installation

Currently, Triton CLI can only be installed from source, with plans to host a
pip wheel soon. When installing Triton CLI, please be aware of the versioning
matrix below:

| Triton CLI Version | TRT-LLM Version | Triton Container Tag |
|:------------------:|:---------------:|:--------------------:|
| 0.1.2  | v0.17.0.post1 | 25.01 |
| 0.1.1  | v0.14.0 | 24.10 |
| 0.1.0  | v0.13.0 | 24.09 |
| 0.0.11 | v0.12.0 | 24.08 |
| 0.0.10 | v0.11.0 | 24.07 |
| 0.0.9  | v0.10.0 | 24.06 |
| 0.0.8  | v0.9.0  | 24.05 |
| 0.0.7  | v0.9.0  | 24.04 |
| 0.0.6  | v0.8.0  | 24.02, 24.03 |
| 0.0.5  | v0.7.1  | 24.01 |

### Install from GitHub

Install latest from `main` branch:

```bash
pip install git+https://github.com/triton-inference-server/triton_cli.git
```

It is also possible to install from a specific branch name, a commit hash
or a tag name. For example to install `triton_cli` with a specific tag:

```bash
GIT_REF="0.1.2"
pip install git+https://github.com/triton-inference-server/triton_cli.git@${GIT_REF}
```

### Install from Source

```bash
# Clone repo for development/contribution
git clone https://github.com/triton-inference-server/triton_cli.git
cd triton_cli

# Should be pointing at directory containing pyproject.toml
pip install .
```

## Quickstart
The instructions below outline the process of deploying a simple `gpt2`
model using Triton's [vLLM backend](https://github.com/triton-inference-server/vllm_backend).
If you are not in an environment where the `tritonserver` executable is
present, Triton CLI will automatically generate and run a custom image
capable of serving the model. This behavior is subject to change.

> [!NOTE]
> `triton start` is a blocking command and will stream server logs to the
> current shell. To interact with the running server, you will need to start
> a separate shell and `docker exec` into the running container if using one.

```bash
# Explore the commands
triton -h

# Add a vLLM model to the model repository, downloaded from HuggingFace
triton import -m gpt2

# Start server pointing at the default model repository
triton start --image nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3

# Infer with CLI
triton infer -m gpt2 --prompt "machine learning is"

# Infer with curl using the generate endpoint
curl -X POST localhost:8000/v2/models/gpt2/generate -d '{"text_input": "machine learning is", "max_tokens": 128}'
```

## Serving LLM Models

Triton CLI simplifies the workflow to deploy and interact with LLM models.
The steps below illustrate how to serve a vLLM or TRT-LLM model from scratch in
minutes.

> [!NOTE]
> Mounting the huggingface cache into the docker containers is optional, but will
> allow saving and re-using downloaded huggingface models across different runs
> and containers.
>
> ex: `docker run -v ${HOME}/.cache/huggingface:/root/.cache/huggingface ...`
>
> Also, usage of certain restricted models like Llama models requires authentication
> in Huggingface through either `huggingface-cli login` or setting the `HF_TOKEN`
> environment variable.
>
> If your huggingface cache is not located at `${HOME}/.cache/huggingface`, you could
> set the huggingface cache with
>
> ex: `export HF_HOME=path/to/your/huggingface/cache`
>

### Model Sources

<!-- TODO: Add more docs on commands, such as a doc on `import` behavior/args -->

The `triton import` command helps automate the process of creating a model repository
to serve with Triton Inference Server. When preparing models, a `--source` is required
to point at the location containing a model/weights. This argument is overloaded to support
a few types of locations:
- HuggingFace (`--source hf:<HUGGINGFACE_ID>`)
- Local Filesystem (`--source local:</path/to/model>`)

#### Model Source Aliases

<!-- TODO: Put known model sources into a JSON file or something separate from the code -->

For convenience, the Triton CLI supports short aliases for a handful
of models which will automatically set the correct `--source` for you.
A full list of aliases can be found from `KNOWN_MODEL_SOURCES` within `parser.py`,
but some examples can be found below:
- `gpt2`
- `opt125m`
- `mistral-7b`
- `llama-2-7b-chat`
- `llama-3-8b-instruct`
- `llama-3.1-8b-instruct`

For example, this command will go get Llama 3.1 8B Instruct from HuggingFace:
```bash
triton import -m llama-3.1-8b-instruct

# Equivalent command without alias:
# triton import --model llama-3.1-8b-instruct --source "hf:meta-llama/Llama-3.1-8B-Instruct"
```

For full control and flexibility, you can always manually specify the `--source`.

### Serving a vLLM Model

vLLM models will be downloaded at runtime when starting the server if not found
locally in the HuggingFace cache. No offline engine building step is required,
but you can pre-download the model in advance to avoid downloading at server
startup time.

The following models are supported by vLLM: https://docs.vllm.ai/en/latest/models/supported_models.html

#### Example

```bash
docker run -ti \
  --gpus all \
  --network=host \
  --shm-size=1g --ulimit memlock=-1 \
  -v ${HOME}/models:/root/models \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3

# Install the Triton CLI
pip install git+https://github.com/triton-inference-server/triton_cli.git@0.1.2

# Authenticate with huggingface for restricted models like Llama-2 and Llama-3
huggingface-cli login

# Generate a Triton model repository containing a vLLM model config
triton remove -m all
triton import -m llama-3.1-8b-instruct --backend vllm

# Start Triton pointing at the default model repository
triton start

# Interact with model
triton infer -m llama-3.1-8b-instruct --prompt "machine learning is"

# Profile model with GenAI-Perf
triton profile -m llama-3.1-8b-instruct --backend vllm
```

### Serving a TRT-LLM Model

> [!NOTE]
> By default, TRT-LLM engines are generated in `/tmp/engines/{model_name}`,
> such as `/tmp/engines/gpt2`. They are intentionally kept outside of the model
> repository to improve re-usability across examples and repositories. This
> default location is subject to change, but can be customized with the
> `ENGINE_DEST_PATH` environment variable.
>
> The model configurations generated by the CLI prioritize accessibility over
> performance. As such, the default number of model instances for each model
> will be set to 1. This value can be manually tuned post-generation by
> modifying the `instance_group` field in each model's corresponding
> `config.pbtxt` file. Increasing the instance counts may result in improved
> performance, especially for large batch sizes. For more information, please
> see [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#instance-groups).

The following models are currently supported for automating TRT-LLM
engine builds through the CLI: https://nvidia.github.io/TensorRT-LLM/llm-api-examples/index.html#supported-models

> [!NOTE]
> 1. Building a TRT-LLM engine for Llama-2-7B, Llama-3-8B, or Llama-3.1-8B
>    models may require system RAM of at least 48GB of RAM.
>
> 2. Llama 3.1 may require `pip install transformers>=4.43.1`


#### Example

```bash
# NOTE: Mounting /tmp is optional, but will allow the saving and re-use of
# TRT-LLM engines across different containers. This assumes the value of
# `ENGINE_DEST_PATH` has not been modified.

# This container comes with all of the dependencies for building TRT-LLM engines
# and serving the engine with Triton Inference Server.
docker run -ti \
  --gpus all \
  --network=host \
  --shm-size=1g --ulimit memlock=-1 \
  -v /tmp:/tmp \
  -v ${HOME}/models:/root/models \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# Install the Triton CLI
pip install git+https://github.com/triton-inference-server/triton_cli.git@0.1.2

# Authenticate with huggingface for restricted models like Llama-2 and Llama-3
huggingface-cli login

# Build TRT LLM engine and generate a Triton model repository pointing at it
triton remove -m all
triton import -m llama-3.1-8b-instruct --backend tensorrtllm

# Start Triton pointing at the default model repository
triton start

# Interact with model
triton infer -m llama-3.1-8b-instruct --prompt "machine learning is"

# Profile model with GenAI-Perf
triton profile -m llama-3.1-8b-instruct --backend tensorrtllm
```
## Serving a LLM model with OpenAI API

Triton CLI could also start the triton server with a OpenAI RESTful API Frontend.
Triton Server's OpenAI Frontend supports the following API endpoints:

- [POST /v1/chat/completions](https://platform.openai.com/docs/api-reference/chat/create)
- [POST /v1/completions](https://platform.openai.com/docs/api-reference/completions/create)
- [GET /v1/models](https://platform.openai.com/docs/api-reference/models/list)
- [GET /v1/models/{model_name}](https://platform.openai.com/docs/api-reference/models/retrieve)
- GET /metrics

To start the triton server with a OpenAI RESTful API Frontend, attach the `--frontend openai` to the  `triton start` command.
```bash
triton start --frontend openai
```
By default, the server and its OpenAI api can be accessed at `http://localhost:9000`.

> [!NOTE]
> There could be more than one LLM models in the model repository, each model could have its own tokenizer_config.json.
> OpenAI's `/v1/chat/completions` API requires a chat template from a tokenizer. By default, Triton CLI will
> automatically search for a tokenizer for the chat template in the model repository. If you'd like to set
> a tokenizer's chat template, specify the tokenzier with `--openai-chat-template-tokenizer {higgingface id or path to the tokenizer directory}`
>
> ex:  `triton start --frontend openai --openai-chat-template-tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct`

#### Example

```bash
docker run -ti \
  --gpus all \
  --network=host \
  --shm-size=1g --ulimit memlock=-1 \
  -v /tmp:/tmp \
  -v ${HOME}/models:/root/models \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# Install the Triton CLI
pip install git+https://github.com/triton-inference-server/triton_cli.git@main

# Authenticate with huggingface for restricted models like Llama-2 and Llama-3
huggingface-cli login

# Build TRT LLM engine and generate a Triton model repository pointing at it
triton remove -m all
triton import -m llama-3.1-8b-instruct --backend tensorrtllm
# For vllm backend:
# triton import -m llama-3.1-8b-instruct --backend vllm

# Start Triton with a OpenAI RESTful API Frontend
triton start --frontend openai

# Interact with model at http://localhost:9000
curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "llama-3.1-8b-instruct",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}'

# Profile model with GenAI-Perf
triton profile -m llama-3.1-8b-instruct --service-kind openai --endpoint-type chat --url localhost:9000 --streaming
```

## Additional Dependencies for Custom Environments

When using Triton CLI outside of official Triton NGC containers, you may
encounter the following issues, indicating additional dependencies need
to be installed.

1. If you encounter an error related to `libb64.so` from `triton profile`
or `perf_analyzer` such as:
```
perf_analyzer: error while loading shared libraries: libb64.so.0d
```

Then you likely need to install this system dependency:
```
apt install libb64-dev
```

2. If you encounter an error related to `libcudart.so` from `triton profile`
or `perf_analyzer` such as:
```
perf_analyzer: error while loading shared libraries: libcudart.so
```

Then you likely need to install the CUDA toolkit or set your `LD_LIBRARY_PATH`
correctly. Refer to: https://developer.nvidia.com/cuda-downloads.

3. To build TensorRT LLM engines, you will need MPI installed in your environment.
MPI should be shipped in any relevant Triton or TRT-LLM containers, but if
building engines on host you can install them like so:
```
sudo apt install libopenmpi-dev
```

## Known Limitations
- Models and configurations generated by Triton CLI are focused on ease-of-use,
and may not be as optimized as possible for your system or use case.
- Triton CLI currently uses the TRT-LLM dependencies installed in its environment
to build TRT-LLM engines, so you must take care to match the build-time and
run-time versions of TRT-LLM.
