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

| [Quickstart](#quickstart) | [Installation](#installation) | [Serving LLMs](#serving-llms) | [Additional Dependencies for Custom Environments](#additional-dependencies-for-custom-environments) | [Known Limitations](#known-limitations) |

## Quickstart
The instructions below outline the process of building, deploying, and profiling
a simple `gpt2` model using Triton CLI.

### Launch an NGC Docker Image
For this example, we will be running a TRT-LLM model, so we will use the
corresponding Triton + TRT-LLM NGC image to ensure we are operating in an
environment with all the necessary system and runtime dependencies.
Start a TRT-LLM container by running the following command:

```bash
# NOTE: Mounting the huggingface cache and /tmp directories are both optional,
# but will allow model saving and re-use across different runs and containers.

docker run -it \
    --name triton \
    --gpus all --network host \
    --shm-size=1g --ulimit memlock=-1 \
    -v /tmp:/tmp \
    -v ${PWD}:/workspace \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3
```

### Install TRT-LLM Build Dependencies
Once the container has launched, we will install all TRT-LLM build dependencies
with the following command:

```bash
pip install "tensorrt_llm==0.8.0" --extra-index-url https://pypi.nvidia.com/
```

### Clone and Install Triton CLI
Next, we can clone the Triton CLI repo and install it within the container:
```bash
git clone https://github.com/triton-inference-server/triton_cli.git
cd triton_cli
pip install .
```
### Import a Model
With Triton CLI and TRT-LLM dependencies installed, we are now ready to
import gpt2. Running the following command will automatically build a TRT-LLM
gpt2 engine and generate a corresponding model repository:
```bash
triton import -m gpt2 --backend tensorrtllm
```

### Serve a Model
We are now ready to serve our model. Start a Triton Inference Server, hosting
our gpt2 model, with the following command:
```bash
triton start
```
The server has launched successfully when you see the following outputs in
your console:
```bash
I0922 23:28:40.351809 1 grpc_server.cc:2451] Started GRPCInferenceService at 0.0.0.0:8001
I0922 23:28:40.352017 1 http_server.cc:3558] Started HTTPService at 0.0.0.0:8000
I0922 23:28:40.395611 1 http_server.cc:187] Started Metrics Service at 0.0.0.0:8002
```

### Send an Inference Request
With the server now live, we are ready to send an inference request to our gpt2
model. To do so, start a shell in the running docker container and use Triton
CLI to send the gpt2 model an inference request:
```bash
# Exec into running Triton container with CLI installed
docker exec -ti triton bash

#Send inference request
triton infer -m gpt2 --prompt "Alan Turing is the father"
```

### Profile a Model
Triton CLI is also capable of measuring and characterizing the performance
behaviors of LLMs. We can profile our gpt2 model by executing the following
command:
```bash
triton profile -m gpt2 --backend tensorrtllm

# Example Output
[ BENCHMARK SUMMARY ]
 * Avg first token latency: 14.1335 ms
 * Avg end-to-end latency: 213.4111 ms
 * Avg end-to-end throughput: 1199.8399 tokens/s
 * Avg generation throughput: 642.3200 output tokens/s
```

## Installation

Currently, Triton CLI can only be installed from source, with plans to host a
pip wheel soon. When installing Triton CLI, please be aware of the versioning
matrix below:

| Triton CLI Version | TRT-LLM Version | Triton Container Tag |
|:------------------:|:---------------:|:--------------------:|
| 0.0.6 | v0.8.0 | 24.02 |
| 0.0.5 | v0.7.1 | 24.01 |

### Install from Source

```bash
# Clone repo, git mode for development/contribution
git clone git@github.com:triton-inference-server/triton_cli.git
cd triton_cli

# Should be pointing at directory containing pyproject.toml
pip install .
```
## Serving LLMs

Triton CLI currently supports serving the following TRT-LLM and vLLM models.

> [!NOTE]
> Usage of `llama-2-7b` requires authentication in Huggingface through either
> `huggingface-login` or setting the `HF_TOKEN` environment variable.

### Officially Supported TRT-LLM Models

The following models have currently been tested for TRT-LLM through the CLI:
> [!NOTE]
> Building a TRT-LLM engine for `llama-2-7b` will require a system
> with at least 64GB of RAM.
- `gpt2`
- `llama-2-7b`
-  `opt125m`

#### TRT-LLM Specific Information
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

### Officially Supported vLLM Models

The following models have currently been tested for vLLM through the CLI:
- `gpt2`
- `llama-2-7b`
- `opt125m`
- `mistral-7b`
- `falcon-7b`

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
- Triton CLI's `profile` command currently only supports TRT-LLM and vLLM models.
- Models and configurations generated by Triton CLI are focused on ease-of-use,
and may not be as optimized as possible for your system or use case.
- Triton CLI currently uses the TRT-LLM dependencies installed in its environment
to build TRT-LLM engines, so you must take care to match the build-time and
run-time versions of TRT-LLM.
- Triton CLI currently does not support launching the server as a background
process.
