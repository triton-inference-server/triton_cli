FROM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3

# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py

# TRT-LLM engine build dependencies
# NOTE: torch 2.2.0 has a symbol conflict, so WAR is to install 2.1.2
RUN pip install \
  "psutil" \
  "pynvml>=11.5.0" \
  --extra-index-url https://pypi.nvidia.com/ "tensorrt-llm==0.9.0"

# vLLM runtime dependencies
RUN pip install \
  # Triton 24.04 vLLM containers comes with "vllm==0.4.0.post1", but this has
  # incompatible dependencies with trtllm==0.9.0 around torch and transformers.
  "vllm==0.4.1"

# TODO: Install Triton CLI in this image
