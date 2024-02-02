FROM nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3

# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py

# TRT-LLM engine build dependencies
# NOTE: torch 2.2.0 has a symbol conflict, so WAR is to install 2.1.2
RUN pip install \
  "psutil" \
  "pynvml>=11.5.0" \
  "torch==2.1.2" \
  --extra-index-url https://pypi.nvidia.com/ "tensorrt-llm==0.7.1"

# vLLM runtime dependencies
RUN pip install \
  vllm==0.3.0 \
  transformers>=4.37.0 \
  accelerate==0.26.1

# TODO: Install Triton CLI in this image
