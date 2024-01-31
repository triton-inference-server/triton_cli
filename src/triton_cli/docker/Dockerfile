FROM nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3

# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py

# TRT-LLM engine build dependencies
RUN pip install \
  --extra-index-url https://pypi.nvidia.com/ "tensorrt-llm==0.7.1"

# vLLM runtime dependencies
RUN pip install \
  vllm==0.2.3 \
  transformers==4.35.2 \
  accelerate==0.26.1

# TODO: Install Triton CLI in this image
