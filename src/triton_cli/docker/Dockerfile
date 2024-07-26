# TRT-LLM image contains engine building and runtime dependencies
FROM nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3

# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/r24.06/src/model.py

# vLLM runtime dependencies
RUN pip install "vllm==0.4.3"
