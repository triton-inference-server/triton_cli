# TRT-LLM image contains engine building and runtime dependencies
FROM nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    git clone -b r24.08 https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend && \
    cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm && \
    rm -r /tmp/vllm_backend

# vLLM runtime dependencies
RUN pip install "vllm==0.5.3.post1" "setuptools==74.0.0"
