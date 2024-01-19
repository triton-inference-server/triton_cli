FROM nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3

# vLLM runtime dependencies
RUN pip install "vllm==0.2.3"
# Setup vLLM Triton backend
RUN mkdir -p /opt/tritonserver/backends/vllm && \
    wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py

# TRT-LLM engine build dependencies
RUN pip install --extra-index-url https://pypi.nvidia.com/ "tensorrt-llm==0.7.0"

# Mistral support
RUN pip install "transformers>=4.34.1"
