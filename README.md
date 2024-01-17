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

## Example

```
# Default install location from pip install
rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ which triton
/home/rmccormick/.local/bin/triton

rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton -h
usage: triton [-h] {model,repo,server} ...

CLI to interact with Triton Inference Server

positional arguments:
  {model,repo,server}
    model              Interact with running server using model APIs
    repo               Interact with a Triton model repository.
    server             Interact with a Triton server.

options:
  -h, --help           show this help message and exit

rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton repo -h
usage: triton repo [-h] {add,remove,list,clear} ...

positional arguments:
  {add,remove,list,clear}
    add                 Add model to model repository
    remove              Remove model from model repository
    list                List the models in the model repository
    clear               Delete all contents in model repository

options:
  -h, --help            show this help message and existing

# Just to get a fresh start. Not necessary.
# Can specify --repo to any repo command for custom paths.
rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton repo clear
triton - INFO - Using existing model repository: /home/rmccormick/models
triton - INFO - Clearing all contents from /home/rmccormick/models...

rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton repo list
triton - INFO - Created new model repository: /home/rmccormick/models
triton - INFO - Current repo at /home/rmccormick/models:
models/

# Will be fleshed out for TRT LLM support soon
rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton repo add -m opt125 --source hf:facebook/opt-125m
triton - INFO - Using existing model repository: /home/rmccormick/models
triton - INFO - HuggingFace prefix detected, parsing HuggingFace ID
triton - INFO - Adding new model to repo at: /home/rmccormick/models/opt125/1
triton - INFO - Current repo at /home/rmccormick/models:
models/
└── opt125/
    ├── 1/
    │   └── model.json
    └── config.pbtxt


rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton server start -h
usage: triton server start [-h] [--mode {local,docker}] [--image IMAGE] [--repo MODEL_REPOSITORY]

options:
  -h, --help            show this help message and exit
  --mode {local,docker}
                        Mode to start Triton with. (Default: 'docker')
  --image IMAGE         Image to use when starting Triton with 'docker' mode
  --repo MODEL_REPOSITORY, --model-repository MODEL_REPOSITORY, --model-store MODEL_REPOSITORY
                        Path to local model repository to use (default: ~/models)


# Can specify custom --image
rmccormick@ced35d0-lcedt:~/triton/cli/triton_cli$ triton server start
triton - INFO - Starting a Triton Server using docker
triton - INFO - Pulling docker image nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3
...

```
