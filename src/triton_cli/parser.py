import argparse
from pathlib import Path
from triton_cli.constants import DEFAULT_MODEL_REPO

import logging

logger = logging.getLogger("triton")


def handle_repo(args: argparse.Namespace):
    from triton_cli.repository import ModelRepository

    repo = ModelRepository(args.model_repository)
    if args.subcommand == "add":
        repo.add(
            args.name,
            version=1,
            model_path=args.model,
            huggingface_id=args.huggingface_id,
            backend=args.backend,
        )
    elif args.subcommand == "rm":
        repo.remove(args.name)
    elif args.subcommand == "list":
        repo.list()
    elif args.subcommand == "clear":
        repo.clear()
    else:
        raise NotImplementedError(
            f"Repo subcommand {args.subcommand} not implemented yet"
        )


def handle_client(args: argparse.Namespace):
    from triton_cli.client.client import TritonClient

    client = TritonClient(args.protocol)
    if args.subcommand == "infer":
        client.infer(args.model, args.data)
    else:
        raise NotImplementedError(
            f"Client subcommand {args.subcommand} not implemented yet"
        )


def handle_server(args: argparse.Namespace):
    from triton_cli.server.server_factory import TritonServerFactory

    # TODO: No support for specifying GPUs for now, default to all available.
    gpus = []
    server = TritonServerFactory.get_server_handle(args, gpus)
    logger.debug(server)
    try:
        logger.info(
            f"Starting server with model repository: [{args.model_repository}]..."
        )
        server.start()
        logger.info("Reading server output...")
        server.logs()
        logger.info("Done")
    except KeyboardInterrupt:
        print()
        pass

    logger.info("Stopping server...")
    server.stop()


def parse_args_client(subcommands):
    # Infer
    client = subcommands.add_parser(
        "client", help="Interact with running server using Client APIs"
    )
    client.set_defaults(func=handle_client)

    client_commands = client.add_subparsers(required=True, dest="subcommand")
    infer = client_commands.add_parser(
        "infer", help="Send inference requests to models"
    )
    infer.add_argument("-m", "--model", type=str, required=True, help="Model name")
    infer.add_argument(
        "--data",
        type=str,
        choices=["random", "scalar"],
        default="random",
        help="Method to provide input data to model",
    )
    infer.add_argument(
        "-i",
        "--protocol",
        type=str,
        default="grpc",
        choices=["http", "grpc"],
        help="Protocol to use for Client APIs",
    )
    return infer


def parse_args_repo(subcommands):
    # Model Repository Management
    repo = subcommands.add_parser(
        "repo", help="Interact with a Triton model repository."
    )
    repo.set_defaults(func=handle_repo)
    repo_commands = repo.add_subparsers(required=True, dest="subcommand")
    repo_add = repo_commands.add_parser("add", help="Add model to model repository")
    repo_add.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name to assign to model in repository",
    )
    repo_add.add_argument(
        "--repo",
        "--model-repository",
        "--model-store",
        dest="model_repository",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    model_group = repo_add.add_mutually_exclusive_group()
    model_group.add_argument(
        "--hf",
        "--huggingface",
        dest="huggingface_id",
        type=str,
        required=False,
        help="HuggingFace model ID to deploy (Currently limited to Transformers models)",
    )
    model_group.add_argument(
        "-m",
        "--model",
        type=Path,
        required=False,
        help="Path to model to add to repository",
    )
    repo_add.add_argument(
        "-b",
        "--backend",
        type=str,
        required=False,
        help="Backend type of model. Will be inferred by default.",
    )
    repo_remove = repo_commands.add_parser(
        "rm", help="Remove model from model repository"
    )
    repo_remove.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of model to remove from repository",
    )
    repo_remove.add_argument(
        "--repo",
        "--model-repository",
        "--model-store",
        dest="model_repository",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    repo_list = repo_commands.add_parser(
        "list", help="List the models in the model repository"
    )
    repo_list.add_argument(
        "--repo",
        "--model-repository",
        "--model-store",
        dest="model_repository",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    repo_clear = repo_commands.add_parser(
        "clear", help="Delete all contents in model repository"
    )
    repo_clear.add_argument(
        "--repo",
        "--model-repository",
        "--model-store",
        dest="model_repository",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    return repo


def parse_args_server(subcommands):
    # Model Repository Management
    server = subcommands.add_parser("server", help="Interact with a Triton server.")
    server.set_defaults(func=handle_server)
    server_commands = server.add_subparsers(required=True, dest="subcommand")
    server_start = server_commands.add_parser("start", help="Start a Triton server")
    server_start.add_argument(
        "--mode",
        choices=["local", "docker"],
        type=str,
        default="docker",
        required=False,
        help="Mode to start Triton with. (Default: 'docker')",
    )
    server_start.add_argument(
        "--repo",
        "--model-repository",
        "--model-store",
        dest="model_repository",
        type=Path,
        required=False,
        default=DEFAULT_MODEL_REPO,
        help="Path to local model repository to use. (Default: '${HOME}/models')",
    )
    server_start.add_argument(
        "--image",
        type=str,
        required=False,
        # or eventually 23.10-py3 for generic image
        # default="nvcr.io/nvidia/tritonserver:23.10-vllm-python-py3",
        default="gitlab-master.nvidia.com:5005/dl/dgx/tritonserver:master.10736241-vllm-amd64",
        help="Image to use when starting Triton with 'docker' mode",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="triton", description="CLI to interact with Triton Inference Server"
    )
    subcommands = parser.add_subparsers(required=True, dest="command")
    _ = parse_args_client(subcommands)
    _ = parse_args_repo(subcommands)
    _ = parse_args_server(subcommands)
    args = parser.parse_args()
    return args
