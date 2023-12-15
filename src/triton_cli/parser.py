import logging
import argparse
from pathlib import Path

from rich import print as rich_print

from triton_cli.constants import DEFAULT_MODEL_REPO
from triton_cli.client.client import TritonClient
from triton_cli.metrics import MetricsClient
from triton_cli.repository import ModelRepository
from triton_cli.server.server_factory import TritonServerFactory
from triton_cli.profiler import Profiler

logger = logging.getLogger("triton")


def add_client_args(subcommands):
    # Add protocol/url/port to all client-based subcommands
    for subcommand in subcommands:
        subcommand.add_argument(
            "-i",
            "--protocol",
            type=str,
            default="grpc",
            choices=["http", "grpc"],
            help="Protocol to use for communicating with server (default: grpc)",
        )
        # TODO: "--ip" instead of "--url"?
        subcommand.add_argument(
            "-u",
            "--url",
            type=str,
            required=False,
            default="localhost",
            help="IP of server (default: localhost)",
        )
        subcommand.add_argument(
            "-p",
            "--port",
            type=int,
            required=False,
            default=None,
            help="Port of server endpoint (default: 8000 for http, 8001 for grpc, 8002 for metrics)",
        )


def add_repo_args(subcommands):
    # All repo subcommands can specify model repository
    for subcommand in subcommands:
        subcommand.add_argument(
            "--repo",
            "--model-repository",
            "--model-store",
            dest="model_repository",
            type=Path,
            required=False,
            default=DEFAULT_MODEL_REPO,
            help="Path to local model repository to use (default: ~/models)",
        )


def handle_repo(args: argparse.Namespace):
    repo = ModelRepository(args.model_repository)
    if args.subcommand == "add":
        repo.add(
            args.model,
            version=1,
            source=args.source,
            backend=args.backend,
        )
    elif args.subcommand == "remove":
        repo.remove(args.model)
    elif args.subcommand == "list":
        repo.list()
    elif args.subcommand == "clear":
        repo.clear()
    else:
        raise NotImplementedError(
            f"Repo subcommand {args.subcommand} not implemented yet"
        )


def handle_model(args: argparse.Namespace):
    client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)

    if args.subcommand == "infer":
        client.infer(args.model, args.data, args.prompt)
    elif args.subcommand == "profile":
        # TODO: run PA LLM benchmark script
        print("pull engine()")
        print("run_server()")
        print("profile()")
        Profiler.profile(
            model=args.model,
            batch_size=64,
            url="localhost:8001",
            input_length=2048,
        )
    elif args.subcommand == "config":
        config = client.get_model_config(args.model)
        if config:
            logger.info(f"{args.subcommand}:")
            # TODO: Table
            rich_print(config)
    # TODO: Consider top-level metrics command/handler instead
    elif args.subcommand == "metrics":
        client = MetricsClient(args.url, args.port)
        # For model subcommand, limit metrics to only specified model metrics
        client.display_table(model_name=args.model)
    else:
        raise Exception(f"model subcommand {args.subcommand} not supported")


def handle_server(args: argparse.Namespace):
    if args.subcommand == "start":
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
        except KeyboardInterrupt:
            print()

        logger.info("Stopping server...")
        server.stop()
    # TODO: Consider top-level metrics command/handler
    elif args.subcommand == "metrics":
        client = MetricsClient(args.url, args.port)
        client.display_table()
    elif args.subcommand == "health":
        client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)
        health = client.get_server_health()
        if health:
            logger.info(f"{args.subcommand}:\n{health}")
    elif args.subcommand == "metadata":
        client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)
        metadata = client.get_server_metadata()
        if metadata:
            logger.info(f"{args.subcommand}:")
            # TODO: Table
            rich_print(metadata)
    else:
        raise NotImplementedError(f"server subcommand {args.subcommand} not supported")


def parse_args_model(subcommands):
    # Infer
    model = subcommands.add_parser(
        "model", help="Interact with running server using model APIs"
    )
    model.set_defaults(func=handle_model)

    model_commands = model.add_subparsers(required=True, dest="subcommand")
    infer = model_commands.add_parser("infer", help="Send inference requests to models")
    infer.add_argument("-m", "--model", type=str, required=True, help="Model name")
    infer.add_argument(
        "--data",
        type=str,
        choices=["random", "scalar"],
        default="random",
        help="Method to provide input data to model",
    )
    infer.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text input to LLM-like models. Required for inference on LLMs, optional otherwise.",
    )
    profile = model_commands.add_parser("profile", help="Run Perf Analyzer")
    profile.add_argument("-m", "--model", type=str, required=True, help="Model name")
    config = model_commands.add_parser("config", help="Get config for model")
    config.add_argument("-m", "--model", type=str, required=True, help="Model name")
    metrics = model_commands.add_parser("metrics", help="Get metrics for model")
    metrics.add_argument("-m", "--model", type=str, required=True, help="Model name")
    add_client_args([infer, profile, config, metrics])
    return model


def parse_args_repo(subcommands):
    # Model Repository Management
    repo = subcommands.add_parser(
        "repo", help="Interact with a Triton model repository."
    )
    repo.set_defaults(func=handle_repo)
    repo_commands = repo.add_subparsers(required=True, dest="subcommand")
    repo_add = repo_commands.add_parser("add", help="Add model to model repository")
    repo_add.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name to assign to model in repository",
    )
    repo_add.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Local model path or model identifier. Use prefix 'hf:' to specify a HuggingFace model ID. "
        "NOTE: HuggingFace model support is currently limited to Transformer models through the vLLM backend.",
    )
    repo_add.add_argument(
        "-b",
        "--backend",
        type=str,
        required=False,
        help="Backend type of model. Will be inferred by default.",
    )

    repo_remove = repo_commands.add_parser(
        "remove", help="Remove model from model repository"
    )
    repo_remove.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name of model to remove from repository",
    )

    repo_list = repo_commands.add_parser(
        "list", help="List the models in the model repository"
    )
    repo_clear = repo_commands.add_parser(
        "clear", help="Delete all contents in model repository"
    )

    add_repo_args([repo_add, repo_remove, repo_list, repo_clear])
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
        "--image",
        type=str,
        required=False,
        default="nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3",
        help="Image to use when starting Triton with 'docker' mode",
    )
    server_start.add_argument(
        "--world-size",
        type=int,
        required=False,
        default=-1,
        help="Number of devices to deploy a tensorrtllm model.",
    )
    add_repo_args([server_start])

    server_metrics = server_commands.add_parser(
        "metrics", help="Get metrics from running Triton server"
    )
    server_health = server_commands.add_parser(
        "health", help="Get health of running Triton server"
    )
    server_metadata = server_commands.add_parser(
        "metadata", help="Get metadata of running Triton server"
    )
    add_client_args([server_metrics, server_health, server_metadata])
    return server


def parse_args():
    parser = argparse.ArgumentParser(
        prog="triton", description="CLI to interact with Triton Inference Server"
    )
    subcommands = parser.add_subparsers(required=True, dest="command")
    _ = parse_args_model(subcommands)
    _ = parse_args_repo(subcommands)
    _ = parse_args_server(subcommands)
    args = parser.parse_args()
    return args
