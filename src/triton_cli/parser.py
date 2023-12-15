import time
import logging
import argparse
from pathlib import Path

from rich import print as rich_print
from rich.progress import Progress

from triton_cli.constants import DEFAULT_MODEL_REPO, LOGGER_NAME
from triton_cli.client.client import InferenceServerException, TritonClient
from triton_cli.metrics import MetricsClient
from triton_cli.repository import ModelRepository
from triton_cli.server.server_factory import TritonServerFactory
from triton_cli.profiler import Profiler

logger = logging.getLogger(LOGGER_NAME)

# TODO: Move to config file approach
# TODO: Per-GPU mappings for TRT LLM models
KNOWN_MODEL_SOURCES = {
    "llama-2-7b": "ngc:whw3rcpsilnj/playground/llama2_7b_trt_a100:0.1",
    "llama-2-13b": "ngc:whw3rcpsilnj/playground/llama2_13b_trt_a100:0.1",
    "gpt2": "hf:gpt2",
    "opt125m": "hf:facebook/opt-125m",
    "mistral-7b": "hf:mistralai/Mistral-7B-v0.1",
}


# TODO: Move out of parser
# TODO: rich progress bar
def wait_for_ready(timeout, server, client):
    with Progress(transient=True) as progress:
        _ = progress.add_task("[green]Loading models...", total=None)
        for _ in range(timeout):
            # Client health will allow early exit of wait if healthy,
            # errors may occur while server starting up, so ignore them.
            try:
                if client.is_server_ready():
                    return
            except InferenceServerException:
                pass

            # Server health will throw exception if error occurs on server side
            server.health()
            time.sleep(1)
        raise Exception("Timed out waiting for server to startup.")


def add_server_start_args(subcommands):
    for subcommand in subcommands:
        subcommand.add_argument(
            "--mode",
            choices=["local", "docker"],
            type=str,
            default="local",
            required=False,
            help="Mode to start Triton with. (Default: 'local')",
        )
        subcommand.add_argument(
            "--image",
            type=str,
            required=False,
            default="nvcr.io/nvidia/tritonserver:23.11-vllm-python-py3",
            help="Image to use when starting Triton with 'docker' mode",
        )
        # TODO: Delete once world-size can be parsed from a known
        # config file location.
        subcommand.add_argument(
            "--world-size",
            type=int,
            required=False,
            default=-1,
            help="Number of devices to deploy a tensorrtllm model.",
        )
        subcommand.add_argument(
            "--server-timeout",
            type=int,
            required=False,
            default=300,
            help="Maximum number of seconds to wait for server startup. (Default: 300)",
        )


def add_client_args(subcommands):
    # Add protocol/url/port to all client-based subcommands
    for subcommand in subcommands:
        subcommand.add_argument(
            "-i",
            "--protocol",
            type=str,
            default="grpc",
            choices=["http", "grpc"],
            help="Protocol to use for communicating with server (Default: grpc)",
        )
        # TODO: "--ip" instead of "--url"?
        subcommand.add_argument(
            "-u",
            "--url",
            type=str,
            required=False,
            default="localhost",
            help="IP of server (Default: localhost)",
        )
        subcommand.add_argument(
            "-p",
            "--port",
            type=int,
            required=False,
            default=None,
            help="Port of server endpoint (Default: 8000 for http, 8001 for grpc, 8002 for metrics)",
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
            help="Path to local model repository to use (Default: ~/models)",
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
        # TODO
        if not args.port:
            args.port = 8001 if args.protocol == "grpc" else 8000

        logger.info(f"Running Perf Analyzer profiler on '{args.model}'...")
        Profiler.profile(
            model=args.model,
            batch_size=args.batch_size,
            url=f"{args.url}:{args.port}",
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
        server = TritonServerFactory.get_server_handle(args)
        try:
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
    # TODO: Remove or consolidate
    profile.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="The batch size / concurrency to benchmark",
    )

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
    add_server_start_args([server_start])
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


def handle_bench(args: argparse.Namespace):
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    ### Add model to repo
    repo = ModelRepository(args.model_repository)
    # To avoid stale models or too many models across runs
    repo.clear()
    # Handle common models for convenience
    if not args.source:
        if args.model in KNOWN_MODEL_SOURCES:
            args.source = KNOWN_MODEL_SOURCES[args.model]
            logger.info(f"Known model source found for '{args.model}': '{args.source}'")
        else:
            logger.error(
                f"No known source for model: '{args.model}'. Known sources: {list(KNOWN_MODEL_SOURCES.keys())}"
            )
            raise Exception("Please use a known model, or provide a --source.")

    repo.add(
        args.model,
        version=1,
        source=args.source,
        backend=None,
        verbose=args.verbose,
    )

    ### Start server
    server = TritonServerFactory.get_server_handle(args)
    try:
        server.start()
        client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)
        wait_for_ready(args.server_timeout, server, client)
        ### Profile model
        logger.info("Server is ready for inference.")
        if not args.port:
            args.port = 8001 if args.protocol == "grpc" else 8000

        logger.info(f"Running Perf Analyzer profiler on '{args.model}'...")
        Profiler.profile(
            model=args.model,
            batch_size=args.batch_size,
            url=f"{args.url}:{args.port}",
            input_length=2048,
        )
    except KeyboardInterrupt:
        print()
    except Exception as ex:
        # Catch timeout exception
        logger.error(ex)

    logger.info("Stopping server...")
    server.stop()


def parse_args_bench(subcommands):
    # Model Repository Management
    bench_run = subcommands.add_parser(
        "bench", help="Run benchmarks on a model loaded into the Triton server."
    )
    bench_run.set_defaults(func=handle_bench)
    # bench_commands = bench.add_subparsers(required=True, dest="subcommand")
    # bench_run = bench_commands.add_parser(
    #    "run", help="Start a Triton benchmarking session."
    # )
    bench_run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    # Conceptually group args for easier visualization
    model_group = bench_run.add_argument_group("model")
    known_models = list(KNOWN_MODEL_SOURCES.keys())
    model_group.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=f"The name of the model to benchmark. Popular models with known sources can be selected directly from this list: {known_models}. Otherwise, provide a --source for unknown models.",
    )
    model_group.add_argument(
        "-s",
        "--source",
        type=str,
        required=False,
        help="Local model path or model identifier. Use prefix 'hf:' to specify a HuggingFace model ID, or 'ngc:' for NGC model ID. "
        "NOTE: HuggingFace models are currently limited to vLLM, and NGC models are currently limited to TRT-LLM",
    )

    model_group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help="The batch size / concurrency to benchmark (Default: 1)",
    )

    server_group = bench_run.add_argument_group("server")
    client_group = bench_run.add_argument_group("client")
    add_server_start_args([server_group])
    add_repo_args([server_group])
    add_client_args([client_group])

    return bench_run


def parse_args():
    parser = argparse.ArgumentParser(
        prog="triton", description="CLI to interact with Triton Inference Server"
    )
    subcommands = parser.add_subparsers(required=True, dest="command")
    _ = parse_args_model(subcommands)
    _ = parse_args_repo(subcommands)
    _ = parse_args_server(subcommands)
    _ = parse_args_bench(subcommands)
    args = parser.parse_args()
    return args
