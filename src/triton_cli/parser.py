import argparse
from pathlib import Path
from repository import ModelRepository
from client import TritonClient


def handle_repo(args):
    repo = ModelRepository(args.repo)
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
        raise NotImplementedError


def handle_infer(args):
    print("handle_infer")
    _ = TritonClient()
    raise NotImplementedError


def parse_args_infer(subcommands: argparse.ArgumentParser):
    # Infer
    infer = subcommands.add_parser("infer", help="Send inference requests to models")
    infer.set_defaults(func=handle_infer)
    infer.add_argument("-m", "--model", type=str, required=True, help="Model name")
    # TODO: Think more about this
    infer_data_type = infer.add_mutually_exclusive_group(required=False)
    infer_data_type.add_argument(
        "--random", action="store_true", help="Generate random data"
    )
    infer_data_type.add_argument(
        "--scalar",
        type=int,
        default=0,
        help="Use scalar data of provided value, default 0",
    )
    infer_data_type.add_argument(
        "--file", type=str, default="", help="Load data from file"
    )
    return infer


def parse_args_repo(subcommands: argparse.ArgumentParser):
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
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    repo_list = repo_commands.add_parser("list", help="Add model to model repository")
    repo_list.add_argument(
        "--repo",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    repo_clear = repo_commands.add_parser(
        "clear", help="Delete all contents in model repository"
    )
    repo_clear.add_argument(
        "--repo",
        type=Path,
        required=False,
        help="Path to local model repository to use. Will use ${HOME}/models by default.",
    )
    return repo


def parse_args():
    parser = argparse.ArgumentParser(
        prog="triton", description="CLI to interact with Triton Inference Server"
    )
    subcommands = parser.add_subparsers(required=True, dest="command")
    _ = parse_args_infer(subcommands)
    _ = parse_args_repo(subcommands)
    args = parser.parse_args()
    return args
