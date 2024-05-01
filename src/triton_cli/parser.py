#!/usr/bin/env python3
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import subprocess
import sys
import time
from typing import List
import logging
import argparse
from pathlib import Path

from rich import print as rich_print
from rich.progress import Progress

from triton_cli.constants import (
    DEFAULT_MODEL_REPO,
    DEFAULT_TRITONSERVER_IMAGE,
    LOGGER_NAME,
    PASSTHROUGH_SUBCOMMANDS,
)
from triton_cli.client.client import InferenceServerException, TritonClient
from triton_cli.metrics import MetricsClient
from triton_cli.repository import ModelRepository
from triton_cli.server.server_factory import TritonServerFactory

logger = logging.getLogger(LOGGER_NAME)

# TODO: Move to config file approach?
# TODO: Per-GPU mappings for TRT LLM models
# TODO: Ordered list of supported backends for models with multi-backend support
KNOWN_MODEL_SOURCES = {
    # Require authentication
    "llama-2-7b": "hf:meta-llama/Llama-2-7b-hf",
    "llama-2-7b-chat": "hf:meta-llama/Llama-2-7b-chat-hf",
    "llama-3-8b": "hf:meta-llama/Meta-Llama-3-8B",
    "llama-3-8b-instruct": "hf:meta-llama/Meta-Llama-3-8B-Instruct",
    # Public
    "gpt2": "hf:gpt2",
    "opt125m": "hf:facebook/opt-125m",
    "mistral-7b": "hf:mistralai/Mistral-7B-v0.1",
    "falcon-7b": "hf:tiiuae/falcon-7b",
}


def check_known_sources(model: str):
    if model in KNOWN_MODEL_SOURCES:
        source = KNOWN_MODEL_SOURCES[model]
        logger.info(f"Known model source found for '{model}': '{source}'")
    else:
        logger.error(
            f"No known source for model: '{model}'. Known sources: {list(KNOWN_MODEL_SOURCES.keys())}"
        )
        raise Exception("Please use a known model, or provide a --source.")

    return source


# TODO: Move out of parser
# TODO: Show server log/progress until ready
# NOTE: This function is not currently used. Keeping for potential use when
# launching the server in the background.
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
        raise TimeoutError(
            f"Timed out waiting {timeout} seconds for server to startup. Try increasing --server-timeout."
        )


# ================================================
# ARG GROUPS
# ================================================
def add_verbose_args(subcommands):
    for subcommand in subcommands:
        subcommand.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            default=False,
            help="Enable verbose logging",
        )


def add_backend_args(subcommands):
    for subcommand in subcommands:
        subcommand.add_argument(
            "--backend",
            type=str,
            required=False,
            help="Backend type of model. Will be inferred by default.",
        )


def add_server_start_args(subcommands):
    for subcommand in subcommands:
        subcommand.add_argument(
            "--mode",
            choices=["local", "docker"],
            type=str,
            default=None,
            required=False,
            help="Mode to start Triton with. If a mode is explicitly specified, only that mode will be tried. If no mode is specified (default), 'local' mode is tried first, then falls back to 'docker' mode on failure.",
        )
        # TODO: Should probably not use the custom image by default, it's more for developer convenience
        subcommand.add_argument(
            "--image",
            type=str,
            required=False,
            default=None,
            help=f"Image to use when starting Triton with 'docker' mode. Default is a custom image tagged '{DEFAULT_TRITONSERVER_IMAGE}'.",
        )
        subcommand.add_argument(
            "--server-timeout",
            type=int,
            required=False,
            default=300,
            help="Maximum number of seconds to wait for server startup. (Default: 300)",
        )


def add_model_args(subcommands):
    for subcommand in subcommands:
        subcommand.add_argument(
            "-m", "--model", type=str, required=True, help="Model name"
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


# ================================================
# REPO
# ================================================
def parse_args_repo(parser):
    repo_import = parser.add_parser("import", help="Import model to model repository")
    repo_import.set_defaults(func=handle_repo_import)
    repo_import.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name to assign to model in repository",
    )
    repo_import.add_argument(
        "-s",
        "--source",
        type=str,
        required=False,
        help="Local model path or model identifier. Use prefix 'hf:' to specify a HuggingFace model ID. "
        "NOTE: HuggingFace model support is currently limited to Transformer models through the vLLM backend.",
    )

    repo_remove = parser.add_parser("remove", help="Remove model from model repository")
    repo_remove.set_defaults(func=handle_repo_remove)
    repo_remove.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name of model to remove from repository. Specify 'all' to remove all models in the model repository.",
    )

    repo_list = parser.add_parser("list", help="List models in the model repository")
    repo_list.set_defaults(func=handle_repo_list)

    add_backend_args([repo_import])
    add_repo_args([repo_import, repo_remove, repo_list])
    return parser


def handle_repo_import(args: argparse.Namespace):
    repo = ModelRepository(args.model_repository)
    # Handle common models for convenience
    if not args.source:
        args.source = check_known_sources(args.model)

    repo.add(
        args.model,
        version=1,
        source=args.source,
        backend=args.backend,
    )


def handle_repo_remove(args: argparse.Namespace):
    repo = ModelRepository(args.model_repository)
    repo.remove(args.model)


def handle_repo_list(args: argparse.Namespace):
    repo = ModelRepository(args.model_repository)
    repo.list()


# ================================================
# SERVER
# ================================================
def parse_args_server(parser):
    server_start = parser.add_parser("start", help="Start a Triton server")
    server_start.set_defaults(func=handle_server_start)
    add_server_start_args([server_start])
    add_repo_args([server_start])

    # TODO:
    #   - triton stop
    #   - triton status


def handle_server_start(args: argparse.Namespace):
    start_server_with_fallback(args, blocking=True)


# TODO: Move to utils <-- Delete?
def start_server_with_fallback(args: argparse.Namespace, blocking=True):
    modes = [args.mode]
    if not args.mode:
        modes = ["local", "docker"]
        logger.debug(f"No --mode specified, trying the following modes: {modes}")

    server = None
    errors = []
    for mode in modes:
        try:
            args.mode = mode
            server = start_server(args, blocking=blocking)
        except Exception as e:
            msg = f"Failed to start server in '{mode}' mode. {e}"
            logger.debug(msg)
            errors.append(msg)
            continue

    if not server:
        # Give nicely formatted errors for each case.
        if len(errors) > 1:
            raise Exception(f"Failed to start server. Errors: {errors}")
        elif len(errors) == 1:
            raise Exception(f"{errors[0]}")
        else:
            raise Exception("Failed to start server, unknown error.")

    return server


def start_server(args: argparse.Namespace, blocking=True):
    assert args.mode is not None
    server = TritonServerFactory.get_server_handle(args)
    server.start()

    if blocking:
        try:
            logger.info("Reading server output...")
            server.logs()
        except KeyboardInterrupt:
            print()

        logger.info("Stopping server...")
        server.stop()

    return server


# ================================================
# INFERENCE
# ================================================
def parse_args_inference(parser):
    infer = parser.add_parser("infer", help="Send inference requests to models")
    infer.set_defaults(func=handle_infer)
    add_model_args([infer])

    infer.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text input to LLM-like models. Required for inference on LLMs, optional otherwise.",
    )
    add_client_args([infer])


def handle_infer(args: argparse.Namespace):
    client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)
    client.infer(model=args.model, prompt=args.prompt)


# ================================================
# Profile
# ================================================
def parse_args_profile(parser):
    profile = parser.add_parser("profile", help="Profile models", add_help=False)
    profile.set_defaults(func=handle_profile)
    profile.add_argument(
        "--help", action="store_true", help="Show help message and exit"
    )
    profile.add_argument(
        "--task",
        type=str,
        required=False,
        help="Specify the task type to select the profiling tool (e.g., llm for LLM models)",
    )


def handle_profile(args: argparse.Namespace):
    if args.task == "llm":
        cmd = build_command("genai-perf", args)
    elif args.task is None:
        cmd = build_command("perf_analyzer", args)
    else:
        raise ValueError(
            "Unsupported task type. Only 'llm' or unspecified tasks are supported for profiling."
        )
    logger.info(f"Running: '{' '.join(cmd)}'")
    subprocess.run(cmd, check=True)


# ================================================
# Optimize
# ================================================
def parse_args_optimize(parser):
    optimize = parser.add_parser(
        "optimize", help="Optimize models using Model Analyzer", add_help=False
    )
    optimize.set_defaults(func=handle_optimize)
    optimize.add_argument(
        "--help", action="store_true", help="Show help message and exit"
    )


def handle_optimize(args: argparse.Namespace):
    args.__delattr__("func")
    cmd = build_command("model-analyzer", args)
    logger.info(f"Running: '{' '.join(cmd)}'")
    subprocess.run(cmd, check=True)


# ================================================
# Util
# ================================================
def parse_args_utils(parser):
    metrics = parser.add_parser("metrics", help="Get metrics for model")
    metrics.set_defaults(func=handle_metrics)
    config = parser.add_parser("config", help="Get config for model")
    config.set_defaults(func=handle_config)
    status = parser.add_parser("status", help="Get status of running Triton server")
    status.set_defaults(func=handle_status)

    add_model_args([config])
    # TODO: Refactor later - No grpc support for metrics endpoint
    add_client_args([config, metrics, status])

    # TODO:
    #   - triton load
    #   - triton unload


def handle_metrics(args: argparse.Namespace):
    client = MetricsClient(args.url, args.port)
    # NOTE: Consider pretty table in future, but JSON output seems more
    #       functionally useful for now.
    # client.display_table()
    client.display_json()


def handle_config(args: argparse.Namespace):
    client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)
    config = client.get_model_config(args.model)
    if config:
        # TODO: Table
        rich_print(config)


def handle_status(args: argparse.Namespace):
    client = TritonClient(url=args.url, port=args.port, protocol=args.protocol)

    # FIXME: Does this need its own subcommand? e.g., triton metadata
    # metadata = client.get_server_metadata()
    # if metadata:
    #     print(json.dumps(metadata))

    health = client.get_server_health()
    if health:
        print(json.dumps(health))


# Optional argv used for testing - will default to sys.argv if None.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="triton", description="CLI to interact with Triton Inference Server"
    )
    subcommands = parser.add_subparsers(required=True)
    parse_args_repo(subcommands)
    parse_args_server(subcommands)
    parse_args_inference(subcommands)
    parse_args_optimize(subcommands)
    parse_args_profile(subcommands)
    parse_args_utils(subcommands)
    add_verbose_args([parser])

    argv_ = argv if argv is not None else sys.argv[1:]
    # If a passthrough command is passed as the first arg,
    # special handling is needed.
    need_special_handling = len(argv_) > 1 and argv_[0] in PASSTHROUGH_SUBCOMMANDS
    if need_special_handling:
        updated_args, pruned_args = prune_extra_subcommand_args(
            argv_, subcommands.choices.keys()
        )
        args, unknown_args = parser.parse_known_args(updated_args)
        args = add_unknown_args_to_args(args, pruned_args, unknown_args)
    else:
        args = parser.parse_args(argv)
    return args


# ================================================
# Helper functions
# ================================================
def build_command(executable: str, args: argparse.Namespace):
    skip_args = ["func", "task"]
    cmd = [executable]
    program_subcommand = None
    for arg, value in vars(args).items():
        if arg in skip_args:
            pass
        elif arg in PASSTHROUGH_SUBCOMMANDS:
            program_subcommand = arg
        elif value is False:
            pass
        elif value is True:
            if len(arg) == 1:
                cmd += [f"-{arg}"]
            else:
                cmd += [f"--{arg}"]
        else:
            if len(arg) == 1:
                cmd += [f"-{arg}", f"{value}"]
            else:
                arg = arg
                cmd += [f"--{arg}", f"{value}"]
    if program_subcommand:
        cmd = [executable, program_subcommand] + cmd[1:]
    return cmd


def prune_extra_subcommand_args(argv: List[str], subcommand_names):
    """Triton CLI can call other programs with passthrough args, so sometimes the first argument is a subcommand."""
    """If so, prune it."""
    pruned_args = []
    if len(argv) > 1 and argv[1] in subcommand_names:
        pruned_args.append(argv[1])
        argv.remove(argv[1])
    return argv, pruned_args


def add_unknown_args_to_args(
    args: argparse.Namespace, pruned_args: List[str], unknown_args: List[str]
):
    """Add unknown and pruned args to args list"""
    unknown_args_dict = turn_unknown_args_into_dict(unknown_args)
    if pruned_args:
        setattr(args, pruned_args[0], True)
    for key, value in unknown_args_dict.items():
        setattr(args, key, value)
    return args


def turn_unknown_args_into_dict(unknown_args: List[str]):
    """Convert list of unknown args to dictionary"""
    it = iter(unknown_args)
    unknown_args_dict = {}
    try:
        while True:
            arg = next(it)
            if arg.startswith(("-", "--")):
                key = arg.lstrip("-")
                # Peek to see if next item is a value or another flag
                next_arg = next(it, None)
                if next_arg and not next_arg.startswith(("-", "--")):
                    unknown_args_dict[key] = next_arg
                else:
                    unknown_args_dict[key] = True
                    if next_arg:
                        it = iter([next_arg] + list(it))
            else:
                raise ValueError(f"Argument does not start with a '-' or '--': {arg}")
    except StopIteration:
        pass
    return unknown_args_dict
