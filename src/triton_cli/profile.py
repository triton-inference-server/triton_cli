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

import argparse
from typing import List


# ================================================
# Helper functions
# ================================================
def build_command(args: argparse.Namespace, executable: str):
    skip_args = ["func"]
    cmd = [executable]
    for arg, value in vars(args).items():
        if arg in skip_args:
            pass
        elif value is False:
            pass
        elif value is True:
            if len(arg) == 1:
                cmd += [f"-{arg}"]
            else:
                cmd += [f"--{arg}"]
        # [DLIS-6656] - Remove backend renaming.
        # This allows "tensorrtllm" to be used as the backend for consistency.
        # Once GenAI-Perf releases 24.05, "tensorrtllm" as the backend value
        # will be supported by default.
        elif arg == "backend" and value in ["tensorrtllm", "trtllm"]:
            cmd += ["--backend", "trtllm"]
        else:
            if len(arg) == 1:
                cmd += [f"-{arg}", f"{value}"]
            else:
                cmd += [f"--{arg}", f"{value}"]
    return cmd


def add_unknown_args_to_args(args: argparse.Namespace, unknown_args: List[str]):
    """Add unknown args to args list"""
    unknown_args_dict = turn_unknown_args_into_dict(unknown_args)
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
