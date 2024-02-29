# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
import psutil
import io
import json
from contextlib import redirect_stdout
from triton_cli.main import run
from subprocess import Popen
from triton_cli.client.client import InferenceServerException


def run_server(repo=None):
    args = ["triton", "server", "start"]
    if repo:
        args += ["--repo", repo]
    # Use Popen to run the server in the background as a separate process.
    p = Popen(args)
    return p.pid


def wait_for_server_ready(timeout: int = 120):
    start = time.time()
    while time.time() - start < timeout:
        print("Waiting for server to be ready...", flush=True)
        time.sleep(1)
        try:
            if check_server_ready():
                return
        except InferenceServerException:
            pass
    raise Exception(f"=== Timeout {timeout} secs. Server not ready. ===")


def kill_server(pid: int, sig: int = 2):
    try:
        proc = psutil.Process(pid)
        proc.send_signal(sig)
        proc.wait()
    except psutil.NoSuchProcess as e:
        print(e)


def check_server_ready():
    args = ["server", "health"]
    output = ""
    # Redirect stdout to a buffer to capture the output of the command.
    with io.StringIO() as buf, redirect_stdout(buf):
        run(args)
        output = buf.getvalue()
    output = json.loads(output)
    return output["ready"]
