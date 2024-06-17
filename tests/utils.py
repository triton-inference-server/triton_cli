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


class TritonCommands:
    def _run_and_capture_stdout(args):
        with io.StringIO() as buf, redirect_stdout(buf):
            run(args)
            return buf.getvalue()

    def _import(model, source=None, repo=None, backend=None, settings=None):
        args = ["import", "-m", model]
        if settings:
            args += ["--setting", settings]
        if repo:
            args += ["--repo", repo]

        run(args)

    def _remove(model, repo=None):
        args = ["remove", "-m", model]
        if repo:
            args += ["--repo", repo]
        run(args)

    def _list(repo=None):
        args = ["list"]
        if repo:
            args += ["--repo", repo]
        run(args)

    # Start Functionality is contained in ScopedTritonServer

    def _infer(model, prompt=None, protocol=None):
        args = ["infer", "-m", model]
        if prompt:
            args += ["--prompt", prompt]
        if protocol:
            args += ["-i", protocol]
        run(args)

    def _profile(model, backend):
        args = ["profile", "-m", model, "--backend", backend]
        run(args)

    def _metrics():
        args = ["metrics"]
        output = TritonCommands._run_and_capture_stdout(args)
        return json.loads(output)

    def _config(model):
        args = ["config", "-m", model]
        output = TritonCommands._run_and_capture_stdout(args)
        return json.loads(output)

    def _status(protocol="grpc"):
        args = ["status", "-i", protocol]
        output = TritonCommands._run_and_capture_stdout(args)
        return json.loads(output)

    def _clear(repo=None):
        args = ["remove", "-m", "all"]
        if repo:
            args += ["--repo", repo]
        run(args)


# Context Manager to start and kill a server running in background and used by testing functions
class ScopedTritonServer:
    def __init__(self, repo=None, mode="local", timeout=60):
        self.repo = repo
        self.mode = mode
        self.timeout = timeout

    def __enter__(self):
        self.pid = self.run_server(self.repo, self.mode)
        self.wait_for_server_ready(timeout=self.timeout)  # Polling

    def __exit__(self, type, value, traceback):
        self.kill_server(self.pid)
        self.repo, self.mode = None, None

    def run_server(self, repo=None, mode="local"):
        args = ["triton", "start"]
        if repo:
            args += ["--repo", repo]
        if mode:
            args += ["--mode", mode]
        # Use Popen to run the server in the background as a separate process.
        p = Popen(args)
        return p.pid

    def wait_for_server_ready(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            print(
                "Waiting for server to be ready ",
                round(timeout - (time.time() - start)),
                flush=True,
            )
            time.sleep(1)
            try:
                # For simplicity in testing, make sure both HTTP and GRPC endpoints
                # are ready before marking server ready.
                if self.check_server_ready(protocol="http") and self.check_server_ready(
                    protocol="grpc"
                ):
                    return
            except ConnectionRefusedError as e:
                # Dump to log for testing transparency
                print(e)
            except InferenceServerException:
                pass
        raise Exception(f"=== Timeout {timeout} secs. Server not ready. ===")

    def kill_server(self, pid: int, sig: int = 2):
        try:
            proc = psutil.Process(pid)
            proc.send_signal(sig)
            proc.wait()
        except psutil.NoSuchProcess as e:
            print(e)

    def check_server_ready(self, protocol="grpc"):
        status = TritonCommands._status(protocol)
        return status["ready"]
