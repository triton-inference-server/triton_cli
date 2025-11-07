# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import json
import time
import psutil
import subprocess
import requests
from contextlib import redirect_stdout
from triton_cli.main import run
from subprocess import Popen


class TritonCommands:
    def _run_and_capture_stdout(args):
        with io.StringIO() as buf, redirect_stdout(buf):
            run(args)
            return buf.getvalue()

    def _import(model, source=None, repo=None, backend=None):
        args = ["import", "-m", model]
        if source:
            args += ["--source", source]
        if repo:
            args += ["--repo", repo]
        if backend:
            args += ["--backend", backend]
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

    def _profile(model, backend=None, service_kind=None, endpoint_type=None, url=None):
        args = ["profile", "-m", model]
        if backend:
            args += ["--backend", backend]
        if service_kind:
            args += ["--service-kind", service_kind]
        if endpoint_type:
            args += ["--endpoint-type", endpoint_type]
        if url:
            args += ["--url", url]
        # For mock models, use a real tokenizer since mock_llm doesn't exist on HuggingFace
        if model == "mock_llm":
            args += ["--tokenizer", "gpt2"]
        # NOTE: With default parameters, genai-perf may take upwards of 1m30s or 2m to run,
        # so limit the genai-perf run with --request-count to reduce time for testing purposes.
        args += ["--synthetic-input-tokens-mean", "100", "--", "--request-count", "10"]
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
    def __init__(self, repo=None, mode="local", timeout=60, frontend=None):
        self.repo = repo
        self.mode = mode
        self.frontend = frontend
        self.timeout = timeout
        self.proc = None

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.proc = self.run_server(self.repo, self.mode, self.frontend)
        self.wait_for_server_ready(timeout=self.timeout)  # Polling

    def stop(self):
        self.kill_server()

    def run_server(self, repo=None, mode="local", frontend=None):
        args = ["triton", "start"]
        if repo:
            args += ["--repo", repo]
        if mode:
            args += ["--mode", mode]
        if frontend:
            args += ["--frontend", frontend]
        # Use Popen to run the server in the background as a separate process.
        p = Popen(args)
        return p

    def wait_for_server_ready(self, timeout: int = 60):
        if not self.proc:
            raise RuntimeError("Server process wasn't started")

        start = time.time()
        while True:
            try:
                if self.check_server_ready():
                    break
            except Exception as err:
                result = self.proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > timeout:
                    raise RuntimeError("Server failed to start in time.") from err

    def kill_server(self, timeout: int = 60):
        if not self.proc:
            # If process wasn't started by this point, just print the error and
            # gracefully exit for now.
            print("ERROR: Server process wasn't started")
            return

        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)  # Wait for triton to clean up
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()  # Indefinetely wait until the process is cleaned up.
        except psutil.NoSuchProcess as e:
            print(e)
        except AttributeError as e:
            print(e)

    def check_server_ready(self):
        if self.frontend == "openai":
            # TODO: change this to status command after we have
            # the dedicated OpenAI client
            response = requests.get("http://localhost:9000/health/ready")
            return response.status_code == 200
        else:
            status_grpc = TritonCommands._status(protocol="grpc")
            status_http = TritonCommands._status(protocol="http")
            return status_grpc["ready"] and status_http["ready"]
