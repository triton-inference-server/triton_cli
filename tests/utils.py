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

import io
import json
import time
import psutil
import subprocess
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

    def _profile(model, backend):
        # FIXME: WAR for genai-perf bug in 24.09, remove in 24.10
        import genai_perf

        if genai_perf.__version__ == "0.0.6dev":
            print(
                "[WARNING] Skipping call to 'triton profile' due to known issue in genai-perf"
            )
            return

        args = ["profile", "-m", model, "--backend", backend]
        # NOTE: With default parameters, genai-perf may take upwards of 1m30s or 2m to run,
        # so limit the genai-perf run with --request-count to reduce time for testing purposes.
        args += ["--", "--request-count", "10"]
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


# TODO: Consider removal if other version works
# Context Manager to start and kill a server running in background and used by testing functions
class ScopedTritonServer:
    def __init__(self, repo=None, mode="local", timeout=60):
        self.repo = repo
        self.mode = mode
        self.timeout = timeout

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.proc = self.run_server(self.repo, self.mode)
        self.wait_for_server_ready(
            # url=self.url_for("v2", "health", "ready"),
            timeout=self.timeout
        )  # Polling

    def stop(self):
        self.kill_server()

    def run_server(self, repo=None, mode="local"):
        args = ["triton", "start"]
        if repo:
            args += ["--repo", repo]
        if mode:
            args += ["--mode", mode]
        # Use Popen to run the server in the background as a separate process.
        p = Popen(args)
        return p

    def wait_for_server_ready(self, timeout: int = 60):
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
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)  # Wait for triton to clean up
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()  # Indefinetely wait until the process is cleaned up.
        except psutil.NoSuchProcess as e:
            print(e)

    def check_server_ready(self):
        status_grpc = TritonCommands._status(protocol="grpc")
        status_http = TritonCommands._status(protocol="http")
        return status_grpc["ready"] and status_http["ready"]


# class ScopedTritonServer:
#     def __init__(
#         self,
#         repo: Optional[str] = None,
#         mode: Optional[str] = "local",
#         timeout: int = 120,
#         extra_args: Optional[List[str]] = None,
#         env_dict: Optional[Dict[str, str]] = None,
#     ) -> None:
#         self.started = False
#         self.stopped = False

#         # TODO: Be more coupled with Triton CLI settings
#         self.host = "localhost"
#         self.port = 8000
#         self.start_timeout = timeout
#         self.proc = None

#         env = os.environ.copy()
#         if env_dict is not None:
#             env.update(env_dict)

#         args: List[str] = ["triton", "start"]
#         if repo:
#             args += ["--repo", repo]
#         if mode:
#             args += ["--mode", mode]
#         if extra_args:
#             args += extra_args

#         self.args = args
#         self.env = env

#     def _startup(self):
#         print("Starting server ...")
#         self.proc = subprocess.Popen(
#             self.args,
#             env=self.env,
#             stdout=sys.stdout,
#             stderr=sys.stderr,
#         )
#         print("Waiting for server ready...")
#         # Wait until health endpoint is responsive
#         self._wait_for_server(
#             url=self.url_for("v2", "health", "ready"),
#             timeout=self.start_timeout,
#         )
#         print("DONE: Server ready!")

#     def _shutdown(self):
#         self.proc.terminate()
#         try:
#             wait_secs = 60
#             self.proc.wait(wait_secs)
#         except subprocess.TimeoutExpired:
#             # force kill if needed
#             self.proc.kill()

#     def start(self):
#         print("[DEBUG] Processes before server start:")
#         find_processes_by_name(["triton", "python"])

#         if self.started:
#             print("[WARNING] Server has already been started, skipping startup.")
#             return

#         self.started = True
#         self._startup()

#         print("[DEBUG] Processes after server start:")
#         find_processes_by_name(["triton", "python"])

#     def stop(self):
#         print("[DEBUG] Processes before server stop:")
#         find_processes_by_name(["triton", "python"])

#         # print(f"[DEBUG] =========== [START] STOPPING SERVER {self.proc} ========")
#         if self.stopped:
#             print("[WARNING] Server has already been stopped, skipping shutdown.")
#             return

#         self.stopped = True
#         self._shutdown()
#         # print(f"[DEBUG] =========== [DONE] STOPPING SERVER {self.proc} ========")

#         print("[DEBUG] Processes after server stop:")
#         find_processes_by_name(["triton", "python"])

#     def __enter__(self):
#         self.start()
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.stop()

#     def _wait_for_server(self, *, url: str, timeout: float):
#         start = time.time()
#         while True:
#             try:
#                 if requests.get(url).status_code == 200:
#                     break
#             except Exception as err:
#                 result = self.proc.poll()
#                 if result is not None and result != 0:
#                     raise RuntimeError("Server exited unexpectedly.") from err

#                 time.sleep(0.5)
#                 if time.time() - start > timeout:
#                     raise RuntimeError("Server failed to start in time.") from err

#     @property
#     def url_root(self) -> str:
#         return f"http://{self.host}:{self.port}"

#     def url_for(self, *parts: str) -> str:
#         return self.url_root + "/" + "/".join(parts)
