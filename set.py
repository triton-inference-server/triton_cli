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
from signal import SIGTERM, signal, SIGINT
import os
from subprocess import STDOUT, PIPE, Popen, TimeoutExpired


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


# Context Manager to start and kill a server running in background and used by testing functions
class ScopedTritonServer:
    def __init__(self, repo=None, mode="local", timeout=60):
        self.repo = repo
        self.mode = mode
        self.timeout = timeout

    def __enter__(self):
        self.run_server(self.repo, self.mode)
        self.wait_for_server_ready(timeout=self.timeout)  # Polling

    def __exit__(self, type, value, traceback):
        self.kill_server()
        self.repo, self.mode = None, None

    def run_server(self, repo=None, mode="local"):
        args = ["triton", "start"]
        if repo:
            args += ["--repo", repo]
        if mode:
            args += ["--mode", mode]
        # Use Popen to run the server in the background as a separate process.
        print(
            "[CUSTOM] [ScopedTritonServer::run_server] About to perform Popen(triton start)"
        )
        self.proc = Popen(args)

        return self.proc

    def check_pid(self):
        """Check the 'triton start' PID and raise an exception if the process is unhealthy"""
        # Check if the PID exists, an exception is raised if not
        self.check_pid_with_signal()
        # If the PID exists, check the status of the process. Raise an exception
        # for a bad status.
        self.check_pid_status()

    def check_pid_with_signal(self):
        """Check for the existence of a PID by sending signal 0"""
        try:
            self.proc.send_signal(0)
        except psutil.NoSuchProcess as e:
            # PID doesn't exist, passthrough the exception
            raise e

    def check_pid_status(self):
        """Check the status of the 'triton start' process based on its PID"""
        process = psutil.Process(self.proc.pid)
        # NOTE: May need to check other statuses in the future, but zombie was observed
        # in some local test cases.
        if process.status() == psutil.STATUS_ZOMBIE:
            raise Exception(
                f"'triton start' PID {self.proc.pid} was in a zombie state."
            )

    def wait_for_server_ready(self, timeout: int = 60):
        start = time.time()
        while time.time() - start < timeout:
            print(
                "[DEBUG] Waiting for server to be ready ",
                round(timeout - (time.time() - start)),
                flush=True,
            )
            time.sleep(1)
            try:
                print(
                    f"[DEBUG] Checking status of 'triton start' PID {self.proc.pid}..."
                )
                self.check_pid()

                # For simplicity in testing, make sure both HTTP and GRPC endpoints
                # are ready before marking server ready.
                if self.check_server_ready(protocol="http") and self.check_server_ready(
                    protocol="grpc"
                ):
                    print("[DEBUG] Server is ready!")
                    return
            except ConnectionRefusedError as e:
                # Dump to log for testing transparency
                print(e)
            except InferenceServerException:
                pass
        raise Exception(f"=== Timeout {timeout} secs. Server not ready. ===")

    def kill_server(self, timeout: int = 30):
        print("[DEBUG] Attempting to Kill Triton Server")
        try:
            self.proc.terminate()
            # Add wait timeout to avoid hanging if process can't be cleanly
            # stopped for some reason.
            ret = self.proc.wait(timeout=timeout)
            print(f"[DEBUG] RETURNCODE = {ret}")
            if ret != 0:
                print("[DEBUG] Needed to forcefully terminate")
                self.proc.kill()
        except psutil.NoSuchProcess as e:
            print(e)

    def check_server_ready(self, protocol="grpc"):
        status = TritonCommands._status(protocol)
        return status["ready"]

    def __str__(self) -> str:
        return self.get_process_info(self.proc.pid)

    def get_process_info(self, pid):
        try:
            # Create a process object
            process = psutil.Process(pid)

            # Initialize a list to store formatted information
            info = []

            # General information
            info.append(f"Process ID (PID): {process.pid}")
            info.append(f"Name: {process.name()}")
            info.append(f"Status: {process.status()}")
            info.append(f"Executable Path: {process.exe()}")
            info.append(f"Current Working Directory: {process.cwd()}")
            info.append(f"Command Line: {' '.join(process.cmdline())}")
            info.append(f"Parent PID: {process.ppid()}")
            info.append(f"Username: {process.username()}")

            # Memory usage details
            memory_info = process.memory_info()
            info.append(
                f"Memory Info: RSS={memory_info.rss / (1024 ** 2):.2f} MB, VMS={memory_info.vms / (1024 ** 2):.2f} MB"
            )

            # CPU usage details
            cpu_times = process.cpu_times()
            info.append(
                f"CPU Times: user={cpu_times.user:.2f}s, system={cpu_times.system:.2f}s"
            )
            info.append(f"CPU Affinity: {process.cpu_affinity()}")
            info.append(f"CPU Percent: {process.cpu_percent(interval=1.0)}%")

            # I/O statistics
            io_counters = process.io_counters()
            info.append(
                f"I/O Counters: read_bytes={io_counters.read_bytes}, write_bytes={io_counters.write_bytes}"
            )

            # Open files
            open_files = process.open_files()
            if open_files:
                info.append("Open Files:")
                for file in open_files:
                    info.append(f" - {file.path}")
            else:
                info.append("Open Files: None")

            # Child Processes (Subprocesses)
            children = process.children(recursive=True)
            if children:
                info.append("Child Processes:")
                for child in children:
                    child_info = (
                        f" - PID: {child.pid}, Name: {child.name()}, "
                        f"Status: {child.status()}, CPU %: {child.cpu_percent(interval=1.0)}%, "
                        f"Memory: RSS={child.memory_info().rss / (1024 ** 2):.2f} MB"
                    )
                    info.append(child_info)
            else:
                info.append("Child Processes: None")

            # Join all information with line breaks for readability
            return "\n".join(info)

        except psutil.NoSuchProcess:
            return f"No process found with PID {pid}"
        except Exception as e:
            return f"An error occurred: {e}"


MODEL_NAME = "gpt2"
print("[CUSTOM] Clearing /root/models: ")
TritonCommands._clear()
print(f"[CUSTOM] Import {MODEL_NAME}:")
TritonCommands._import("gpt2", backend="tensorrtllm")
# print(f"[CUSTOM] Creating ScopedTritonServer object")
# server_proc = ScopedTritonServer()
# print(f"[CUSTOM] Starting the server")
# server_proc.run_server()
# print(f"[CUSTOM] Printing out server_proc details")
# print(server_proc)
# print(f"[CUSTOM] Sending inference request")
# TritonCommands._infer("gpt2", "Testing! Testing! Testing!", "grpc")
# print(f"[CUSTOM] Printi")
