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

import os
import pytest
from triton_cli.main import run
from triton_cli.parser import KNOWN_MODEL_SOURCES

KNOWN_MODELS = KNOWN_MODEL_SOURCES.keys()
KNOWN_SOURCES = KNOWN_MODEL_SOURCES.values()
TEST_REPOS = [None, os.path.join("tmp", "models")]

CUSTOM_VLLM_MODEL_SOURCES = [("vllm-model", "hf:gpt2")]

CUSTOM_TRTLLM_MODEL_SOURCES = [("trtllm-model", "hf:gpt2")]

# TODO: Add public NGC model for testing
CUSTOM_NGC_MODEL_SOURCES = [("my-llm", "ngc:does-not-exist")]


class TestRepo:
    def _list(self, repo=None):
        args = ["list"]
        if repo:
            args += ["--repo", repo]
        run(args)

    def _clear(self, repo=None):
        args = ["remove", "-m", "all"]
        run(args)

    def _import(self, model, source=None, repo=None, backend=None):
        args = ["import", "-m", model]
        if source:
            args += ["--source", source]
        if repo:
            args += ["--repo", repo]
        if backend:
            args += ["--backend", backend]
        run(args)

    def _remove(self, model, repo=None):
        args = ["remove", "-m", model]
        if repo:
            args += ["--repo", repo]
        run(args)

    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_clear(self, repo):
        self._clear(repo)

    # TODO: Add pre/post repo clear to a fixture for setup/teardown
    @pytest.mark.parametrize("model", KNOWN_MODELS)
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_import_known_model(self, model, repo):
        self._clear(repo)
        self._import(model, repo=repo)
        self._clear(repo)

    @pytest.mark.parametrize("source", KNOWN_SOURCES)
    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_import_known_source(self, source, repo):
        self._clear(repo)
        self._import("known_source", source=source, repo=repo)
        self._clear(repo)

    @pytest.mark.parametrize("model,source", CUSTOM_VLLM_MODEL_SOURCES)
    def test_import_vllm(self, model, source):
        self._clear()
        self._import(model, source=source)
        # TODO: Parse repo to find model, with vllm backend in config
        self._clear()

    @pytest.mark.parametrize("model,source", CUSTOM_TRTLLM_MODEL_SOURCES)
    def test_repo_add_trtllm_build(self, model, source):
        # TODO: Parse repo to find TRT-LLM models and backend in config
        self.repo_clear()
        self.repo_add(model, source=source, backend="tensorrtllm")
        self.repo_clear()
        pass

    @pytest.mark.skip(reason="Pre-built TRT-LLM engines not available")
    def test_import_trtllm_prebuilt(self, model, source):
        # TODO: Parse repo to find TRT-LLM models and backend in config
        pass

    def test_import_no_source(self):
        # TODO: Investigate idiomatic way to assert failures for CLIs
        with pytest.raises(
            Exception, match="Please use a known model, or provide a --source"
        ):
            self._import("no_source", source=None)

    def test_remove(self):
        self._import("gpt2", source="hf:gpt2")
        self._remove("gpt2")

    # TODO: Find a way to raise well-typed errors for testing purposes, without
    # always dumping traceback to user-facing output.
    def test_remove_nonexistent(self):
        with pytest.raises(FileNotFoundError, match="No model folder exists"):
            self._remove("does-not-exist")

    @pytest.mark.parametrize("repo", TEST_REPOS)
    def test_list(self, repo):
        self._list(repo)
