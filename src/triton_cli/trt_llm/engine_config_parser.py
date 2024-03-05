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

from argparse import ArgumentParser
from string import Template
import json

FLAGS = None


def parse_and_substitute(
    triton_model_dir, bls_model_name, engine_dir, token_dir, token_type, dry_run
):
    json_path = engine_dir + "/config.json"
    with open(json_path) as j:
        config_file = json.load(j)

    config_dict = {}
    config_dict["engine_dir"] = engine_dir
    config_dict["tokenizer_dir"] = token_dir
    config_dict["tokenizer_type"] = token_type

    # FIXME: Revert handling using 'build_config' as the key when gpt migrates to using unified builder
    build_config_key = (
        "builder_config"
        if config_file.get("builder_config") is not None
        else "build_config"
    )
    config_dict["triton_max_batch_size"] = config_file[build_config_key][
        "max_batch_size"
    ]
    config_dict["max_queue_delay_microseconds"] = 1000
    # The following parameters are based on NGC's model requirements
    config_dict["bls_instance_count"] = 1
    config_dict["postprocessing_instance_count"] = 1
    config_dict["preprocessing_instance_count"] = 1

    trtllm_filepath = triton_model_dir + "/tensorrt_llm/config.pbtxt"
    substitute(trtllm_filepath, config_dict, dry_run)
    preprocessing_filepath = triton_model_dir + "/preprocessing/config.pbtxt"
    substitute(preprocessing_filepath, config_dict, dry_run)
    postprocessing_filepath = triton_model_dir + "/postprocessing/config.pbtxt"
    substitute(postprocessing_filepath, config_dict, dry_run)
    bls_filepath = triton_model_dir + "/" + bls_model_name + "/config.pbtxt"
    substitute(bls_filepath, config_dict, dry_run)


def substitute(file_path, sub_dict, dry_run):
    with open(file_path) as f:
        pbtxt = Template(f.read())
    pbtxt = pbtxt.safe_substitute(sub_dict)
    if not dry_run:
        with open(file_path, "w") as f:
            f.write(pbtxt)
    else:
        print(pbtxt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--triton_model_dir", "-t", help="path of the .pbtxt to modify")
    parser.add_argument("--engine_dir", "-e", help="directory of the engine")
    parser.add_argument(
        "--token_dir",
        "-m",
        help="directory of the tokens, usually in the downloaded model folder",
    )
    parser.add_argument(
        "--token_type", help="type of tokens specified in token_dir. Default is llama"
    )
    parser.add_argument(
        "--dry_run", "-d", action="store_true", help="do the operation in-place"
    )
    FLAGS = parser.parse_args()
    if FLAGS.token_dir is None:
        FLAGS.token_dir = FLAGS.engine_dir
    if FLAGS.token_type is None:
        FLAGS.token_type = "llama"

    parse_and_substitute(
        FLAGS.triton_model_dir,
        FLAGS.engine_dir,
        FLAGS.token_dir,
        FLAGS.token_type,
        FLAGS.dry_run,
    )
