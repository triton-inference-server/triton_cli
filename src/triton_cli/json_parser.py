#! /usr/bin/env python3
from argparse import ArgumentParser
from string import Template
import json
FLAGS=None

def parse_and_substitute(triton_model_dir, engine_dir, token_dir, token_type, dry_run):
    json_path = engine_dir + "/config.json"
    with open(json_path) as j:
        config_file = json.load(j)

    print(config_file)
    print(type(config_file))
    json.dumps(config_file)
    config_dict = {}
    config_dict["engine_dir"] = engine_dir
    config_dict["tokenizer_dir"]=token_dir
    config_dict["tokenizer_type"]=token_type
    config_dict["triton_max_batch_size"]=config_file["builder_config"]["max_batch_size"]

    print(config_dict)

    trtllm_filepath = triton_model_dir + 'tensorrt_llm/config.pbtxt'
    substitute(trtllm_filepath, config_dict, dry_run)
    preprocessing_filepath = triton_model_dir + 'preprocessing/config.pbtxt'
    substitute(preprocessing_filepath, config_dict, dry_run)
    postprocessing_filepath = triton_model_dir + 'postprocessing/config.pbtxt'
    substitute(postprocessing_filepath, config_dict, dry_run)


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
    parser.add_argument("--token_dir", "-m", help="directory of the tokens, usually in the downloaded model folder")
    parser.add_argument("--token_type", help="default is llama")
    parser.add_argument("--dry_run",
                        "-d",
                        action="store_true",
                        help="do the operation in-place")
    FLAGS = parser.parse_args()
    if FLAGS.token_dir is None :
        FLAGS.token_dir = FLAGS.engine_dir
    FLAGS.token_type = "llama"

    parse_and_substitute(FLAGS.triton_model_dir, FLAGS.engine_dir, FLAGS.token_dir, FLAGS.token_type, FLAGS.dry_run)
