# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch
import torch.multiprocessing as mp
import time

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import MoeConfig, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.profiler import check_gpt_mem_usage
from tensorrt_llm.quantization import QuantMode

import configparser

import numpy as np

import tensorrt_llm
from tensorrt_llm._utils import (numpy_to_torch, pad_vocab_size,
                                 str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.functional import is_gated_activation
from tensorrt_llm.models import GPTLMHeadModel
from tensorrt_llm.quantization import QuantMode

import dataclasses
import os
import platform

from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

import functools
from collections import defaultdict

import torch.nn as nn
from transformers.pytorch_utils import Conv1D
import shutil

MODEL_NAME = "gpt"
args = {
    "world_size": 1,
    "model_dir": "./c-model/gpt2/1-gpu",
    "dtype": "float16",
    "logits_dtype":"float32",
    "timing_cache":"model.cache",
    "profiling_verbosity":"layer_names_only",
    "log_level":"info",
    "vocab_size": 50257,
    "n_layer":12, 
    "n_positions":1024, 
    "n_embd":768, 
    "n_head":12,
    "hidden_act":"gelu_new",
    "rotary_base":10000.0,
    "rotary_scaling":None, 
    "rotary_pct":0.0, 
    "inter_size":3072, 
    "no_bias":True, 
    "max_batch_size":1, 
    "max_input_len":200, 
    "max_output_len":200, 
    "max_beam_width":1,
    "use_gpt_attention_plugin": "float16",
    "use_gemm_plugin":False, 
    "use_layernorm_plugin":False,
    "parallel_build":False, 
    "enable_context_fmha":False, 
    "enable_context_fmha_fp32_acc":False, 
    "multi_block_mode":False, 
    "gpus_per_node":8, 
    "builder_opt":None,
    "output_dir":Path('engine_outputs'),
    "multi_query_mode":False, 
    "remove_input_padding":True, 
    "use_smooth_quant":False, 
    "use_weight_only":False, 
    "weight_only_precision":"int8",
    "per_channel":False, 
    "per_token":False, 
    "int8_kv_cache":False, 
    "random_seed":None, 
    "paged_kv_cache":True, 
    "tokens_per_block":128, 
    "max_prompt_embedding_table_size":0,
    "use_inflight_batching":True, 
    "use_parallel_embedding":False, 
    "embedding_sharding_dim":0, 
    "use_embedding_sharing":False, 
    "use_lookup_plugin":False, 
    "gather_all_token_logits":False, 
    "enable_fp8":False, 
    "fp8_kv_cache":False, 
    "max_num_tokens":None, 
    "strongly_typed":False, 
    "use_custom_all_reduce":False, 
    "use_lora_plugin":False, 
    "max_draft_len":0, 
    "use_paged_context_fmha":False, 
    "use_context_fmha_for_generation":False, 
    "lora_target_modules":None, 
    "moe_num_experts":0, 
    "moe_top_k":0,
    "moe_tp_mode":MoeConfig.ParallelismMode.TENSOR_PARALLEL, 
    "moe_renorm_mode":MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE, 
    "bias":True, 
    "quant_mode":QuantMode(0), 
    "moe_config":MoeConfig(num_experts=0, top_k=0, tp_mode=MoeConfig.ParallelismMode.TENSOR_PARALLEL, normalization_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
    "gather_context_logits":False,
    "gather_generation_logits":False,
}

converter_args = {
    "out_dir":"./c-model/gpt2",
    "tensor_parallelism":1, 
    "processes":1, 
    "calibrate_kv_cache":False, 
    "smoothquant":None, 
    "model":"gpt2",
    "storage_type":"float16",
    "dataset_cache_dir":None, 
    "load_model_on_cpu":False, 
    "convert_model_on_cpu":False,
    "in_file":"gpt2",
}

class GPTBuilder():

    def __init__(self, engine_output_path: Path):
        global args; args["output_dir"] = engine_output_path

    def get_engine_name(self, model, dtype, tp_size, rank):
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)

    def serialize_engine(self, engine, path):
        logger.info(f'Serializing engine to {path}...')
        tik = time.time()
        with open(path, 'wb') as f:
            f.write(engine)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'Engine serialized. Total time: {t}')


    def build_rank_engine(self, builder: Builder,
                        builder_config: tensorrt_llm.builder.BuilderConfig,
                        engine_name, rank):
        '''
        @brief: Build the engine on the given rank.
        @param rank: The rank to build the engine.
        @param args: The cmd line arguments.
        @return: The built engine.
        '''
        kv_dtype = str_dtype_to_trt(args["dtype"])

        # Share_embedding_table can be set True only when:
        # 1) the weight for lm_head() does not exist while other weights exist
        # 2) For multiple-processes, use_parallel_embedding=True and embedding_sharding_dim == 0.
        # Besides, for TensorRT 9.0, we can observe the engine size reduction when the lookup and gemm plugin are enabled.
        share_embedding_table = False
        if args["use_embedding_sharing"]:
            if args["world_size"] > 1:
                if args["model_dir"] is not None and args["embedding_sharding_dim"] == 0 and args["use_parallel_embedding"]:
                    share_embedding_table = self.check_embedding_share(args["model_dir"])
            else:
                if args["model_dir"] is not None:
                    share_embedding_table = self.check_embedding_share(args["model_dir"])

            if not share_embedding_table:
                logger.warning(f'Cannot share the embedding lookup table.')

        if share_embedding_table:
            logger.info(
                'Engine will share embedding and language modeling weights.')

        # Initialize Module
        tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
            num_layers=args["n_layer"],
            num_heads=args["n_head"],
            hidden_size=args["n_embd"],
            inter_size=args["inter_size"],
            vocab_size=args["vocab_size"],
            hidden_act=args["hidden_act"],
            max_position_embeddings=args["n_positions"],
            position_embedding_type=PositionEmbeddingType.learned_absolute
            if args["rotary_pct"] == 0.0 else PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_percentage=args["rotary_pct"],
            rotary_base=args["rotary_base"],
            rotary_scaling=args["rotary_scaling"],
            dtype=kv_dtype,
            logits_dtype=args["logits_dtype"],
            mapping=Mapping(world_size=args["world_size"],
                            rank=rank,
                            tp_size=args["world_size"]),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling,
            quant_mode=args["quant_mode"],
            bias=args["bias"],
            num_kv_heads=1 if args["multi_query_mode"] else args["n_head"],
            use_prompt_tuning=args["max_prompt_embedding_table_size"] > 0,
            use_parallel_embedding=args["use_parallel_embedding"],
            embedding_sharding_dim=args["embedding_sharding_dim"],
            share_embedding_table=share_embedding_table,
            moe_config=args["moe_config"],
        )

        if args["use_smooth_quant"] or args["use_weight_only"]:
            tensorrt_llm_gpt = quantize_model(tensorrt_llm_gpt, args["quant_mode"])

        if args["model_dir"] is not None:
            gpt_dummy_fp8_scaling_factors = {
                'fc_act': [0.5 for _ in range(args["n_layer"])],
                'fc_weights': [0.5 for _ in range(args["n_layer"])],
                'proj_act': [0.5 for _ in range(args["n_layer"])],
                'proj_weights': [0.5 for _ in range(args["n_layer"])],
                'qkv_act': [0.5 for _ in range(args["n_layer"])],
                'qkv_weights': [0.5 for _ in range(args["n_layer"])],
                'qkv_output': [0.5 for _ in range(args["n_layer"])],
                'dense_act': [0.5 for _ in range(args["n_layer"])],
                'dense_weights': [0.5 for _ in range(args["n_layer"])],
            }

            self.load_from_ft(tensorrt_llm_gpt,
                        args["model_dir"],
                        rank,
                        args["world_size"],
                        args["dtype"],
                        args["use_parallel_embedding"],
                        args["embedding_sharding_dim"],
                        share_embedding_table,
                        scaling_factors=gpt_dummy_fp8_scaling_factors
                        if args["enable_fp8"] else None)

        # Module -> Network
        network = builder.create_network()
        network.trt_network.name = engine_name
        if args["use_gpt_attention_plugin"]:
            network.plugin_config.set_gpt_attention_plugin(
                dtype=args["use_gpt_attention_plugin"])
        if args["use_gemm_plugin"]:
            if not args["enable_fp8"]:
                network.plugin_config.set_gemm_plugin(dtype=args["use_gemm_plugin"])
            else:
                logger.info(
                    "Gemm plugin does not support FP8. Disabled Gemm plugin.")
        if args["use_layernorm_plugin"]:
            network.plugin_config.set_layernorm_plugin(
                dtype=args["use_layernorm_plugin"])
        assert not (args["enable_context_fmha"] and args["enable_context_fmha_fp32_acc"])
        if args["enable_context_fmha"]:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if args["enable_context_fmha_fp32_acc"]:
            network.plugin_config.set_context_fmha(
                ContextFMHAType.enabled_with_fp32_acc)
        if args["multi_block_mode"]:
            network.plugin_config.enable_mmha_multi_block_mode()
        if args["remove_input_padding"]:
            network.plugin_config.enable_remove_input_padding()
        if args["paged_kv_cache"]:
            network.plugin_config.enable_paged_kv_cache(args["tokens_per_block"])
        if args["use_lora_plugin"]:
            network.plugin_config.set_lora_plugin(dtype=args["use_lora_plugin"])

        # Quantization plugins.
        if args["use_smooth_quant"]:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args["dtype"])
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=args["dtype"])

            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif args["use_weight_only"]:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype=args["dtype"])

        if args["world_size"] > 1:
            network.plugin_config.set_nccl_plugin(args["dtype"],
                                                args["use_custom_all_reduce"])

        if args["use_lookup_plugin"]:
            # Use the plugin for the embedding parallelism and sharing
            network.plugin_config.set_lookup_plugin(dtype=argsdtype)

        if args["use_paged_context_fmha"] or args["max_draft_len"] > 0:
            assert args["enable_context_fmha"] or args["enable_context_fmha_fp32_acc"], "context fmha must be enabled"
            network.plugin_config.set_paged_context_fmha()

        if args["use_context_fmha_for_generation"]:
            logger.warning(
                f'use_context_fmha_for_generation is set. This flag must be used only for testing'
            )
            assert args["use_gpt_attention_plugin"] and args["paged_kv_cache"] and args["use_paged_context_fmha"], "use_context_fmha_for_generation must be used with paged KV cache and attention."
            network.plugin_config.set_context_fmha_for_generation()

        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

            # Forward
            inputs = tensorrt_llm_gpt.prepare_inputs(
                args["max_batch_size"],
                args["max_input_len"],
                args["max_output_len"],
                True,
                args["max_beam_width"],
                args["max_num_tokens"],
                prompt_embedding_table_size=args["max_prompt_embedding_table_size"],
                gather_all_token_logits=args["gather_all_token_logits"],
                max_draft_len=args["max_draft_len"],
                lora_target_modules=args["lora_target_modules"])
            tensorrt_llm_gpt(*inputs)

        tensorrt_llm.graph_rewriting.optimize(network)

        engine = None

        # Network -> Engine
        engine = builder.build_engine(network, builder_config)
        if rank == 0:
            config_path = args["output_dir"] / 'config.json'
            builder.save_config(builder_config, config_path)

        return engine


    def build(self, rank=0):
        # Convert weights first
        #mp.set_start_method("spawn")
        self.hf_gpt_converter()
        torch.cuda.set_device(rank % args["gpus_per_node"])
        tensorrt_llm.logger.set_level(args["log_level"])
        args["output_dir"].mkdir(parents=True, exist_ok=True)
        timing_cache_file = args["timing_cache"] if args["timing_cache"] else args["output_dir"] / "model.cache"
        timing_cache = timing_cache_file

        builder = Builder()
        apply_query_key_layer_scaling = False
        for cur_rank in range(args["world_size"]):
            # NOTE: when only int8 kv cache is used together with paged kv cache no int8 tensors are exposed to TRT
            int8_trt_flag = args["quant_mode"].has_act_or_weight_quant() or (
                args["paged_kv_cache"] == False
                and args["quant_mode"].has_int8_kv_cache())
            num_kv_heads = 1 if args["multi_query_mode"] else args["n_head"]
            builder_config = builder.create_builder_config(
                name=MODEL_NAME,
                precision=args["dtype"],
                timing_cache=timing_cache,
                profiling_verbosity=args["profiling_verbosity"],
                tensor_parallel=args["world_size"],  # TP only
                parallel_build=args["parallel_build"],
                num_layers=args["n_layer"],
                num_heads=args["n_head"],
                num_kv_heads=num_kv_heads,
                hidden_size=args["n_embd"],
                vocab_size=args["vocab_size"],
                hidden_act=args["hidden_act"],
                max_position_embeddings=args["n_positions"],
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                max_batch_size=args["max_batch_size"],
                max_beam_width=args["max_beam_width"],
                max_input_len=args["max_input_len"],
                max_output_len=args["max_output_len"],
                max_num_tokens=args["max_num_tokens"],
                max_draft_len=args["max_draft_len"],
                int8=int8_trt_flag,
                opt_level=args["builder_opt"],
                strongly_typed=args["strongly_typed"],
                max_prompt_embedding_table_size=args
                ["max_prompt_embedding_table_size"],
                gather_context_logits=args["gather_context_logits"],
                gather_generation_logits=args["gather_generation_logits"],
                quant_mode=args["quant_mode"],
                use_parallel_embedding=args["use_parallel_embedding"],
                lora_target_modules=args["lora_target_modules"],
            )
            engine_name = self.get_engine_name(MODEL_NAME, args["dtype"], args["world_size"],
                                        cur_rank)
            engine = self.build_rank_engine(builder, builder_config, engine_name,
                                    cur_rank)
            assert engine is not None, f'Failed to build engine for rank {cur_rank}'
            local_num_kv_heads = (num_kv_heads + args["world_size"] -
                                1) // args["world_size"]
            kv_dtype = str_dtype_to_trt(args["dtype"])
            if args["quant_mode"].has_int8_kv_cache():
                kv_dtype = str_dtype_to_trt('int8')
            elif args["quant_mode"].has_fp8_kv_cache():
                kv_dtype = str_dtype_to_trt('fp8')

            if cur_rank == 0:
                # Use in-memory timing cache for multiple builder passes.
                if not args["parallel_build"]:
                    timing_cache = builder_config.trt_builder_config.get_timing_cache(
                    )

            self.serialize_engine(engine, args["output_dir"] / engine_name)
            del engine

        if rank == 0:
            ok = builder.save_timing_cache(builder_config, timing_cache_file)
            assert ok, "Failed to save timing cache."

        # Cleanup intermmediate
        shutil.rmtree(Path("./c-model")) 

    # FROM weight.py
    def gen_suffix(self, rank, use_smooth_quant, quant_per_channel):
        suffix = f"{rank}.bin"
        if use_smooth_quant:
            sq_prefix = "int8."
            if quant_per_channel:
                sq_prefix += "col."
            suffix = sq_prefix + suffix
        return suffix


    def extract_layer_idx(self, name):
        ss = name.split('.')
        for s in ss:
            if s.isdigit():
                return s
        return None


    def split(self, v, tp_size, idx, dim=0):
        if tp_size == 1:
            return v
        if len(v.shape) == 1:
            return np.ascontiguousarray(np.split(v, tp_size)[idx])
        elif len(v.shape) == 2:
            return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])
        return None


    def parse_ft_config(self, ini_file):
        gpt_config = configparser.ConfigParser()
        gpt_config.read(ini_file)

        n_embd = gpt_config.getint('gpt', 'n_embd')
        n_head = gpt_config.getint('gpt', 'n_head')
        n_layer = gpt_config.getint('gpt', 'n_layer')
        n_positions = gpt_config.getint('gpt', 'n_positions')
        vocab_size = gpt_config.getint('gpt', 'vocab_size')
        do_layer_norm_before = gpt_config.getboolean('gpt',
                                                    'do_layer_norm_before',
                                                    fallback=True)
        rotary_base = gpt_config.getfloat('gpt', 'rotary_base', fallback=None)
        rotary_scaling_type = gpt_config.get('gpt',
                                            'rotary_scaling_type',
                                            fallback=None)
        rotary_scaling_factor = gpt_config.get('gpt',
                                            'rotary_scaling_factor',
                                            fallback=None)
        if rotary_scaling_type is None:
            if rotary_scaling_factor is not None:
                raise ValueError(
                    f"'rotary_scaling_factor={rotary_scaling_factor}' is found in ini "
                    f"config file {ini_file}, whereas 'rotary_scaling_type' is missing "
                    f"in the config. The 'rotary_scaling_factor' will be ignored and "
                    f"rotary scaling will not be used.")
            rotary_scaling = None
        else:
            if rotary_scaling_factor is None:
                raise ValueError(
                    f"'rotary_scaling_factor={rotary_scaling_factor}' was not found "
                    f"in ini config file {ini_file}, whereas 'rotary_scaling_type' is "
                    f"provided  and equals {repr(rotary_scaling_type)}.")
            rotary_scaling = [rotary_scaling_type, rotary_scaling_factor]
        rotary_pct = gpt_config.getfloat('gpt', 'rotary_pct', fallback=None)
        hidden_act = gpt_config.get('gpt', 'activation_function')
        bias = gpt_config.getboolean('gpt', 'bias', fallback=True)
        inter_size = gpt_config.getint('gpt', 'intermediate_size', fallback=None)
        dtype = gpt_config.get('gpt', 'storage_dtype', fallback='float32')

        if inter_size is None:
            inter_size = 4 * n_embd

        multi_query_mode = gpt_config.getboolean('gpt',
                                                'multi_query_mode',
                                                fallback=False)
        prompt_num_tasks = gpt_config.getint('gpt', 'prompt_num_tasks', fallback=0)
        prompt_max_vocab_size = gpt_config.getint('gpt',
                                                'prompt_max_vocab_size',
                                                fallback=0)
        return {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "n_positions": n_positions,
            "vocab_size": vocab_size,
            "do_layer_norm_before": do_layer_norm_before,
            "hidden_act": hidden_act,
            "rotary_pct": rotary_pct,
            "rotary_base": rotary_base,
            "rotary_scaling": rotary_scaling,
            "bias": bias,
            "inter_size": inter_size,
            "multi_query_mode": multi_query_mode,
            "dtype": dtype,
            "prompt_num_tasks": prompt_num_tasks,
            "prompt_max_vocab_size": prompt_max_vocab_size
        }


    def check_embedding_share(self, dir_path):
        share_embedding_table = False
        lm_file = dir_path + '/' + 'model.lm_head.weight.bin'
        if not Path(lm_file).exists():
            share_embedding_table = True
        return share_embedding_table


    def load_from_ft(self, tensorrt_llm_gpt: GPTLMHeadModel,
                    dir_path,
                    rank=0,
                    tensor_parallel=1,
                    dtype='float32',
                    use_parallel_embedding=False,
                    sharding_dim=0,
                    share_embedding_table=False,
                    scaling_factors=None):
        tensorrt_llm.logger.info('Loading weights from FT...')
        tik = time.time()

        quant_mode = getattr(tensorrt_llm_gpt, 'quant_mode', QuantMode(0))
        if quant_mode.is_int8_weight_only():
            plugin_weight_only_quant_type = torch.int8
        elif quant_mode.is_int4_weight_only():
            plugin_weight_only_quant_type = torch.quint4x2
        _parsed_params = self.parse_ft_config(Path(dir_path) / 'config.ini')
        n_embd = _parsed_params["n_embd"]
        n_head = _parsed_params["n_head"]
        n_layer = _parsed_params["n_layer"]
        n_positions = _parsed_params["n_positions"]
        vocab_size = _parsed_params["vocab_size"]
        do_layer_norm_before = _parsed_params["do_layer_norm_before"]
        hidden_act = _parsed_params["hidden_act"]
        bias = _parsed_params["bias"]
        inter_size = _parsed_params["inter_size"]
        multi_query_mode = _parsed_params["multi_query_mode"]

        np_dtype = str_dtype_to_np(dtype)

        def fromfile(dir_path, name, shape=None, dtype=None):
            dtype = np_dtype if dtype is None else dtype
            p = dir_path + '/' + name
            if Path(p).exists():
                t = np.fromfile(p, dtype=dtype)
                if shape is not None:
                    t = t.reshape(shape)
                return t
            return None

        def set_smoothquant_scale_factors(module,
                                        pre_scale_weight,
                                        dir_path,
                                        basename,
                                        shape,
                                        per_tok_dyn,
                                        per_channel,
                                        is_qkv=False,
                                        rank=None):
            suffix = "bin"
            if per_channel:
                if rank is not None:
                    suffix = f"{rank}." + suffix
                suffix = "col." + suffix

            col_shape = shape if (per_channel or is_qkv) else [1, 1]
            if per_tok_dyn:
                if pre_scale_weight is not None:
                    pre_scale_weight.value = np.array([1.0], dtype=np.float32)
                t = fromfile(dir_path, f"{basename}scale_w_quant_orig.{suffix}",
                            col_shape, np.float32)
                module.per_channel_scale.value = t
            else:
                t = fromfile(dir_path, f"{basename}scale_x_orig_quant.bin", [1],
                            np.float32)
                pre_scale_weight.value = t
                t = fromfile(dir_path, f"{basename}scale_y_accum_quant.{suffix}",
                            col_shape, np.float32)
                module.per_channel_scale.value = t
                t = fromfile(dir_path, f"{basename}scale_y_quant_orig.bin", [1, 1],
                            np.float32)
                module.act_scale.value = t

        # Determine the quantization mode.
        quant_mode = getattr(tensorrt_llm_gpt, "quant_mode", QuantMode(0))
        # Do we use SmoothQuant?
        use_smooth_quant = quant_mode.has_act_and_weight_quant()
        # Do we use quantization per token?
        quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
        # Do we use quantization per channel?
        quant_per_channel = quant_mode.has_per_channel_scaling()

        # Do we use INT4/INT8 weight-only?
        use_weight_only = quant_mode.is_weight_only()

        # Int8 KV cache
        use_int8_kv_cache = quant_mode.has_int8_kv_cache()

        #Enable FP8 Gemm
        enable_fp8_qdq = quant_mode.has_fp8_qdq()

        # Debug
        suffix = self.gen_suffix(rank, use_smooth_quant, quant_per_channel)
        # The type of weights.
        w_type = np_dtype if not use_smooth_quant else np.int8

        pe = fromfile(dir_path, 'model.wpe.bin', [n_positions, n_embd])

        if pe is not None:
            tensorrt_llm_gpt.embedding.position_embedding.weight.value = (pe)

        vocab_embedding_weight = fromfile(dir_path, 'model.wte.bin',
                                        [vocab_size, n_embd])
        if not use_parallel_embedding:
            tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = vocab_embedding_weight
        else:
            if sharding_dim == 0:
                if vocab_size % tensor_parallel != 0:
                    # padding
                    vocab_size_padded = pad_vocab_size(
                        tensorrt_llm_gpt.embedding.vocab_embedding.num_embeddings,
                        tensor_parallel)
                    pad_width = vocab_size_padded - vocab_size
                    vocab_embedding_weight = np.pad(vocab_embedding_weight,
                                                    ((0, pad_width), (0, 0)),
                                                    'constant',
                                                    constant_values=0)
            tensorrt_llm_gpt.embedding.vocab_embedding.weight.value = np.ascontiguousarray(
                split(vocab_embedding_weight,
                    tensor_parallel,
                    rank,
                    dim=sharding_dim))

        if do_layer_norm_before:
            tensorrt_llm_gpt.ln_f.bias.value = (fromfile(
                dir_path, 'model.final_layernorm.bias.bin'))
            tensorrt_llm_gpt.ln_f.weight.value = (fromfile(
                dir_path, 'model.final_layernorm.weight.bin'))

        # share input embedding
        if not share_embedding_table:
            lm_head_weight = fromfile(dir_path, 'model.lm_head.weight.bin',
                                    [vocab_size, n_embd])
            if lm_head_weight is None:
                lm_head_weight = fromfile(dir_path, 'model.wte.bin',
                                        [vocab_size, n_embd])
            if vocab_size % tensor_parallel != 0:
                # padding
                vocab_size_padded = tensorrt_llm_gpt.lm_head.out_features * tensor_parallel
                pad_width = vocab_size_padded - vocab_size
                lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)),
                                        'constant',
                                        constant_values=0)
            tensorrt_llm_gpt.lm_head.weight.value = np.ascontiguousarray(
                self.split(lm_head_weight, tensor_parallel, rank))
        fake_fp8_sf_dt = np.float32
        for i in range(n_layer):
            c_attn_out_dim = (3 * n_embd //
                            tensor_parallel) if not multi_query_mode else (
                                n_embd // tensor_parallel +
                                (n_embd // n_head) * 2)
            gpt_layer = tensorrt_llm_gpt.layers[i]
            gpt_layer.input_layernorm.weight.value = (fromfile(
                dir_path, 'model.layers.' + str(i) + '.input_layernorm.weight.bin'))
            gpt_layer.input_layernorm.bias.value = (fromfile(
                dir_path, 'model.layers.' + str(i) + '.input_layernorm.bias.bin'))
            t = fromfile(
                dir_path, 'model.layers.' + str(i) +
                '.attention.query_key_value.weight.' + suffix,
                [n_embd, c_attn_out_dim], w_type)
            if t is not None:
                dst = gpt_layer.attention.qkv.weight
                if use_smooth_quant:
                    dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                    set_smoothquant_scale_factors(
                        gpt_layer.attention.qkv,
                        gpt_layer.input_layernorm.scale_to_int,
                        dir_path,
                        'model.layers.' + str(i) + '.attention.query_key_value.',
                        [1, c_attn_out_dim],
                        quant_per_token_dyn,
                        quant_per_channel,
                        rank=rank,
                        is_qkv=True)
                elif use_weight_only:
                    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                        numpy_to_torch(t), plugin_weight_only_quant_type)
                    dst.value = torch_to_numpy(processed_torch_weights)
                    scales = tensorrt_llm_gpt.layers[
                        i].attention.qkv.per_channel_scale
                    scales.value = torch_to_numpy(torch_weight_scales)
                else:
                    dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
            if bias:
                t = fromfile(
                    dir_path, 'model.layers.' + str(i) +
                    '.attention.query_key_value.bias.' + str(rank) + '.bin')
                if t is not None:
                    dst = gpt_layer.attention.qkv.bias
                    dst.value = np.ascontiguousarray(t)
            if enable_fp8_qdq:
                tensorrt_llm_gpt.layers[
                    i].attention.qkv.activation_scaling_factor.value = np.array(
                        [scaling_factors['qkv_act'][i]], dtype=fake_fp8_sf_dt)
                tensorrt_llm_gpt.layers[
                    i].attention.qkv.weights_scaling_factor.value = np.array(
                        [scaling_factors['qkv_weights'][i]], dtype=fake_fp8_sf_dt)
                tensorrt_llm_gpt.layers[
                    i].attention.kv_orig_quant_scale.value = np.array(
                        [scaling_factors['qkv_output'][i]], dtype=np.float32)
                tensorrt_llm_gpt.layers[
                    i].attention.kv_quant_orig_scale.value = np.array(
                        [1.0 / scaling_factors['qkv_output'][i]], dtype=np.float32)

            dst = gpt_layer.attention.dense.weight
            t = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.attention.dense.weight.' + suffix,
                [n_embd // tensor_parallel, n_embd], w_type)
            if use_smooth_quant:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))
                dense_scale = getattr(gpt_layer.attention,
                                    "quantization_scaling_factor", None)
                set_smoothquant_scale_factors(
                    gpt_layer.attention.dense, dense_scale, dir_path,
                    'model.layers.' + str(i) + '.attention.dense.', [1, n_embd],
                    quant_per_token_dyn, quant_per_channel)
                # change it to the real smoother if dense layer is applied smooth quant
                gpt_layer.attention.dense.smoother.value = np.ones(
                    [1, n_embd // tensor_parallel], dtype=np.float32)
            elif use_weight_only:
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    numpy_to_torch(t), plugin_weight_only_quant_type)
                dst.value = torch_to_numpy(processed_torch_weights)
                scales = tensorrt_llm_gpt.layers[
                    i].attention.dense.per_channel_scale
                scales.value = torch_to_numpy(torch_weight_scales)
            else:
                dst.value = np.ascontiguousarray(np.transpose(t, [1, 0]))

            if bias:
                dst = gpt_layer.attention.dense.bias
                dst.value = fromfile(
                    dir_path,
                    'model.layers.' + str(i) + '.attention.dense.bias.bin')
            if enable_fp8_qdq:
                tensorrt_llm_gpt.layers[
                    i].attention.dense.activation_scaling_factor.value = np.array(
                        [scaling_factors['dense_act'][i]], dtype=fake_fp8_sf_dt)
                tensorrt_llm_gpt.layers[
                    i].attention.dense.weights_scaling_factor.value = np.array(
                        [scaling_factors['dense_weights'][i]], dtype=fake_fp8_sf_dt)

            dst = gpt_layer.post_layernorm.weight
            dst.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.post_attention_layernorm.weight.bin')

            dst = gpt_layer.post_layernorm.bias
            dst.value = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.post_attention_layernorm.bias.bin')
            t = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_h_to_4h.weight.' + suffix,
                [n_embd, inter_size // tensor_parallel], w_type)
            if use_smooth_quant:
                tensorrt_llm_gpt.layers[
                    i].mlp.fc.weight.value = np.ascontiguousarray(
                        np.transpose(t, [1, 0]))
                set_smoothquant_scale_factors(gpt_layer.mlp.fc,
                                            gpt_layer.post_layernorm.scale_to_int,
                                            dir_path,
                                            'model.layers.' + str(i) +
                                            '.mlp.dense_h_to_4h.',
                                            [1, inter_size // tensor_parallel],
                                            quant_per_token_dyn,
                                            quant_per_channel,
                                            rank=rank)
            elif use_weight_only:
                dst = gpt_layer.mlp.fc.weight
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    numpy_to_torch(t), plugin_weight_only_quant_type)
                dst.value = torch_to_numpy(processed_torch_weights)
                scales = gpt_layer.mlp.fc.per_channel_scale
                scales.value = torch_to_numpy(torch_weight_scales)
            else:
                tensorrt_llm_gpt.layers[
                    i].mlp.fc.weight.value = np.ascontiguousarray(
                        np.transpose(t, [1, 0]))
            if bias:
                gpt_layer.mlp.fc.bias.value = fromfile(
                    dir_path, 'model.layers.' + str(i) +
                    '.mlp.dense_h_to_4h.bias.' + str(rank) + '.bin')
            if is_gated_activation(hidden_act):
                t = fromfile(
                    dir_path, 'model.layers.' + str(i) +
                    '.mlp.dense_h_to_4h.gate.weight.' + str(rank) + '.bin',
                    [n_embd, inter_size // tensor_parallel])
                tensorrt_llm_gpt.layers[
                    i].mlp.gate.weight.value = np.ascontiguousarray(
                        np.transpose(t, [1, 0]))
            if enable_fp8_qdq:
                tensorrt_llm_gpt.layers[
                    i].mlp.fc.activation_scaling_factor.value = np.array(
                        [scaling_factors['fc_act'][i]], dtype=fake_fp8_sf_dt)
                tensorrt_llm_gpt.layers[
                    i].mlp.fc.weights_scaling_factor.value = np.array(
                        [scaling_factors['fc_weights'][i]], dtype=fake_fp8_sf_dt)

            t = fromfile(
                dir_path,
                'model.layers.' + str(i) + '.mlp.dense_4h_to_h.weight.' + suffix,
                [inter_size // tensor_parallel, n_embd], w_type)
            if use_smooth_quant:
                tensorrt_llm_gpt.layers[
                    i].mlp.proj.weight.value = np.ascontiguousarray(
                        np.transpose(t, [1, 0]))
                proj_scale = getattr(gpt_layer.mlp, "quantization_scaling_factor",
                                    None)
                set_smoothquant_scale_factors(
                    gpt_layer.mlp.proj, proj_scale, dir_path,
                    'model.layers.' + str(i) + '.mlp.dense_4h_to_h.', [1, n_embd],
                    quant_per_token_dyn, quant_per_channel)
                # change it to the real smoother if proj layer is applied smooth quant
                gpt_layer.mlp.proj.smoother.value = np.ones(
                    [1, inter_size // tensor_parallel], dtype=np.float32)
            elif use_weight_only:
                dst = gpt_layer.mlp.proj.weight
                processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
                    numpy_to_torch(t), plugin_weight_only_quant_type)
                dst.value = torch_to_numpy(processed_torch_weights)
                scales = gpt_layer.mlp.proj.per_channel_scale
                scales.value = torch_to_numpy(torch_weight_scales)
            else:
                gpt_layer.mlp.proj.weight.value = (np.ascontiguousarray(
                    np.transpose(t, [1, 0])))
            if bias:
                gpt_layer.mlp.proj.bias.value = fromfile(
                    dir_path,
                    'model.layers.' + str(i) + '.mlp.dense_4h_to_h.bias.bin')

            if use_int8_kv_cache:
                t = fromfile(
                    dir_path, 'model.layers.' + str(i) +
                    '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                    np.float32)
                tensorrt_llm_gpt.layers[
                    i].attention.kv_orig_quant_scale.value = 1.0 / t
                gpt_layer.attention.kv_quant_orig_scale.value = t

            if enable_fp8_qdq:
                tensorrt_llm_gpt.layers[
                    i].mlp.proj.activation_scaling_factor.value = np.array(
                        [scaling_factors['proj_act'][i]], dtype=fake_fp8_sf_dt)
                tensorrt_llm_gpt.layers[
                    i].mlp.proj.weights_scaling_factor.value = np.array(
                        [scaling_factors['proj_weights'][i]], dtype=fake_fp8_sf_dt)

        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')

    # From hf_gpt_convert.py
    @torch.no_grad()
    def smooth_gpt_model(self, model, scales, alpha):
        # Smooth the activation and weights with smoother = $\diag{s}$
        for name, module in model.named_modules():
            if not isinstance(module, GPT2Block):
                continue

            # qkv_proj
            layer_name = name + ".attn.c_attn"
            smoother = self.smooth_gemm(module.attn.c_attn.weight.T,
                                scales[layer_name]["x"], module.ln_1.weight,
                                module.ln_1.bias, alpha)
            scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
            scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

            # fc1
            layer_name = name + ".mlp.c_fc"
            smoother = self.smooth_gemm(module.mlp.c_fc.weight.T,
                                scales[layer_name]["x"], module.ln_2.weight,
                                module.ln_2.bias, alpha)
            scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
            scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


    # SantaCoder separates Q projection from KV projection
    def concat_qkv_weight_bias(self, q, hf_key, hf_model):
        kv = hf_model.state_dict()[hf_key.replace("q_attn", "kv_attn")]
        return torch.cat([q, kv], dim=-1)


    # StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
    def transpose_weights(self, hf_name, param):
        weight_to_transpose = ["c_attn", "c_proj", "c_fc"]
        if any([k in hf_name for k in weight_to_transpose]):
            if len(param.shape) == 2:
                param = param.transpose(0, 1)
        return param


    def gpt_to_ft_name(self, orig_name):
        global_weights = {
            "transformer.wpe.weight": "model.wpe",
            "transformer.wte.weight": "model.wte",
            "transformer.ln_f.bias": "model.final_layernorm.bias",
            "transformer.ln_f.weight": "model.final_layernorm.weight",
            "lm_head.weight": "model.lm_head.weight"
        }

        if orig_name in global_weights:
            return global_weights[orig_name]

        _, _, layer_id, *weight_name = orig_name.split(".")
        layer_id = int(layer_id)
        weight_name = "transformer." + ".".join(weight_name)

        per_layer_weights = {
            "transformer.ln_1.bias": "input_layernorm.bias",
            "transformer.ln_1.weight": "input_layernorm.weight",
            "transformer.attn.c_attn.bias": "attention.query_key_value.bias",
            "transformer.attn.c_attn.weight": "attention.query_key_value.weight",
            "transformer.attn.q_attn.weight": "attention.query.weight",
            "transformer.attn.q_attn.bias": "attention.query.bias",
            "transformer.attn.kv_attn.weight": "attention.key_value.weight",
            "transformer.attn.kv_attn.bias": "attention.key_value.bias",
            "transformer.attn.c_proj.bias": "attention.dense.bias",
            "transformer.attn.c_proj.weight": "attention.dense.weight",
            "transformer.ln_2.bias": "post_attention_layernorm.bias",
            "transformer.ln_2.weight": "post_attention_layernorm.weight",
            "transformer.mlp.c_fc.bias": "mlp.dense_h_to_4h.bias",
            "transformer.mlp.c_fc.weight": "mlp.dense_h_to_4h.weight",
            "transformer.mlp.c_proj.bias": "mlp.dense_4h_to_h.bias",
            "transformer.mlp.c_proj.weight": "mlp.dense_4h_to_h.weight",
        }
        return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


    @torch.no_grad()
    def hf_gpt_converter(self):
        infer_tp = converter_args["tensor_parallelism"]
        multi_query_mode = True if converter_args["model"] in ["santacoder", "starcoder"
                                                ] else False
        saved_dir = Path(converter_args["out_dir"]) / f"{infer_tp}-gpu"
        saved_dir.mkdir(parents=True, exist_ok=True)

        # load position_embedding from rank 0
        model = AutoModelForCausalLM.from_pretrained(converter_args["in_file"],
                                                    torch_dtype="auto",
                                                    device_map="auto",
                                                    trust_remote_code=True)
        
        if converter_args["load_model_on_cpu"]:
            model = model.cpu()
            torch.cuda.empty_cache()
        act_range = {}
        if converter_args["smoothquant"] is not None or converter_args["calibrate_kv_cache"]:
            os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
                "TOKENIZERS_PARALLELISM", "false")
            from datasets import load_dataset
            dataset = load_dataset("lambada",
                                split="validation",
                                cache_dir=converter_args["dataset_cache_dir"])
            act_range = self.capture_activation_range(
                model, AutoTokenizer.from_pretrained(converter_args["in_file"]), dataset)
            if converter_args["smoothquant"] is not None:
                self.smooth_gpt_model(model, act_range, converter_args["smoothquant"])

        config = configparser.ConfigParser()
        config["gpt"] = {}
        for key in converter_args:
            config["gpt"][key] = f"{key}"
        for k, v in vars(model.config).items():
            config["gpt"][k] = f"{v}"
        config["gpt"]["storage_dtype"] = converter_args["storage_type"]
        config["gpt"]["multi_query_mode"] = str(multi_query_mode)
        with open(saved_dir / "config.ini", 'w') as configfile:
            config.write(configfile)

        storage_type = str_dtype_to_torch(converter_args["storage_type"])

        global_ft_weights = [
            "model.wpe", "model.wte", "model.final_layernorm.bias",
            "model.final_layernorm.weight", "model.lm_head.weight"
        ]

        int8_outputs = None
        if converter_args["calibrate_kv_cache"]:
            int8_outputs = "kv_cache_only"
        if converter_args["smoothquant"] is not None:
            int8_outputs = "all"

        starmap_converter_args = []
        for name, param in model.named_parameters():
            if "weight" not in name and "bias" not in name:
                continue
            ft_name = self.gpt_to_ft_name(name)

            if converter_args["convert_model_on_cpu"]:
                param = param.cpu()
            if converter_args["model"] == "starcoder":
                param = self.transpose_weights(name, param)
            if ft_name in global_ft_weights:
                torch_to_numpy(param.to(storage_type).cpu()).tofile(
                    saved_dir / f"{ft_name}.bin")
            else:
                if 'q_attn' in name:
                    param = self.concat_qkv_weight_bias(param, name, model)
                    ft_name = ft_name.replace("query", "query_key_value")
                # Needed by QKV projection weight split. With multi_query_mode one does not simply take
                # out_dim and divide it by 3 to get local_dim because out_dim = local_dim + 2 * head_size
                local_dim = model.transformer.h[
                    0].attn.embed_dim if multi_query_mode else None
                if converter_args["processes"] == 1:
                    self.split_and_save_weight(
                        0, saved_dir, infer_tp, ft_name, param.to(storage_type),
                        storage_type, act_range.get(name.replace(".weight", "")), {
                            "int8_outputs": int8_outputs,
                            "multi_query_mode": multi_query_mode,
                            "local_dim": local_dim
                        })
                else:
                    starmap_converter_args.append(
                        (0, saved_dir, infer_tp, ft_name, param.to(storage_type),
                        storage_type, act_range.get(name.replace(".weight", "")), {
                            "int8_outputs": int8_outputs,
                            "multi_query_mode": multi_query_mode,
                            "local_dim": local_dim
                        }))

        starmap_converter_args = tqdm(starmap_converter_args, desc="saving weights")
        if converter_args["processes"] > 1:
            with mp.Pool(converter_args["processes"]) as pool:
                pool.starmap(self.split_and_save_weight, starmap_converter_args)

    # From smoothquant.py
    @torch.no_grad()
    def apply_smoothing(self, scales,
                        gemm_weights,
                        layernorm_weights=None,
                        layernorm_bias=None,
                        dtype=torch.float32,
                        layernorm_1p=False):
        if not isinstance(gemm_weights, list):
            gemm_weights = [gemm_weights]

        if layernorm_weights is not None:
            assert layernorm_weights.numel() == scales.numel()
            layernorm_weights.div_(scales).to(dtype)
        if layernorm_bias is not None:
            assert layernorm_bias.numel() == scales.numel()
            layernorm_bias.div_(scales).to(dtype)
        if layernorm_1p:
            layernorm_weights += (1 / scales) - 1

        for gemm in gemm_weights:
            gemm.mul_(scales.view(1, -1)).to(dtype)


    @torch.no_grad()
    def smooth_gemm(self, gemm_weights,
                    act_scales,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    alpha=0.5,
                    weight_scales=None):
        if not isinstance(gemm_weights, list):
            gemm_weights = [gemm_weights]
        orig_dtype = gemm_weights[0].dtype

        for gemm in gemm_weights:
            # gemm_weights are expected to be transposed
            assert gemm.shape[1] == act_scales.numel()

        if weight_scales is None:
            weight_scales = torch.cat(
                [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
                dim=0)
            weight_scales = weight_scales.max(dim=0)[0]
        weight_scales.to(float).clamp(min=1e-5)
        scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
                weight_scales.pow(1 - alpha)).clamp(min=1e-5)

        self.apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                        orig_dtype)

        return scales

    @torch.no_grad()
    def capture_activation_range(self, model,
                                tokenizer,
                                dataset,
                                num_samples=512,
                                seq_len=512):
        model.eval()
        device = next(model.parameters()).device
        act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

        def stat_tensor(self, name, tensor, act_scales, key):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float()

            if act_scales[name][key] is None:
                act_scales[name][key] = comming_max
            else:
                act_scales[name][key] = torch.max(act_scales[name][key],
                                                comming_max)

        def stat_input_hook(self, m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            self.stat_tensor(name, x, act_scales, "x")
            self.stat_tensor(name, y, act_scales, "y")

            if act_scales[name]["w"] is None:
                act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                            None).max(dim=0)[0]

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name)))

        for i in tqdm(range(num_samples), desc="calibrating model"):
            input_ids = tokenizer(dataset[i]["text"],
                                return_tensors="pt",
                                max_length=seq_len,
                                truncation=True).input_ids.to(device)
            model(input_ids)

        for h in hooks:
            h.remove()

        return act_scales

    def save_val(self, val, dir, key, tp_num=None):
        suffix = "bin" if tp_num is None else f"{tp_num}.bin"
        val.tofile(dir / f"model.{key}.{suffix}")


    def save_split(self, split_vals, dir, key, i, split_factor):
        for j, val in enumerate(split_vals):
            self.save_val(val, dir, key, i * split_factor + j)


    def generate_int8(self, weights, act_range, is_qkv=False, multi_query_mode=False):
        """
        This function has two purposes:
        - compute quantized weights, scaled either per-tensor or per-column
        - compute scaling factors

        Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
        CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
        CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

        Here is the list of what we need (T means per-tensor, C per-column):
            - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
            - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
            - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
            - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
            to quant range (int8) (used for CUBLAS) (T, C)

        Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
        but then the model would change depending on the number of GPUs used.

        For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
        as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
        """

        # compute weight scaling factors for fp->int8 and int8->fp
        if is_qkv and not multi_query_mode:
            scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
                dim=-1, keepdims=True)[0].cpu().numpy()
            scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                                -1).cpu().numpy()
        elif is_qkv and multi_query_mode:
            raise ValueError(
                f"Multi-query w/ int8 quant has not been supported yet")
        else:
            scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
            scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
        scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
        scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

        # compute the rest of needed scaling factors
        scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
        scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
        scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
        scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                        scale_w_orig_quant_t)
        scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                        scale_w_orig_quant_c)
        if is_qkv:
            scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                    scale_w_orig_quant_c.shape)
            scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                                scale_w_orig_quant_c.shape)

        to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)
        return {
            "weight.int8": to_i8(weights * scale_w_orig_quant_t),
            "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
            "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
            "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
            "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
            "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
            "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
            "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
        }


    def write_int8(self, vals,
                dir,
                base_key,
                split_dim,
                tp_rank,
                split_factor,
                kv_cache_only=False):
        if not kv_cache_only:
            self.save_split(np.split(vals["weight.int8"], split_factor, axis=split_dim),
                    dir, f"{base_key}.weight.int8", tp_rank, split_factor)
            self.save_split(
                np.split(vals["weight.int8.col"], split_factor, axis=split_dim),
                dir, f"{base_key}.weight.int8.col", tp_rank, split_factor)

        saved_keys_once = ["scale_y_quant_orig"]
        if not kv_cache_only:
            saved_keys_once += [
                "scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant"
            ]
        # per-column scaling factors are loaded per-gpu for ColumnParallel GEMMs (QKV, FC1)
        if not kv_cache_only:
            if split_dim == -1:
                self.save_split(
                    np.split(vals["scale_w_quant_orig.col"],
                            split_factor,
                            axis=split_dim), dir,
                    f"{base_key}.scale_w_quant_orig.col", tp_rank, split_factor)
                self.save_split(
                    np.split(vals["scale_y_accum_quant.col"],
                            split_factor,
                            axis=split_dim), dir,
                    f"{base_key}.scale_y_accum_quant.col", tp_rank, split_factor)
            else:
                saved_keys_once += [
                    "scale_w_quant_orig.col", "scale_y_accum_quant.col"
                ]

        if tp_rank == 0:
            for save_key in saved_keys_once:
                self.save_val(vals[save_key], dir, f"{base_key}.{save_key}")


    # Note: in multi_query_mode, only query heads are split between multiple GPUs, while key/value head
    # are not split as there is only one head per key/value.
    @torch.no_grad()
    def split_and_save_weight(self, tp_rank, saved_dir, split_factor, key, vals,
                            storage_type, act_range, config):
        use_attention_nemo_shape = config.get("use_attention_nemo_shape", False)
        split_gated_activation = config.get("split_gated_activation", False)
        num_attention_heads = config.get("num_attention_heads", 0)
        tp_size = config.get("tp_size", 1)
        int8_outputs = config.get("int8_outputs", None)
        multi_query_mode = config.get("multi_query_mode", False)
        local_dim = config.get("local_dim", None)

        save_int8 = int8_outputs == "all" or int8_outputs == "kv_cache_only"

        if not isinstance(vals, list):
            vals = [vals]

        if config.get("transpose_weights", False) and vals[0].ndim == 2:
            vals = [val.T for val in vals]
        if "layernorm.weight" in key and config.get("apply_layernorm_1p", False):
            vals = [val + 1.0 for val in vals]
        vals = [torch_to_numpy(val.cpu().to(storage_type)) for val in vals]

        if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
            "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
            "post_attention_layernorm.bias" in key or "mlp.dense_4h_to_h.bias" in key or \
            "final_layernorm.weight" in key or "final_layernorm.bias" in key:

            # shared weights, only need to convert the weights of rank 0
            if tp_rank == 0:
                self.save_val(vals[0], saved_dir, key)

        elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
            cat_dim = 0
            val = np.concatenate(vals, axis=cat_dim)
            split_vals = np.split(val, split_factor, axis=cat_dim)
            self.save_split(split_vals, saved_dir, key, tp_rank, split_factor)
            if act_range is not None and int8_outputs == "all":
                base_key = key.replace(".weight", "")
                vals_i8 = self.generate_int8(val,
                                        act_range,
                                        multi_query_mode=multi_query_mode)
                self.write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank,
                        split_factor)

        elif "mlp.dense_h_to_4h.weight" in key or "mlp.dense_h_to_4h.bias" in key:
            if split_gated_activation:
                splits = [np.split(val, 2, axis=-1) for val in vals]
                vals, gates = list(zip(*splits))
            cat_dim = -1
            val = np.concatenate(vals, axis=cat_dim)
            split_vals = np.split(val, split_factor, axis=cat_dim)
            self.save_split(split_vals, saved_dir, key, tp_rank, split_factor)
            if act_range is not None and int8_outputs == "all":
                base_key = key.replace(".weight", "")
                vals_i8 = self.generate_int8(val,
                                        act_range,
                                        multi_query_mode=multi_query_mode)
                self.write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank,
                        split_factor)

            if split_gated_activation:
                assert not save_int8
                prefix, dot, suffix = key.rpartition(".")
                key = prefix + ".gate" + dot + suffix

                gate = np.concatenate(gates, axis=cat_dim)
                split_vals = np.split(gate, split_factor, axis=cat_dim)
                self.save_split(split_vals, saved_dir, key, tp_rank, split_factor)

        elif "attention.query_key_value.bias" in key:
            if local_dim is None:
                local_dim = vals[0].shape[-1] // 3

            if multi_query_mode:
                val = vals[0]
                # out_feature = local_dim + 2 * head_size; assumes local_dim equals to hidden_dim
                b_q, b_kv = np.split(val, [local_dim], axis=-1)
                b_q_split = np.split(b_q, split_factor, axis=-1)
                split_vals = [np.concatenate((i, b_kv), axis=-1) for i in b_q_split]
            else:
                if use_attention_nemo_shape:
                    head_num = num_attention_heads // tp_size
                    size_per_head = local_dim // head_num
                    nemo_shape = (head_num, 3, size_per_head)
                    vals = [val.reshape(nemo_shape) for val in vals]
                    vals = [val.transpose(1, 0, 2) for val in vals]

                vals = [val.reshape(3, local_dim) for val in vals]
                val = np.concatenate(vals, axis=-1)
                split_vals = np.split(val, split_factor, axis=-1)
            self.save_split(split_vals, saved_dir, key, tp_rank, split_factor)

        elif "attention.query_key_value.weight" in key:
            hidden_dim = vals[0].shape[0]
            if local_dim is None:
                local_dim = vals[0].shape[-1] // 3
            if multi_query_mode:
                val = vals[0]
                # out_feature = local_dim + 2 * head_size; assumes local_dim equals to hidden_dim
                head_size = (val.shape[-1] - local_dim) // 2
                val = val.reshape(hidden_dim, local_dim + 2 * head_size)
                w_q, w_kv = np.split(val, [local_dim], axis=-1)
                w_q_split = np.split(w_q, split_factor, axis=-1)
                split_vals = [np.concatenate((i, w_kv), axis=-1) for i in w_q_split]
            else:
                if use_attention_nemo_shape:
                    head_num = num_attention_heads // tp_size
                    size_per_head = hidden_dim // num_attention_heads
                    vals = [
                        val.reshape(hidden_dim, head_num, 3, size_per_head)
                        for val in vals
                    ]
                    vals = [val.transpose(0, 2, 1, 3) for val in vals]

                vals = [val.reshape(hidden_dim, 3, local_dim) for val in vals]
                cat_dim = -1
                val = np.concatenate(vals, axis=cat_dim)
                split_vals = np.split(val, split_factor, axis=cat_dim)
            self.save_split(split_vals, saved_dir, key, tp_rank, split_factor)
            if save_int8:
                base_key = key.replace(".weight", "")
                vals_i8 = self.generate_int8(val,
                                        act_range,
                                        is_qkv=True,
                                        multi_query_mode=multi_query_mode)
                self.write_int8(vals_i8,
                        saved_dir,
                        base_key,
                        cat_dim,
                        tp_rank,
                        split_factor,
                        kv_cache_only=int8_outputs == "kv_cache_only")
        elif ("attention.query.weight" in key or "attention.query.bias" in key
            or "attention.key_value.weight" in key
            or "attention.key_value.bias" in key):
            pass
        else:
            print(f"[WARNING] {key} not handled by converter")



