"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time

import accelerate
import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

from fastvideo.utils.communications import broadcast_data

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)
from videox_fun.data.bucket_sampler import (ASPECT_RATIO_720P,
                                            ASPECT_RATIO_RANDOM_CROP_720P,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            get_closest_ratio)
from videox_fun.data.dataset_image_video import ImageVideoControlDataset
from videox_fun.models import (AutoencoderKLWan, CLIPModel,
                               CogOmniControlWanModel, WanT5EncoderModel)
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.lora_utils import create_network, merge_lora_to_modules
from videox_fun.utils.params import format_numel_str, get_model_numel
from videox_fun.utils.utils import save_videos_grid

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

@torch.no_grad()
@torch.autocast(device_type="cuda")
def _batch_encode_vae(pixel_values, args, vae):
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    bs = args.vae_mini_batch
    new_pixel_values = []
    for i in range(0, pixel_values.shape[0], bs):
        pixel_values_bs = pixel_values[i : i + bs]
        pixel_values_bs = vae.encode(pixel_values_bs)[0]
        pixel_values_bs = pixel_values_bs.sample()
        new_pixel_values.append(pixel_values_bs)
    return torch.cat(new_pixel_values, dim = 0)

def encode_cogomni_context(args, vae, control_pixel_values, ref_pixel_values):
    control_latents = _batch_encode_vae(control_pixel_values, args, vae)
    ref_image_latents = ref_pixel_values
    if ref_pixel_values.numel():
        # Process reference image latents
        ref_image_latents = []
        for i, ref_item_pixel_values in enumerate(torch.split(ref_pixel_values, 1, dim=1)):
            ref_item_latents = _batch_encode_vae(ref_item_pixel_values, args, vae)
            ref_image_latents.append(ref_item_latents)
        ref_image_latents = torch.cat(ref_image_latents, dim=2)
    return control_latents, ref_image_latents

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="control",
        help=(
            'The format of training data. Support `"control"`'
            ' (default), `"control_ref"`.'
        ),
    )
    parser.add_argument(
        "--control_ref_image",
        type=str,
        default="first_frame",
        help=(
            'The format of training data. Support `"first_frame"`'
            ' (default), `"random"`.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--caption_mode", help='support: (1). "dense_caption", (2). ["dense_caption", "caption"], (3). [["dense_caption", 0.8], ["caption", 0.2]], (4). [["dense_caption", 0.7], ["caption+tags", 0.2], ["tags+caption", 0.1]]'
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="lora path."
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.0, help="lora weight."
    )
    # args for ref_masks
    parser.add_argument(
        "--use_latent_masks_loss", action="store_true", help="whether to use use latent_masks to compute loss."
    )
    parser.add_argument(
        "--dilated_masks_prob", type=float, default=0.3, help="prob of dilate masks."
    )
    parser.add_argument(
        "--rected_masks_prob", type=float, default=0.1, help="prob of rectangle masks."
    )
    parser.add_argument(
        "--ref_rmbg_prob", type=float, default=0.8, help="prob of remove bg of ref image."
    )
    parser.add_argument(
        "--sparse_control_mode", action="store_true", help="whether to use sparse control mode."
    )
    parser.add_argument(
        "--random_ref_frame_mode", action="store_true", help="whethee to use random ref frame mode."
    )
    # args for wan2.2
    parser.add_argument(
        "--boundary_type",
        type=str,
        default="low",
        help=(
            'The format of training data. Support `"low"` and `"high"`'
        ),
    )
    # args for squence parrallel
    parser.add_argument(
        "--sp_size", type=int, default=1, help="squence parrallel size."
    )
    # args for reference training
    parser.add_argument(
        "--max_ref_num_per_subject", type=int, default=1, help="max reference image num for every subject."
    )
    parser.add_argument(
        "--max_subject_num", type=int, default=1, help="max num of subject in a video."
    )
    # args for cogomni control
    parser.add_argument(
        "--lora_rank", type=int, default=128, help="lora rank for cogomni control model."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128, help="lora alpha for cogomni control model."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        print(f"Using DeepSpeed Zero stage: {zero_stage}")
    else:
        zero_stage = 0
        print("DeepSpeed is not enabled.")
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Squence parrallel init
    initialize_sequence_parallel_state(args.sp_size)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
            
    # Get Transformer
    print(config)
    if args.boundary_type == "low" or args.boundary_type == "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    
    transformer3d = CogOmniControlWanModel(
        **filter_kwargs(CogOmniControlWanModel, OmegaConf.to_container(config["transformer_additional_kwargs"]))
    ).to(weight_dtype)

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        transformer3d.load_state_dict_from_single_file(args.transformer_path)
    else:
        transformer3d.load_state_dict_from_single_file(
            os.path.join(args.pretrained_model_name_or_path, sub_path))
    
    if args.lora_path is not None and os.path.exists(args.lora_path):
        accelerator.print("merge lora weights!")
        transformer3d, text_encoder = merge_lora_to_modules(transformer3d, text_encoder, lora_path=args.lora_path, multiplier=args.lora_weight, dtype=weight_dtype)
        accelerator.print("done!")
    

    # Lora will work with this...
    network = create_network(
        1.0,
        args.lora_rank,
        args.lora_alpha,
        text_encoder,
        transformer3d,
        neuron_dropout=None,
        add_lora_in_attn_temporal=True,
        skip_names=["head", "text_embedding", "time_embedding", "time_projection"],
    )
    network.apply_to(
        text_encoder, transformer3d, False and not args.training_with_video_token_length, True
    )
    model_numel, trainable_model_numel = get_model_numel(network)

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        if not zero_stage == 3:
            # TODO(squirrelli): for stateful distributedsampler save
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

            def load_model_hook(models, input_dir):
                # TODO(squirrelli): for stateful distributedsampler load
                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = CogOmniControlWanModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                # for stateful distributedsampler save
                if accelerator.is_main_process:
                    sampler_state_path = os.path.join(output_dir, "sampler_state_dict.pt")
                    sampler_state_dict = batch_sampler.sampler.state_dict()
                    if "epoch" not in sampler_state_dict:
                        sampler_state_dict["epoch"] = batch_sampler.sampler.epoch
                    # revise prefetch_factor
                    sampler_state_dict[batch_sampler.sampler._YIELDED] -= train_dataloader.prefetch_factor * train_dataloader.num_workers
                    sampler_state_dict[batch_sampler.sampler._YIELDED] = max(sampler_state_dict[batch_sampler.sampler._YIELDED], 0)
                    torch.save(sampler_state_dict, sampler_state_path)

            def load_model_hook(models, input_dir):
                # for stateful distributedsampler load
                sampler_state_path = os.path.join(input_dir, "sampler_state_dict.pt")
                if os.path.exists(sampler_state_path):
                    sampler_state_dict = torch.load(sampler_state_path)
                    batch_sampler.sampler.load_state_dict(sampler_state_dict)
                    print(f"Load sampler state from {sampler_state_path}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    logging.info("Add network parameters")
    trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    trainable_params_optim = network.prepare_optimizer_params(
        args.learning_rate / 2, args.learning_rate, args.learning_rate
    )

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    
    # Get the dataset
    train_dataset = ImageVideoControlDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=False,
        caption_mode=args.caption_mode,
        max_ref_num_per_subject=args.max_ref_num_per_subject,
        max_subject_num=args.max_subject_num,
        dilated_masks_prob=args.dilated_masks_prob,
        rected_masks_prob=args.rected_masks_prob,
        ref_rmbg_prob=args.ref_rmbg_prob,
        sparse_control_mode=args.sparse_control_mode,
        random_ref_frame_mode=args.random_ref_frame_mode
    )

    if accelerator.is_main_process:
        train_dataset.show_meta_stats()

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            set_seed(_seed)
        return _worker_init_fn
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 960 * args.video_sample_size for x in ASPECT_RATIO_720P[key]] for key in ASPECT_RATIO_720P.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        dp_size = accelerator.num_processes // nccl_info.sp_size
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=StatefulDistributedSampler(train_dataset, num_replicas=dp_size, rank=nccl_info.group_id, shuffle=True, seed=args.seed),
            dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["idx"] = []
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            # Used in Control Mode
            new_examples["control_pixel_values"] = []
            # Used in Control Ref Mode
            new_examples["mask_pixel_values"] = []
            new_examples["ref_pixel_values"] = []
            new_examples["control_type"] = []

            # Get downsample ratio in image and videos
            data_type       = examples[0]["data_type"]
            pixel_value     = examples[0]["pixel_values"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 960 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_720P[key]] for key in ASPECT_RATIO_720P.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 960 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_720P[key]] for key in ASPECT_RATIO_RANDOM_CROP_720P.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                        # The video will be resized to a lower resolution than its own.
                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        local_video_sample_size = np.random.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 960 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_720P[key]] for key in ASPECT_RATIO_720P.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 960 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_720P[key]] for key in ASPECT_RATIO_RANDOM_CROP_720P.keys()}

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]
            if args.random_ratio_crop:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

            for example in examples:
                new_examples["idx"].append(torch.tensor(example["idx"]))
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                control_pixel_values = control_pixel_values / 255.

                if example["ref_pixel_values"].size:
                    ref_pixel_values = torch.from_numpy(example["ref_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    ref_pixel_values = ref_pixel_values / 255.
                else:
                    ref_pixel_values = torch.from_numpy(example["ref_pixel_values"])

                if example["mask_pixel_values"].size:
                    mask_pixel_values = torch.from_numpy(example["mask_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    mask_pixel_values = mask_pixel_values / 255.
                    mask_pixel_values = torch.where(mask_pixel_values > 0.5, 1.0, 0.0)
                else:
                    mask_pixel_values = torch.from_numpy(example["mask_pixel_values"])

                if args.random_ratio_crop:
                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                    mask_transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                    ])
                else:
                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                    mask_transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(closest_size),
                    ])
    
                new_examples["pixel_values"].append(transform(pixel_values))
                new_examples["control_pixel_values"].append(transform(control_pixel_values))
                new_examples["text"].append(example["text"])
                if ref_pixel_values.numel():
                    new_examples["ref_pixel_values"].append(transform(ref_pixel_values))
                else:
                    new_examples["ref_pixel_values"].append(ref_pixel_values)
                if mask_pixel_values.numel():
                    new_examples["mask_pixel_values"].append(mask_transform(mask_pixel_values))
                else:
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                new_examples["control_type"].append(example["control_type"])

                # Magvae needs the number of frames to be 4n + 1.
                batch_video_length = int(
                    min(
                        batch_video_length,
                        (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1, 
                    )
                )
                if batch_video_length == 0:
                    batch_video_length = 1

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
            new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
            new_examples["ref_pixel_values"] = torch.stack([example for example in new_examples["ref_pixel_values"]])
            new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]]) 
            new_examples["idx"] = torch.stack([example for example in new_examples["idx"]]) 

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + nccl_info.group_id)
        )
    else:
        raise NotImplementedError("Not supported yet!")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    network, optimizer, lr_scheduler = accelerator.prepare(
        network, optimizer, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    transformer3d.to(accelerator.device, dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps // args.sp_size

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total params = {format_numel_str(model_numel)}")
    logger.info(f"  Trainable params = {format_numel_str(trainable_model_numel)}")
    global_step = 0
    first_epoch = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        resume_dir = os.path.dirname(args.resume_from_checkpoint)
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(resume_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            sampler_state_path = os.path.join(os.path.join(resume_dir, path), "sampler_state_dict.pt")
            if os.path.exists(sampler_state_path):
                first_epoch = torch.load(sampler_state_path)["epoch"]
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(f"Load sampler from {sampler_state_path}. Get first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {os.path.join(resume_dir, path)}")
            accelerator.load_state(os.path.join(resume_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    # Calculate the index we need
    boundary        = config['transformer_additional_kwargs'].get('boundary', 0.900)
    split_timesteps = args.train_sampling_steps * boundary
    differences     = torch.abs(noise_scheduler.timesteps - split_timesteps)
    closest_index   = torch.argmin(differences).item()
    if args.boundary_type == "high" or args.boundary_type == "low":
        print(f"The boundary is {boundary} and the boundary_type is {args.boundary_type}. The closest_index we calculate is {closest_index}")
    if args.boundary_type == "high":
        start_num_idx = 0
        train_sampling_steps = closest_index
    elif args.boundary_type == "low":
        start_num_idx = closest_index
        train_sampling_steps = args.train_sampling_steps - closest_index
    else:
        start_num_idx = 0
        train_sampling_steps = args.train_sampling_steps

    idx_sampling = DiscreteSampling(train_sampling_steps, start_num_idx=start_num_idx, uniform_sampling=args.uniform_sampling)

    os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_mse_loss = 0.0

        batch_sampler.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            # NOTE(squirrelli): Synchronize across sequence parallel processes,
            # although data consistency within sp_group is maintained through the sampler,
            # random retries due to video read timeouts may occur.
            # because data is forcibly broadcast here, maybe additional overhead.
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(accelerator.device)
                if args.sp_size > 1:
                    batch[k] = broadcast_data(batch[k])

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                control_pixel_values = batch["control_pixel_values"].cpu()
                mask_pixel_values = batch["mask_pixel_values"].cpu()
                ref_pixel_values = batch["ref_pixel_values"].cpu()

                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                control_pixel_values = rearrange(control_pixel_values, "b f c h w -> b c f h w")
                if ref_pixel_values.numel():
                    ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> b c f h w")
                if mask_pixel_values.numel():
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")

                for idx, (pixel_value, control_pixel_value, mask_pixel_value, ref_pixel_value, text) in enumerate(zip(pixel_values, control_pixel_values, mask_pixel_values, ref_pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    control_pixel_value = control_pixel_value[None, ...]
                    mask_pixel_value = mask_pixel_value[None, ...]
                    ref_pixel_value = ref_pixel_value[None, ...]

                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/rank{accelerator.process_index}-{gif_name[:10]}.gif", rescale=True)
                    save_videos_grid(control_pixel_value, f"{args.output_dir}/sanity_check/rank{accelerator.process_index}-{gif_name[:10]}_control.gif", rescale=True)
                    if ref_pixel_values.numel():
                        save_videos_grid(ref_pixel_value, f"{args.output_dir}/sanity_check/rank{accelerator.process_index}-{gif_name[:10]}_ref.gif", rescale=True)
                    if mask_pixel_value.numel():
                        save_videos_grid(mask_pixel_value, f"{args.output_dir}/sanity_check/rank{accelerator.process_index}-{gif_name[:10]}_mask.gif")
            
            step_start_time = time.time()
            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                control_pixel_values = batch["control_pixel_values"].to(weight_dtype)
                mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)

                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.training_with_video_token_length and zero_stage != 3:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        control_pixel_values = torch.tile(control_pixel_values, (4, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        control_pixel_values = torch.tile(control_pixel_values, (2, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.training_with_video_token_length and zero_stage != 3 and ref_pixel_values.numel():
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        ref_pixel_values = torch.tile(ref_pixel_values, (4, 1, 1, 1, 1))
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        ref_pixel_values = torch.tile(ref_pixel_values, (2, 1, 1, 1, 1))

                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]
                    control_pixel_values = control_pixel_values[:, :temp_n_frames, :, :]
                    if mask_pixel_values.numel():
                        mask_pixel_values = mask_pixel_values[:, :temp_n_frames, :, :]
                    
                # Keep all node same token length to accelerate the traning when resolution grows.
                if args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
            
                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    control_pixel_values = control_pixel_values[:, :actual_video_length, :, :]
                    if mask_pixel_values.numel():
                        mask_pixel_values = mask_pixel_values[:, :actual_video_length, :, :]

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values, args, vae)
                    else:
                        latents = _batch_encode_vae(pixel_values, args, vae)

                    control_latents, ref_image_latents = encode_cogomni_context(args, vae, control_pixel_values, ref_pixel_values)

                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                if args.input_perturbation > 0:
                    # (B, C, F, H, W)
                    noise_offset = torch.randn(noise.shape[0], noise.shape[1], 1, 1, 1, generator=torch_rng, device=latents.device, dtype=weight_dtype)
                    noise = noise + noise_offset * args.input_perturbation

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                if args.sp_size > 1:
                    timesteps = broadcast_data(timesteps)
                    noise = broadcast_data(noise)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents
                
                target_shape = (vae.latent_channels, num_frames, width, height)
                noisy_seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (config["transformer_additional_kwargs"]["patch_size"][1] * config["transformer_additional_kwargs"]["patch_size"][2]) *
                    target_shape[1]
                )
                control_seq_len = noisy_seq_len
                ref_seq_len = math.ceil(
                    (ref_image_latents.shape[3] * ref_image_latents.shape[4]) /
                    (config["transformer_additional_kwargs"]["patch_size"][1] * config["transformer_additional_kwargs"]["patch_size"][2]) *
                    ref_image_latents.shape[2]
                ) if ref_image_latents.numel() else 0

                seq_len = noisy_seq_len + control_seq_len + ref_seq_len

                # pad seq_len for squence parrallel
                seq_len = math.ceil(seq_len / args.sp_size) * args.sp_size

                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        t=timesteps,
                        control_latents=control_latents,
                        ref_image_latents=ref_image_latents,
                        context=prompt_embeds,
                        seq_len=seq_len,
                    )

                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50, latent_masks=None):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    thres_mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * thres_mask
                    if latent_masks is not None:
                        latent_masks = torch.mean(latent_masks, dim=1, keepdim=True)
                        latent_masks = torch.where(latent_masks > 0.1, 1.0, 0.0)
                        mask_weight = (1 / latent_masks.mean())**0.5
                        mask_weight.clamp_(min=1, max=10)
                        masked_loss = latent_masks * masked_loss * mask_weight
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                mse_loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = mse_loss
                loss = loss.mean()

                if args.motion_sub_loss and noise_pred.size()[2] > 1:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()

                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_mse_loss += avg_mse_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"train_mse_loss": train_mse_loss}, step=global_step)

                train_loss = 0.0
                train_mse_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(),
                    "mse_loss": mse_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                step_time = time.time() - step_start_time
                accelerator.print(logs, f"Step num: {global_step}, Step time: {step_time:.2f}s")

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_model(safetensor_save_path, accelerator.unwrap_model(network))
        if args.save_state:
            accelerator.save_state(accelerator_save_path)
        logger.info(f"Saved state to {accelerator_save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
