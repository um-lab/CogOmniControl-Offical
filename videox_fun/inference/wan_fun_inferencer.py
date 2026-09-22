import os
import sys
from typing import List, Optional
from functools import partial


import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import torch.multiprocessing as mp


from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanFunControlPipeline, WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from videox_fun.utils import FlowDPMSolverMultistepScheduler
from third_party.Enhance_A_Video.enhance_a_video import enable_enhance, set_num_frames, set_enhance_weight


class Wan_Fun_Inferencer(object):

    def __init__(self):
        # TODO(squirrelli): model init here.
        pass

    def predict(self, **kwargs):
        ring_degree = kwargs.get("ring_degree", 1)
        ulysses_degree = kwargs.get("ulysses_degree", 1)
        if ulysses_degree > 1 or ring_degree > 1:
            # NOTE(squirrelli): since torch.mp.spawn do not support kwargs, we need use partial func.
            manager = mp.Manager()
            queue = manager.Queue()
            partial_predict = partial(self._predict, **kwargs)
            mp.spawn(partial_predict, args=(queue,),
                     nprocs=ring_degree*ulysses_degree)
            sample = queue.get()  # (bs, c, f, h, w)
            sample_array = sample.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            sample_array = (sample_array * 255).astype("uint8")
            return sample_array
        else:
            sample = self._predict(0, None, **kwargs)
            return sample_array

    def _predict(
        self,
        rank,
        queue,
        # Hardware and memory settings
        GPU_memory_mode: str = "model_cpu_offload",
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        # Cache configuration
        enable_teacache: bool = False,
        teacache_threshold: float = 0.10,
        num_skip_start_steps: int = 5,
        teacache_offload: bool = False,
        # Model configuration
        config_path: str = "config/wan2.1/wan_civitai.yaml",
        model_name: str = "models/Diffusion_Transformer/Wan2.1-Fun-14B-Control",
        sampler_name: str = "Flow",
        # Model components
        transformer_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        # Output specifications
        sample_size: List[int] = [720, 1280],
        video_length: int = 49,
        fps: int = 24,
        video_context: List[str] = None,
        # Precision settings
        weight_dtype: torch.dtype = torch.bfloat16,
        # Control inputs
        control_video: Optional[str] = None,
        control_strength: float = 1.0,
        # Reference inputs
        ref_video: Optional[str] = None,
        v2v_strength: float = 0.9,
        ref_image: Optional[str] = None,
        ref_image_strength: float = 1.0,
        clip_image: Optional[str] = None,
        # Text prompts
        prompt: str = "",
        negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        # Generation parameters
        guidance_scale: float = 6.0,
        seed: int = 258738739797534,
        num_inference_steps: int = 20,
        # LoRA settings
        lora_weight: int = 0,
        # Enhance a video
        enhance_a_video: bool = False,
        enhance_weight: float = 4.0,
        remove_ref_image: bool = False
    ):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(ulysses_degree * ring_degree)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        

        device = set_multi_gpus_devices(ulysses_degree, ring_degree)
        config = OmegaConf.load(config_path)

        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get(
                'transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(
                config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        if transformer_path is not None:
            print(f"From checkpoint: {transformer_path}")
            if transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(transformer_path)
            else:
                state_dict = torch.load(transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config['vae_kwargs'].get(
                'vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)

        if vae_path is not None:
            print(f"From checkpoint: {vae_path}")
            if vae_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(vae_path)
            else:
                state_dict = torch.load(vae_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = vae.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        # Get Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get(
                'tokenizer_subpath', 'tokenizer')),
        )

        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get(
                'text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(
                config['text_encoder_kwargs']),
        ).to(weight_dtype)
        text_encoder = text_encoder.eval()

        # Get Clip Image Encoder
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_name, config['image_encoder_kwargs'].get(
                'image_encoder_subpath', 'image_encoder')),
        ).to(weight_dtype)
        clip_image_encoder = clip_image_encoder.eval()

        # Get Scheduler
        Choosen_Scheduler = scheduler_dict = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "unipc": FlowDPMSolverMultistepScheduler
        }[sampler_name]
        scheduler = Choosen_Scheduler(
            **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        if sampler_name == "unipc":
            scheduler.set_timesteps(num_inference_steps, device=device)

        # Get Pipeline
        pipeline = WanFunControlPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder
        )
        if ulysses_degree > 1 or ring_degree > 1:
            transformer.enable_multi_gpus_inference()

        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(
                transformer, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(
                transformer, exclude_module_name=["modulation",])
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        coefficients = get_teacache_coefficients(
            model_name) if enable_teacache else None
        if coefficients is not None:
            print(
                f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )

        generator = torch.Generator(device=device).manual_seed(seed)

        if lora_path is not None:
            pipeline = merge_lora(pipeline, lora_path,
                                  lora_weight, device=device)

        with torch.no_grad():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio *
                               vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (
                video_length - 1) // vae.config.temporal_compression_ratio + 1

        # ### prepare video translation ref latents start
        input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(
            control_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=ref_image, clip_image=clip_image, video_context=video_context)
        ref_video, ref_video_mask, _, _ = get_video_to_video_latent(
            ref_video, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None, video_context=video_context)

        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=input_video,
            ref_video=ref_video,
            ref_image=ref_image,
            clip_image=clip_image,
            ref_image_strength=ref_image_strength,
            v2v_strength=v2v_strength,
            control_strength=control_strength,
            remove_ref_image=remove_ref_image,
            enhance_a_video=enhance_a_video,
            enhance_weight=enhance_weight
        ).videos

        if lora_path is not None:
            pipeline = unmerge_lora(
                pipeline, lora_path, lora_weight, device=device)

        if ulysses_degree * ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                queue.put(sample)
                return sample
        else:
            return sample
