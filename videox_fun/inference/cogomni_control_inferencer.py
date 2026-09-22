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
import torch.distributed as dist

from videox_fun.dist import shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer,
                               WanT5EncoderModel, CogOmniControlWanModel_Connector)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import CogOmniControlConnectorPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fm_solvers_lcm import FlowMatchLCMScheduler

from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen2Tokenizer, AutoProcessor
except:
    print("Qwen3VLForConditionalGeneration not found")

def load_connector(pipeline, connector_paths, device, dtype):
    print("Loading Connector from", connector_paths)
    
    sequential_cpu_offload_flag = False
    if pipeline.transformer.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True
        offload_device = pipeline._offload_device

    # transformer = transformer.cpu()
    from safetensors.torch import load_file
    state_dict = load_file(connector_paths[0])
    new_state_dict = {
        k.replace("llm_fea_connector.", ""): v
        for k, v in state_dict.items()
        if k.startswith("llm_fea_connector.")
    }
    pipeline.transformer.llm_fea_connector.load_state_dict(
        new_state_dict,
        assign=True,
        )
    # transformer = transformer.to(device=device, dtype=dtype)
    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=offload_device)

    return pipeline

def load_high_connector(pipeline, connector_paths, device, dtype):
    print("Loading Connector from", connector_paths)
    
    sequential_cpu_offload_flag = False
    if pipeline.transformer_2.device == torch.device(type="meta"):
        pipeline.remove_all_hooks()
        sequential_cpu_offload_flag = True
        offload_device = pipeline._offload_device

    from safetensors.torch import load_file
    state_dict = load_file(connector_paths[0])
    new_state_dict = {
        k.replace("llm_fea_connector.", ""): v
        for k, v in state_dict.items()
        if k.startswith("llm_fea_connector.")
    }
    pipeline.transformer_2.llm_fea_connector.load_state_dict(
        new_state_dict,
        assign=True,
        )
    if sequential_cpu_offload_flag:
        pipeline.enable_sequential_cpu_offload(device=offload_device)

    return pipeline

# Global cache for pipeline to keep it resident in worker processes
_PIPELINE_CACHE = {}


def _worker_loop_fn(rank, task_queue, result_queue):
    """Global worker function that keeps running and processes tasks."""
    inferencer = CogOmniControl_Connector_Inferencer()
    while True:
        task = task_queue.get()
        if task is None:
            break
        inferencer._predict(rank, result_queue, **task)


class CogOmniControl_Connector_Inferencer(object):

    def __init__(self):
        """Initialize the inferencer. Pipeline will be lazily initialized in worker processes."""
        self._processes = None
        self._task_queues = None
        self._result_queue = None

    def predict(self, **kwargs):
        """Main prediction interface that handles both single and multi-process inference."""
        ulysses_degree = kwargs.get("ulysses_degree", 1)
        if ulysses_degree > 1:
            # Multi-process mode: use persistent worker processes
            if self._processes is None:
                self._start_workers(ulysses_degree)
            
            # Broadcast task to all workers (each worker needs the same task info for collaboration)
            for task_queue in self._task_queues:
                task_queue.put(kwargs)
            
            # Get result from rank 0
            sample = self._result_queue.get()
            sample_array = sample.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            sample_array = (sample_array * 255).astype("uint8")
            return sample_array
        else:
            # Single process mode: initialize pipeline if needed
            sample = self._predict(0, None, **kwargs)
            sample_array = sample.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            sample_array = (sample_array * 255).astype("uint8")
            return sample_array

    def _start_workers(self, ulysses_degree):
        """Start persistent worker processes."""
        # Set spawn method for CUDA compatibility
        mp.set_start_method('spawn', force=True)
        
        self._task_queues = [mp.Queue() for _ in range(ulysses_degree)]
        self._result_queue = mp.Queue()
        self._processes = []
        for rank in range(ulysses_degree):
            p = mp.Process(target=_worker_loop_fn, args=(rank, self._task_queues[rank], self._result_queue))
            p.start()
            self._processes.append(p)

    def _shutdown_workers(self):
        """Shutdown all worker processes."""
        if self._processes is None:
            return
        
        # Send shutdown signal to all workers
        for task_queue in self._task_queues:
            task_queue.put(None)
        
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        self._processes = None
        self._task_queues = None
        self._result_queue = None

    def __del__(self):
        """Cleanup when the inferencer is destroyed."""
        self._shutdown_workers()

    def _predict(
        self,
        rank,
        queue,
        # Hardware and memory settings
        GPU_memory_mode: str = "sequential_cpu_offload",
        ulysses_degree: int = 1,
        # Use FSDP to save more GPU memory in multi gpus
        fsdp_dit: bool = False,
        fsdp_text_encoder: bool = False,
        # Cache configuration
        enable_teacache: bool = False,
        teacache_threshold: float = 0.10,
        num_skip_start_steps: int = 0,
        teacache_offload: bool = False,
        # Skip some cfg steps in inference
        cfg_skip_ratio: Optional[float] = None,
        # Riflex config
        enable_riflex: bool = False,
        riflex_k: int = 6,
        # Model configuration
        config_path: str = "config/wan2.2/wan_cogomni_14b_civitai.yaml",
        model_name: str = "models/Diffusion_Transformer/Wan2.2-T2V-A14B",
        sampler_name: str = "Flow",
        shift: int = 5,
        # Model components
        transformer_path: Optional[str] = None,
        transformer_high_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        llm_lora_path: Optional[str] = None,
        lora_paths: Optional[List[str]] = None,
        lora_high_paths: Optional[List[str]] = None,
        connector_paths: Optional[List[str]] = None,
        connector_high_paths: Optional[List[str]] = None,
        lora_weights: Optional[List[float]] = None,
        lora_high_weights: Optional[List[float]] = None,
        # Output specifications
        sample_size: List[int] = [720, 1280],
        video_length: int = 49,
        fps: int = 24,
        # Precision settings
        weight_dtype: torch.dtype = torch.bfloat16,
        # Control inputs
        control_video: Optional[str] = None,
        # Reference inputs
        ref_images: Optional[List[str]] = None,
        # Text prompts
        prompt: str = "",
        negative_prompt: str = "模糊,突变,变形,失真,画面暗,文本字幕,画面固定,连环画,漫画,线稿,没有主体",
        # Generation parameters
        guidance_scale: float = 2.0,
        seed: int = 43,
        num_inference_steps: int = 8,
        # Additional parameters
        boundary: Optional[float] = None,
        save_path: Optional[str] = None,
    ):
        """Run inference. Pipeline will be initialized once and reused for subsequent calls."""

        # Setup distributed environment
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(ulysses_degree)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "28500"

        torch.cuda.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        # Create a cache key based on model configuration
        cache_key = f"pipeline_{rank}"
        
        # Check if pipeline is already initialized in this process
        if cache_key not in _PIPELINE_CACHE:
            print(f"[Rank {rank}] Initializing pipeline (first time)...")
            pipeline, device, config, vae = self._initialize_pipeline(
                rank=rank,
                ulysses_degree=ulysses_degree,
                GPU_memory_mode=GPU_memory_mode,
                fsdp_dit=fsdp_dit,
                fsdp_text_encoder=fsdp_text_encoder,
                config_path=config_path,
                model_name=model_name,
                sampler_name=sampler_name,
                transformer_path=transformer_path,
                transformer_high_path=transformer_high_path,
                vae_path=vae_path,
                weight_dtype=weight_dtype,
                llm_lora_path=llm_lora_path
            )
            _PIPELINE_CACHE[cache_key] = {
                "pipeline": pipeline,
                "device": device,
                "config": config,
                "vae": vae,
                "weight_dtype": weight_dtype,
                "lora_low_applied": False,  # Track if low LoRA has been applied
                "lora_high_applied": False,  # Track if high LoRA has been applied
            }
            print(f"[Rank {rank}] Pipeline initialized and cached")
        else:
            print(f"[Rank {rank}] Reusing cached pipeline")
        
        # Get cached pipeline and related objects
        cached = _PIPELINE_CACHE[cache_key]
        pipeline = cached["pipeline"]
        device = cached["device"]
        config = cached["config"]
        vae = cached["vae"]
        weight_dtype = cached["weight_dtype"]
        lora_low_applied = cached["lora_low_applied"]
        lora_high_applied = cached["lora_high_applied"]
        
        # Get boundary from config if not provided
        if boundary is None:
            boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)
        
        # Setup TeaCache if enabled
        if enable_teacache:
            coefficients = get_teacache_coefficients(model_name)
            if coefficients is not None:
                print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
                pipeline.transformer.enable_teacache(
                    coefficients, num_inference_steps, teacache_threshold, 
                    num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
                )
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

        # Setup cfg_skip if enabled
        if cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        # Setup generator
        generator = torch.Generator(device=device).manual_seed(seed)

        # Apply LoRA if provided and not already applied
        if lora_paths is not None and not lora_low_applied:
            if lora_weights is None:
                lora_weights = [1.0] * len(lora_paths)
            for lora_path, lora_weight in zip(lora_paths, lora_weights):
                print(f"Merging LoRA: {lora_path} with weight {lora_weight}")
                pipeline = merge_lora(pipeline, lora_path, lora_weight,
                                      device=device, dtype=weight_dtype)
            cached["lora_low_applied"] = True
        
        if lora_high_paths is not None and not lora_high_applied:
            if lora_high_weights is None:
                lora_high_weights = [1.0] * len(lora_high_paths)
            for lora_high_path, lora_high_weight in zip(lora_high_paths, lora_high_weights):
                print(f"Merging high LoRA: {lora_high_path} with weight {lora_high_weight}")
                pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight,
                                      device=device, dtype=weight_dtype,
                                      sub_transformer_name="transformer_2")
            cached["lora_high_applied"] = True

        if connector_paths is not None and not lora_low_applied:
            pipeline = load_connector(
                pipeline, connector_paths, device=device, dtype=weight_dtype
            )

        if connector_high_paths is not None and not lora_high_applied:
            pipeline = load_high_connector(
                pipeline, connector_high_paths, device=device, dtype=weight_dtype,
                )

        # Adjust video_length for VAE compression
        with torch.no_grad():
            video_length = int((video_length - 1) // vae.config.temporal_compression_ratio *
                               vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

            # Enable riflex if needed
            if enable_riflex:
                pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
                pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

        # Run inference
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=control_video,
            ref_images=ref_images,
            boundary=boundary,
            shift=shift,
        ).videos

        # Update cache with pipeline (LoRA remains merged for next inference)
        _PIPELINE_CACHE[cache_key]["pipeline"] = pipeline

        # Save results if save_path is provided
        if save_path is not None:
            if ulysses_degree > 1:
                if dist.get_rank() == 0:
                    self._save_results(sample, save_path, fps)
            else:
                self._save_results(sample, save_path, fps)

        # Return result
        if ulysses_degree > 1:
            if dist.get_rank() == 0:
                queue.put(sample)
                return sample
        else:
            return sample

    def _initialize_pipeline(
        self,
        rank,
        ulysses_degree,
        GPU_memory_mode,
        fsdp_dit,
        fsdp_text_encoder,
        config_path,
        model_name,
        sampler_name,
        transformer_path,
        transformer_high_path,
        vae_path,
        weight_dtype,
        llm_lora_path=None,
    ):
        """Initialize pipeline in the current process. This is called once per process."""

        if ulysses_degree > 1:
            initialize_sequence_parallel_state(ulysses_degree)
        device = torch.device(f"cuda:{nccl_info.rank_within_group}")
        config = OmegaConf.load(config_path)

        # llm = text_encoder
        llm = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Thinking",
                device_map="cpu",
            ).to(weight_dtype)
        llm = llm.eval()

        
        from peft import PeftModel
        if llm_lora_path is not None:
            llm = PeftModel.from_pretrained(
                llm, 
                llm_lora_path,
            )

        meta_params = [n for n, p in llm.named_parameters() if p.is_meta]
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Thinking",
        )



        # Load main transformer (low noise model)
        transformer = CogOmniControlWanModel_Connector.from_pretrained(
            os.path.join(model_name, config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=False,
        ).to(weight_dtype)

        

        # Load transformer_2 (high noise model)
        transformer_2 = CogOmniControlWanModel_Connector.from_pretrained(
            os.path.join(model_name, config["transformer_additional_kwargs"].get(
                "transformer_high_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=False,
        ).to(weight_dtype)

        

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

        if transformer_high_path is not None:
            print(f"From checkpoint: {transformer_high_path}")
            if transformer_high_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open
                state_dict = load_file(transformer_high_path)
            else:
                state_dict = torch.load(transformer_high_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            m, u = transformer_2.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        meta_params = [n for n, p in transformer.named_parameters() if p.is_meta]
        print(len(meta_params), "meta params")
        print(meta_params[:10])

        meta_params = [n for n, p in transformer_2.named_parameters() if p.is_meta]
        print(len(meta_params), "meta params")
        print(meta_params[:10])

        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(model_name, config["vae_kwargs"].get(
                "vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
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

        meta_params = [n for n, p in vae.named_parameters() if p.is_meta]
        print(len(meta_params), "meta params")
        print(meta_params[:10])

        # Get Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get(
                "tokenizer_subpath", "tokenizer")),
        )

        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config["text_encoder_kwargs"].get(
                "text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(
                config["text_encoder_kwargs"]),
        ).to(weight_dtype)
        text_encoder = text_encoder.eval()
        

        # Get Scheduler
        Chosen_Scheduler = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
            "lcm": FlowMatchLCMScheduler
        }[sampler_name]
        if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
            config["scheduler_kwargs"]["shift"] = 1
        scheduler = Chosen_Scheduler(
            **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
        )

        # Get Pipeline
        pipeline = CogOmniControlConnectorPipeline(
            transformer=transformer,
            transformer_2=transformer_2,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            llm=llm,
            processor=processor,
        )
        
        # Apply FSDP if needed
        if ulysses_degree > 1:
            if fsdp_dit:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.transformer = shard_fn(pipeline.transformer)
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
                print("Add FSDP DIT")
            if fsdp_text_encoder:
                shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
                pipeline.text_encoder = shard_fn(pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        # Apply memory optimization mode
        if GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation",], device=device)
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        return pipeline, device, config, vae

    def _save_results(self, sample, save_path, fps):
        """Save generated video or image to disk.
        
        Args:
            sample: Generated video tensor with shape [bs, c, f, h, w]
            save_path: Directory path to save the output
            fps: Frames per second for video output
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        
        # Get video_length from sample shape: [bs, c, f, h, w]
        video_length = sample.shape[2]
        
        if video_length == 1:
            video_path = os.path.join(save_path, prefix + ".png")
            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
            print(f"Image saved to {video_path}")
        else:
            video_path = os.path.join(save_path, prefix + ".mp4")
            save_videos_grid(sample, video_path, fps=fps)
            print(f"Video saved to {video_path}")
