import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel, AutoencoderKLWan3_8,
                              WanT5EncoderModel, CogOmniControlWanModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import CogOmniControlPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fm_solvers_lcm import FlowMatchLCMScheduler

from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info

def setup_distributed():
    import torch.distributed as dist
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            print(f"[Rank {rank}/{world_size}] Process group initialized.")
            
        return local_rank
    else:
        print("Not running in distributed mode.")
        return 0 # 默认回退到单卡模式

if __name__ == "__main__":
    GPU_memory_mode     = "sequential_cpu_offload"
    # Multi GPUs config
    ulysses_degree      = 8
    # Use FSDP to save more GPU memory in multi gpus.
    fsdp_dit            = False
    fsdp_text_encoder   = False

    # TeaCache config
    enable_teacache     = False
    # Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    # # --------------------------------------------------------------------------------------------------- #
    # | Model Name          | threshold | Model Name          | threshold |
    # | Wan2.2-T2V-A14B     | 0.10~0.15 | Wan2.2-I2V-A14B     | 0.15~0.20 |
    # | Wan2.2-Fun-A14B-*   | 0.15~0.20 |
    # # --------------------------------------------------------------------------------------------------- #
    teacache_threshold  = 0.0
    # The number of steps to skip TeaCache at the beginning of the inference process, which can
    # reduce the impact of TeaCache on generated video quality.
    num_skip_start_steps = 0
    # Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
    teacache_offload    = False

    # Skip some cfg steps in inference
    # Recommended to be set between 0.00 and 0.25
    cfg_skip_ratio      = None

    # Riflex config
    enable_riflex       = False
    # Index of intrinsic frequency
    riflex_k            = 6

    # Config and model path
    config_path         = "config/wan2.2/wan_cogomni_14b_civitai.yaml"
    model_name          = "models/Diffusion_Transformer/Wan2.2-T2V-A14B"

    # Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
    # sampler_name        = "lcm"
    # sampler_name        = "Flow_Unipc"
    # sampler_name        = "Flow_DPM++"
    sampler_name        = "Flow"
    # [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
    # Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
    shift               = 5

    # Load pretrained model if need
    # The transformer_path is used for low noise model, the transformer_high_path is used for high noise model.
    transformer_path        = None
    transformer_high_path   = None
    vae_path                = None

    # Input control video and reference images
    control_video_path = "control.mp4"
    ref_image_paths = [
        "ref.jpg",
    ]

    # Load lora model if need
    # The lora_path is used for low noise model, the lora_high_path is used for high noise model.
    lora_paths               = [
        # distill lora for low noise model
        "/group/40063/squirrelli/projects/VideoX-Fun/models/loras/Wan2.1-14B-T2V/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
        # cogomni control lora for low noise model
        "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/low_cogomni_icl_base_lora_layout/2026.02.18-22.18.36/checkpoint-6600-lora/Wan2_2-CogOmni_low_lora_14B.safetensors",

        # "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/low_cogomni_icl_base_lora/2026.02.14-22.55.55/checkpoint-2500-lora/Wan2_2-CogOmni_low_lora_14B.safetensors",
        # "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/low_cogomni_icl_base_lora_layout/2026.02.18-22.18.36/checkpoint-4000-lora/Wan2_2-CogOmni_low_lora_14B.safetensors",
    ]
    lora_high_paths          = [
        # distill lora for high noise model
        "/group/40063/squirrelli/projects/VideoX-Fun/models/loras/Wan2.1-14B-T2V/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
        # cogomni control lora for high noise model
        "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/high_cogomni_icl_base_lora_layout/2026.02.18-22.22.40/checkpoint-7000-lora/Wan2_2-CogOmni_high_lora_14B.safetensors",

        # "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/high_cogomni_icl_base_lora/2026.02.13-13.47.39/checkpoint-2500-lora/Wan2_2-CogOmni_high_lora_14B.safetensors",
        # "/group/40063/squirrelli/projects/VideoX-Fun/exps/20260212_CogOmniControl_train/output_dir/high_cogomni_icl_base_lora_layout/2026.02.18-22.22.40/checkpoint-4000-lora/Wan2_2-CogOmni_high_lora_14B.safetensors",
    ]
    lora_weights         = [1.0, 1.0]
    lora_high_weights    = [1.0, 1.0]

    # Other params
    # sample_size         = [720, 1280]
    sample_size         = [640, 640]
    video_length        = 33
    fps                 = 24

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype            = torch.bfloat16
    # 使用更长的neg prompt如"模糊,突变,变形,失真,画面暗,文本字幕,画面固定,连环画,漫画,线稿,没有主体。",可以增加稳定性
    # 在neg prompt中添加"安静,固定"等词语可以增加动态性。
    prompt              = "一张极具震撼力的特写镜头,画面中央是一位体型极其肥胖、面目阴森、圆脸肉粉色皮肤的老年男性反派,他满脸横肉,下巴层叠,皮肤纹理清晰可见,布满皱纹与斑点。他有一头浓密的银白色长发,发丝细长且质感分明,如同被超自然力量或狂风吹散般向四周张开飞舞。他长着浓密的黑色眉毛,额头正中有一个鲜红色的螺旋图案竖向印记,双眼深陷在厚重的眉弓之下,闪烁着深红色、邪恶且强烈的红光。他的表情阴沉、严肃而平静,充满威慑力,嘴巴微微张开,露出一排小而白的牙齿,似乎在缓慢地说话,头部随之轻微晃动。他穿着墨绿色的厚重质感长袍,脖子上戴着一个极其巨大、厚重且华丽的金色圆环项圈,上面雕刻着精美的编织纹路、漩涡图案和球形装饰,并连接着沉重的金链。背景昏暗模糊,隐约可见垂下的藤蔓和古旧木质结构,顶光照明强化了面部阴影与轮廓,光影深邃。摄像机从对他面部的稳定特写开始,镜头缓慢向后拉远。整体呈现出一种压抑、邪恶且充满力量感的黑暗奇幻电影质感,细节极其丰富,超写实3D渲染风格。"
    negative_prompt     = "模糊,突变,变形,失真,画面暗,文本字幕,画面固定,连环画,漫画,线稿,没有主体, 安静, 固定"
    guidance_scale      = 2
    seed                = 43
    num_inference_steps = 8
    save_path           = "samples/cogomni_control"

    if ulysses_degree > 1:
        setup_distributed()
        initialize_sequence_parallel_state(ulysses_degree)
    device = torch.device(f"cuda:{nccl_info.rank_within_group}")

    config = OmegaConf.load(config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

    transformer = CogOmniControlWanModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    transformer_2 = CogOmniControlWanModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
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

    # Get Vae
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
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
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Scheduler
    Chosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        "lcm": FlowMatchLCMScheduler
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = CogOmniControlPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if ulysses_degree > 1:
        from functools import partial
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

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

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    if cfg_skip_ratio is not None:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_paths is not None:
        for lora_path, lora_weight in zip(lora_paths, lora_weights):
            pipeline = merge_lora(pipeline, lora_path, lora_weight,
                                  device=device, dtype=weight_dtype)
    
    if lora_high_paths is not None:
        for lora_high_path, lora_high_weight in zip(lora_high_paths, lora_high_weights):
            pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight,
                                  device=device, dtype=weight_dtype,
                                  sub_transformer_name="transformer_2")    

    with torch.no_grad():
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
            pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

        sample = pipeline(
            prompt, 
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            control_video = control_video_path,
            ref_images = ref_image_paths,
            boundary = boundary,
            shift = shift,
        ).videos

    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        if video_length == 1:
            video_path = os.path.join(save_path, prefix + ".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(save_path, prefix + ".mp4")
            save_videos_grid(sample, video_path, fps=fps)

    if ulysses_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()
