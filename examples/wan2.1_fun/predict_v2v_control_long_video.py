"""
1. 长视频读取并切分
For
    2. 给每个视频片段做left padding
    3. 给每个视频片段生成control
    4. 转绘视频片段
End For
5. 合并视频
"""
import os
import cv2
import torch
import gc
import numpy as np

from videox_fun.pipeline.context import get_context_scheduler
from videox_fun.inference import Wan_Fun_Controller
from videox_fun.utils.utils import save_video


def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    return video_frames


if __name__ == "__main__":
    context_scheduler_name = "sequential"
    num_inference_steps = 20
    context_size = 81
    context_overlap = 1
    context_stride = 1
    context_closed_loop = False
    control_video = "exps/20250512_dldl_toon_shadow/data/videos/walk003_softedge_hed.mp4"
    control_strength = 1.0
    ref_video = "exps/20250512_dldl_toon_shadow/data/videos/walk003.mp4"
    v2v_strength = 0.8
    ref_image = "exps/20250512_dldl_toon_shadow/data/videos/walk003-f1-shadow.png"
    clip_image = None
    ref_image_strength = 1.0
    prompt = "anime-style, a character with long hair, black background."
    negative_prompt = "3D，立体的，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，胡子"
    sample_size = [720, 1280]
    # sample_size = [1088, 1920]
    save_path = "exps/20250512_dldl_toon_shadow/samples"
    fps = 24
    total_frame_num = 81  # -1代表全部视频
    # total_frame_num = 25

    keyframe_dict = {
        # 80: "exps/20250416_dldl_toon_shadow/data/videos/toon_style_softedge_hed_TS_Head_RMBG_F241-f80.png",
        # 96: "exps/20250416_dldl_toon_shadow/data/videos/toon_style_softedge_hed_TS_Head_RMBG_F241-f96.png",
        # 104: "exps/20250416_dldl_toon_shadow/data/videos/toon_style_softedge_hed_TS_Head_RMBG_F241-f104.png",
        # 184: "exps/20250416_dldl_toon_shadow/data/videos/toon_style_softedge_hed_TS_Head_RMBG_F241-f184.png"
        # 160: "exps/20250416_dldl_toon_shadow/data/videos/toon_style_softedge_hed_TS_Head_RMBG_F241-f160.png"
    }

    controller = Wan_Fun_Controller()

    # 1. 读取视频并切分
    context_scheduler = get_context_scheduler(context_scheduler_name)
    total_frame_num = None if total_frame_num == -1 else total_frame_num
    video_frames = read_video(ref_video)[:total_frame_num]
    frame_num = len(video_frames)
    video_contexts = list(
        context_scheduler(
            0,
            num_inference_steps,
            frame_num,
            context_size,
            context_overlap,
            context_stride,
            context_closed_loop,
        )
    )
   
    pred_video = np.zeros([frame_num, *sample_size, 3])
    frame_counter = np.zeros([frame_num, 1, 1, 1])

    for video_context in video_contexts:
        if video_context[0] in keyframe_dict:
            ref_image = keyframe_dict[video_context[0]]
        pred_video_context = controller.predict(
            ulysses_degree = 2, 
            ring_degree = 3,
            # ulysses_degree = 1, 
            # ring_degree = 1,
            video_context = video_context,
            video_length = len(video_context),
            num_inference_steps = num_inference_steps,
            # v2v params
            control_video = control_video,
            control_strength = control_strength,
            ref_video = ref_video,
            v2v_strength = v2v_strength,
            ref_image = ref_image,
            ref_image_strength = ref_image_strength,
            prompt = prompt,
            sample_size = sample_size,
            fps = fps,
            clip_image = clip_image,
            model_name = "exps/20250423_anime_fun_control_train/output_dir/anime_clark_250432_100k/2025.04.28-11.07.54/checkpoint-700-diffusers",
            enhance_a_video = True,
            remove_ref_image = True
        )
        # get next ref image
        ref_image = pred_video_context[-1]
        # accumulate pred_video_context
        pred_video[video_context] += pred_video_context
        frame_counter[video_context] += 1
    pred_video /= frame_counter
    # save video
    os.makedirs(save_path, exist_ok=True)
    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    video_path = os.path.join(save_path, prefix + ".mp4")
    save_video(pred_video, video_path, fps=fps)

    # TODO 
    # 1. 加入风格参考 clip_img, 结论:67视频风格力度控制不够

    # 2. vlm推10s用转绘后的视频control, CAN

    # 训练端
    # 3. framepack训练
