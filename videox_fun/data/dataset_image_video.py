
import ast
import csv
import gc
import io
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import reduce
from threading import Thread

import albumentations as A
import cv2
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from pprint import pprint
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from videox_fun.utils.utils import removesuffix

VIDEO_READER_TIMEOUT = 60

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

def process_reference_images(ref_img_paths, target_size, data_root=None):
    """
    Process multiple reference images: 
    1. Resize each image to a common scale while maintaining aspect ratio
    2. Horizontally concatenate them
    3. Resize the concatenated image to target size while maintaining aspect ratio
    4. Place on white canvas to match target size
    
    Args:
        ref_img_paths: List of paths to reference images
        target_size: Tuple of (height, width) for target canvas size
        data_root: Optional data root directory
    
    Returns:
        Processed reference image (h, w, 3) resized to target_size
    """
    if not isinstance(ref_img_paths, list):
        ref_img_paths = [ref_img_paths]

    # Determine common scale based on target size and number of images
    canvas_height, canvas_width = target_size
    num_images = len(ref_img_paths)
    
    # Calculate available width per image (accounting for potential spacing)
    available_width_per_img = canvas_width // num_images
    
    # Process each image to the common scale
    processed_imgs = []
    max_height = 0
    total_width = 0
    
    for ref_img_path in ref_img_paths:
        if not data_root:
            full_path = ref_img_path
        else:
            if not ref_img_path.startswith(data_root):
                full_path = os.path.join(data_root, ref_img_path.lstrip("/"))
            else:
                full_path = ref_img_path
            
        # Load image
        img = Image.open(full_path).convert("RGB")
        img = np.array(img)
        
        # Calculate scale to fit within available width while maintaining aspect ratio
        orig_height, orig_width = img.shape[:2]
        scale = min(available_width_per_img / orig_width, canvas_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized_img = cv2.resize(img, (new_width, new_height))
        processed_imgs.append(resized_img)
        
        # Update dimensions for concatenation
        max_height = max(max_height, new_height)
        total_width += new_width
    
    # Create canvas for concatenated image
    concatenated_img = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place images on canvas
    x_offset = 0
    for img in processed_imgs:
        h, w = img.shape[:2]
        # Center vertically
        y_offset = (max_height - h) // 2
        concatenated_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
        x_offset += w
    
    # Resize the concatenated image to target size while maintaining aspect ratio
    ref_height, ref_width = concatenated_img.shape[:2]
    
    # Calculate scale to fit within target size
    scale = min(canvas_height / ref_height, canvas_width / ref_width)
    new_height = int(ref_height * scale)
    new_width = int(ref_width * scale)
    resized_image = cv2.resize(concatenated_img, (new_width, new_height))
    
    # Create white canvas for final output
    white_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Center the resized image on the canvas
    top = (canvas_height - new_height) // 2
    left = (canvas_width - new_width) // 2
    white_canvas[top:top + new_height, left:left + new_width] = resized_image
    
    return white_canvas

def starmap(pool, func, iterable):
    return pool.map(lambda p: func(*p), iterable)

def _dilate_mask(mask, dilation_size=50):
    # 二值化处理
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 确保膨胀大小是正奇数
    if dilation_size <= 0:
        raise ValueError("Dilation size must be a positive integer")
    if dilation_size % 2 == 0:
        dilation_size += 1  # 自动转换为下一个奇数
    
    # 创建圆形结构元素（核）
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_ELLIPSE, 
        ksize=(dilation_size, dilation_size)
    )
    
    dilated_mask = cv2.dilate(
        src=mask, 
        kernel=kernel,
        iterations=1
    )
    return np.where(dilated_mask > 127, 255, 0).astype(np.uint8)

def dilate_mask(mask_pixel_values, min_dilation_size=10, max_dilation_size=50, proc_num=8):
    mask_pixel_values = mask_pixel_values.squeeze(-1)
    dilation_size = random.randint(min_dilation_size, max_dilation_size)
    with ThreadPoolExecutor(max_workers=proc_num) as pool:
        dilated_masks = starmap(pool, _dilate_mask, [(mask_pixel_values[i], dilation_size) for i in range(len(mask_pixel_values))])
        dilated_masks = list(dilated_masks)
    mask_pixel_values = np.array(dilated_masks)
    mask_pixel_values = np.expand_dims(mask_pixel_values, axis=-1)
    return mask_pixel_values

# NOTE(squirrelli): The case of multiple objects is not considered for now,
# because the single object feature of the mask is guaranteed by data processing
def _rect_mask(mask):
    # 二值化处理
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓 - 使用近似方法减少点数
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    rected_mask = np.zeros_like(mask)
    rected_mask[y:y+h, x:x+w] = 255
    return rected_mask

def rect_mask(mask_pixel_values, proc_num=8):
    mask_pixel_values = mask_pixel_values.squeeze(-1)
    with ThreadPoolExecutor(max_workers=proc_num) as pool:
        rected_masks = starmap(pool, _rect_mask, [(mask_pixel_values[i],) for i in range(len(mask_pixel_values))])
        rected_masks = list(rected_masks)
    mask_pixel_values = np.array(rected_masks)
    mask_pixel_values = np.expand_dims(mask_pixel_values, axis=-1)
    return mask_pixel_values


def sample_masked_frame_ids(pixel_values):
    """
    pixel_values: (F, H, W, C)
    return: masked_frame_ids, list of frame indices that should be masked
    策略比例：
        不 mask
        stride mask
        完全随机 mask
    """
    F = pixel_values.shape[0]
    mode_prob = random.random()

    # =============================
    # Mode 1: 不 mask
    # =============================
    if mode_prob < 0.5:
        return []

    # =============================
    # Mode 2: stride mask
    # =============================
    elif mode_prob < 0.8:
        stride = random.choice([2, 3, 4])

        # 保留的帧 = key frames
        keep_frame_ids = set(np.arange(0, F, stride).tolist())
        keep_frame_ids.add(0)
        keep_frame_ids.add(F - 1)

        # mask 掉其余帧
        masked_frame_ids = [i for i in range(F) if i not in keep_frame_ids]
        return masked_frame_ids

    # =============================
    # Mode 3: 完全随机 mask
    # =============================
    else:
        keep_prob = random.random()
        keep_mask = (np.random.rand(F) < keep_prob)

        masked_frame_ids = [i for i in range(F) if not keep_mask[i]]
        return masked_frame_ids


def mask_frames_by_ids(pixel_values, masked_frame_ids):
    """
    pixel_values: (F, H, W, C) numpy array
    masked_frame_ids: list[int] — 需要被 mask 替换的帧编号
    
    对 masked_frame_ids 中的帧：
        mean > 127  -> 替换为全白(255)
        mean <= 127 -> 替换为全黑(0)
    """
    if len(masked_frame_ids) == 0:
        return pixel_values

    pixel_values = pixel_values.copy()
    F, H, W, C = pixel_values.shape

    # 转成 numpy array 以便向量化索引
    masked_frame_ids = np.asarray(masked_frame_ids, dtype=np.int64)

    # 取出所有需要替换的帧 (N, H, W, C)
    frames = pixel_values[masked_frame_ids]

    # 一次性计算均值 (N,)
    means = frames.mean(axis=(1, 2, 3))

    # 判断哪些应该白/黑
    to_white = masked_frame_ids[means > 127]
    to_black = masked_frame_ids[means <= 127]

    # 批量替换
    if len(to_white) > 0:
        pixel_values[to_white] = 255
    if len(to_black) > 0:
        pixel_values[to_black] = 0

    return pixel_values


def sample_and_mask_frames(pixel_values):
    """
    参数:
        pixel_values: (F, H, W, C) numpy array 

    返回:
        masked_pixel_values: 处理后的 pixel_values
        masked_frame_ids: List[int], 被 mask 的帧 id
    """
    masked_frame_ids = sample_masked_frame_ids(pixel_values)
    masked_pixel_values = mask_frames_by_ids(pixel_values, masked_frame_ids)
    return masked_pixel_values, masked_frame_ids


def repeat_ref_frame_to_neighbors(pixel_values):
    """
    随机选一帧作为参考帧，将该帧前后随机 3~8 帧替换为该帧内容。

    规则：
        - 第一帧（index=0）永远不参与替换，也不会被选为参考帧
        - 参考帧本身不被替换
        - 前后替换帧数随机分配，总数在 3~8 之间

    参数:
        pixel_values: (F, H, W, C) numpy array

    返回:
        new_pixel_values: 处理后的 pixel_values，shape 不变
        ref_frame_id: int，被选中的参考帧索引
        replaced_frame_ids: List[int]，被替换的帧索引列表（不含第一帧）
    """

    F = pixel_values.shape[0]
    # 替换帧数为总帧数的 20%~50%，至少 1 帧
    min_replace = max(1, int(F * 0.1))
    max_replace = max(min_replace, int(F * 0.3))

    # 参考帧从 index=1 开始选，保证第一帧不被选为参考帧
    ref_frame_id = random.randint(1, F - 1)
    ref_frame = pixel_values[ref_frame_id]  # (H, W, C)

    # 随机决定前后各替换多少帧
    total_replace = random.randint(min_replace, max_replace)
    before_num = random.randint(0, total_replace)
    after_num = total_replace - before_num

    # 计算实际可替换的帧范围（不替换参考帧本身，且第一帧 index=0 不可替换）
    before_ids = list(range(max(1, ref_frame_id - before_num), ref_frame_id))
    after_ids = list(range(ref_frame_id + 1, min(F, ref_frame_id + after_num + 1)))
    replaced_frame_ids = before_ids + after_ids

    # 执行替换
    new_pixel_values = pixel_values.copy()
    for fid in replaced_frame_ids:
        new_pixel_values[fid] = ref_frame

    return new_pixel_values, ref_frame_id, replaced_frame_ids


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'video')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            image_repeat=0,
            video_repeat=0,
            text_drop_ratio=0.1,
            enable_bucket=False,
            video_length_drop_start=0.0, 
            video_length_drop_end=1.0,
            enable_inpaint=False,
            use_vae_cache=False,
            caption_mode=None,
        ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        elif ann_path.endswith('.jsonl'):
            with jsonlines.open(ann_path, 'r') as jsonl_reader:
                dataset = list(jsonl_reader)
        elif ann_path.endswith('.txt'):
            # support multi datasource
            data_list_file = ann_path
            dataset = list()
            with open(data_list_file, 'r') as f:
                data_anno_list = [i.strip().split(',') for i in f.readlines() if len(i.strip()) > 0]
                for dataset_name, dataset_meta_file in data_anno_list:
                    with jsonlines.open(dataset_meta_file, 'r') as jsonl_reader:
                        dataset.extend(list(jsonl_reader))
        else:
            raise ValueError(f"Unsupported meta file: {ann_path}.")
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            self.dataset.append(data)
            if data.get('type', 'video') != 'video':
                if image_repeat > 0:
                    for _ in range(image_repeat):
                        self.dataset.append(data)
            else:
                if video_repeat > 0:
                    for _ in range(video_repeat):
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
        self.use_vae_cache = use_vae_cache

        # caption mode
        if caption_mode:
            try:
                caption_mode = ast.literal_eval(caption_mode)
            except (SyntaxError, ValueError) as e:
                pass
        self.caption_mode = caption_mode

    def get_mode_text(self, data_info, default_mode="dense_caption"):
        """
        support like:
        (1). "dense_caption"
        (2). ["dense_caption", "caption"] 
        (3). [["dense_caption", 0.8], ["caption", 0.2]]
        (4). [["dense_caption", 0.7], ["caption+tags", 0.2], ["tags+caption", 0.1]]
        """
        modes = self.caption_mode
        if isinstance(modes, str):
            modes = [modes]
        if isinstance(modes[0], list):
            options, probs = zip(*modes)
        else:
            options = modes
            probs = [1.0 / len(modes)] * len(modes)
        chosen_modes = random.choices(options, probs)[0]
        chosen_modes = [x.strip() for x in chosen_modes.split("+")]
        anno = []
        for mode_idx, chosen_mode in enumerate(chosen_modes):
            if chosen_mode not in data_info or (not data_info[chosen_mode]):
                chosen_mode = default_mode
            chosen_mode_idx = random.randint(0, len(data_info[chosen_mode])-1)
            mode_anno = data_info[chosen_mode][chosen_mode_idx]["content"]
            mode_anno = mode_anno.strip()
            if mode_idx != len(chosen_modes) - 1:
                # mode_anno = mode_anno.removesuffix(".")
                mode_anno = removesuffix(mode_anno, ".")
            anno.append(mode_anno)
        anno = ", ".join(anno)

        return anno

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('type', 'video')=='video':
            if 'file_path' in data_info:
                video_id, text = data_info['file_path'], data_info['text']
            else:
                # video_id, text = data_info['path'], data_info['dense_caption'][0]['content']
                video_id = data_info["path"]
                if self.caption_mode:
                    text = self.get_mode_text(data_info)
                else:
                    choice = random.randint(0, len(data_info["dense_caption"])-1)
                    text = data_info['dense_caption'][choice]['content'] 

            if self.use_vae_cache:
                latents_path = data_info["latents"]
                if isinstance(latents_path, dict):
                    latents_path = latents_path["path"]

                if latents_path.endswith(".npz"):
                    latents = np.load(latents_path)["latents"]  # (b, c, f, h, w)
                elif latents_path.endswith(".safetensors"):
                    latents = load_file(latents_path)["latents"]
                else:
                    raise ValueError
                latents = torch.tensor(latents)
                sample_n_latent = (self.video_sample_n_frames - 1) // 4 + 1
                # For vae_cache, start directly from the first frame
                start_idx = 0
                latents = latents[:, :, start_idx: start_idx+sample_n_latent, ...]
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
                return latents, text, "video"

            if not self.data_root:
                video_dir = video_id
            else:
                if not video_id.startswith(self.data_root):
                    video_dir = os.path.join(self.data_root, video_id.lstrip("/"))
                else:
                    video_dir = video_id

            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            return pixel_values, text, 'video'
        else:
            if 'file_path' in data_info:
                image_path, text = data_info['file_path'], data_info['text']
            else:
                image_path = data_info["path"]
                if self.caption_mode:
                    text = self.get_mode_text(data_info)
                else:
                    choice = random.randint(0, len(data_info["dense_caption"])-1)
                    text = data_info['dense_caption'][choice]['content'] 
            
            if self.use_vae_cache:
                latents_path = data_info["latents"]
                if isinstance(latents_path, dict):
                    latents_path = latents_path["path"]

                if latents_path.endswith(".npz"):
                    latents = np.load(latents_path)["latents"]  # (b, c, f, h, w)
                elif latents_path.endswith(".safetensors"):
                    latents = load_file(latents_path)["latents"]
                else:
                    raise ValueError
                latents = torch.tensor(latents)
                sample_n_latent = (self.video_sample_n_frames - 1) // 4 + 1
                # For vae_cache, start directly from the first frame
                start_idx = 0
                latents = latents[:, :, start_idx: start_idx+sample_n_latent, ...]
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
                return latents, text, "image"

            if not self.data_root:
                if not image_path.startswith(self.data_root):
                    image_path = os.path.join(self.data_root, image_path.lstrip("/"))

            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample


class ImageVideoControlDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, 
            video_sample_stride=4,
            video_sample_n_frames=16,
            image_sample_size=512,
            image_repeat=0,
            video_repeat=0,
            text_drop_ratio=0.1,
            enable_bucket=False,
            video_length_drop_start=0.0, 
            video_length_drop_end=1.0,
            enable_inpaint=False,
            use_vae_cache=False,
            caption_mode=None,
            max_ref_num_per_subject=1,
            max_subject_num=1,
            dilated_masks_prob=0.3,
            rected_masks_prob=0.1,
            ref_rmbg_prob=0.5,
            sparse_control_mode=False,
            random_ref_frame_mode=False
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        dataset_meta_stats = {}
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
            dataset_meta_stats[ann_path] = len(dataset)
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
            dataset_meta_stats[ann_path] = len(dataset)
        elif ann_path.endswith('.jsonl'):
            with jsonlines.open(ann_path, 'r') as jsonl_reader:
                dataset = list(jsonl_reader)
            dataset_meta_stats[ann_path] = len(dataset)
        elif ann_path.endswith('.txt'):
            # support multi datasource
            data_list_file = ann_path
            dataset = list()
            with open(data_list_file, 'r') as f:
                data_anno_list = [line.strip().split(',') for line in f.readlines() if len(line.strip()) > 0]
                for data_meta_info in data_anno_list:
                    if len(data_meta_info) == 2:
                        dataset_name, dataset_meta_file = data_meta_info
                        dataset_repeat_num = 1
                    elif len(data_meta_info) == 3:
                        dataset_name, dataset_meta_file, dataset_repeat_num = data_meta_info
                    else:
                        raise NotImplementedError(f"Invalid line format: {data_meta_info}")

                    with jsonlines.open(dataset_meta_file, 'r') as jsonl_reader:
                        curr_dataset = list(jsonl_reader) * int(dataset_repeat_num)
                        dataset.extend(curr_dataset)
                        dataset_meta_stats[dataset_name] = len(curr_dataset)
        else:
            raise ValueError(f"Unsupported meta file: {ann_path}.")
    
        self.dataset_meta_stats = dataset_meta_stats

        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []

        for data in dataset:
            self.dataset.append(data)
            if data.get('type', 'video') != 'video':
                if image_repeat > 0:
                    for _ in range(image_repeat):
                        self.dataset.append(data)
            else:
                if video_repeat > 0:
                    for _ in range(video_repeat):
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
        self.use_vae_cache = use_vae_cache

        # caption mode
        if caption_mode:
            try:
                caption_mode = ast.literal_eval(caption_mode)
            except (SyntaxError, ValueError) as e:
                pass
        self.caption_mode = caption_mode

        # reference training
        self.max_ref_num_per_subjct = max_ref_num_per_subject
        self.max_subject_num = max_subject_num
        self.dilated_masks_prob = dilated_masks_prob
        self.rected_masks_prob = rected_masks_prob
        self.ref_rmbg_prob = ref_rmbg_prob
        self.sparse_control_mode = sparse_control_mode
        self.random_ref_frame_mode = random_ref_frame_mode

        # image augmentation
        self.geo_transform = A.Compose([
            A.HorizontalFlip(p=0.5)
        ])

        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.2,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=20,
                p=0.5
            ),
            # A.ChannelShuffle(p=0.2),
            # A.GaussNoise(var_limit=(10, 50), p=0.5),
            # A.Blur(blur_limit=(3, 7), p=0.3),
            # A.MotionBlur(blur_limit=(3, 7), p=0.3)
        ])
        
    def images_augment(self, images, skip_first_frame=False, aug_opt_names=["geo", "color"]):
        if images.size == 0:
            return images

        if isinstance(aug_opt_names, str):
            aug_opt_names = [aug_opt_names]

        augmented_images = []
        for i in range(len(images)):
            if skip_first_frame and i == 0:
                augmented_images.append(images[i])
                continue
            image = images[i]
            for aug_opt_name in aug_opt_names:
                aug_opt = getattr(self, f"{aug_opt_name}_transform")
                image = aug_opt(image=image)["image"]
            augmented_images.append(image)
        return np.array(augmented_images)

    def show_meta_stats(self):
        if hasattr(self, "dataset_meta_stats"):
            print("===================== Dataset Meta Stats Start ========================")
            pprint(self.dataset_meta_stats)
            print("===================== Dataset Meta Stats End ========================")

    def get_mode_text(self, data_info, default_mode="dense_caption"):
        """
        support like:
        (1). "dense_caption"
        (2). ["dense_caption", "caption"] 
        (3). [["dense_caption", 0.8], ["caption", 0.2]]
        (4). [["dense_caption", 0.7], ["caption+tags", 0.2], ["tags+caption", 0.1]]
        """
        modes = self.caption_mode
        if isinstance(modes, str):
            modes = [modes]
        if isinstance(modes[0], list):
            options, probs = zip(*modes)
        else:
            options = modes
            probs = [1.0 / len(modes)] * len(modes)
        chosen_modes = random.choices(options, probs)[0]
        chosen_modes = [x.strip() for x in chosen_modes.split("+")]
        anno = []
        for mode_idx, chosen_mode in enumerate(chosen_modes):
            if chosen_mode not in data_info or (not data_info[chosen_mode]):
                chosen_mode = default_mode
            chosen_mode_idx = random.randint(0, len(data_info[chosen_mode])-1)

            source_tag = data_info.get("source_tag") or data_info.get("data_source")
            if source_tag is not None and "online" in source_tag:
                chosen_mode_idx = random.randint(0, len(data_info["dense_caption"])-1)
                mode_anno = data_info["dense_caption"][chosen_mode_idx]["content"]
            else:
                mode_anno = data_info[chosen_mode][chosen_mode_idx]["content"]
            mode_anno = mode_anno.strip()
            if mode_idx != len(chosen_modes) - 1:
                # mode_anno = mode_anno.removesuffix(".")
                mode_anno = removesuffix(mode_anno, ".")
            anno.append(mode_anno)
        anno = ", ".join(anno)

        return anno

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        # Process caption
        if 'file_path' in data_info:
            video_id, text = data_info['file_path'], data_info['text']
        else:
            if self.caption_mode:
                text = self.get_mode_text(data_info)
            else:
                choice = random.randint(0, len(data_info["dense_caption"])-1)
                text = data_info['dense_caption'][choice]['content'] 

        # Use vae_cache
        if self.use_vae_cache:
            latents_path = data_info["latents"]
            if isinstance(latents_path, dict):
                latents_path = latents_path["path"]

            if latents_path.endswith(".npz"):
                latents = np.load(latents_path)["latents"]  # (b, c, f, h, w)
            elif latents_path.endswith(".safetensors"):
                latents = load_file(latents_path)["latents"]
            else:
                raise ValueError

            control_latents_path = data_info["control_latents"]
            if isinstance(control_latents_path, dict):
                control_latents_path = control_latents_path["path"]

            if control_latents_path.endswith(".npz"):
                control_latents = np.load(control_latents_path)
                if "latents" in control_latents:
                    control_latents = control_latents["latents"]
                else:
                    control_latents = control_latents["control_latents"]
            elif control_latents_path.endswith(".safetensors"):
                control_latents = load_file(control_latents_path)
                if "latents" in control_latents:
                    control_latents = control_latents["latents"]
                else:
                    control_latents = control_latents["control_latents"]
            else:
                raise ValueError
            latents = torch.tensor(latents)
            control_latents = torch.tensor(control_latents)
            sample_n_latent = (self.video_sample_n_frames - 1) // 4 + 1
            # For vae_cache, start directly from the first frame
            start_idx = 0
            latents = latents[:, :, start_idx: start_idx+sample_n_latent, ...]
            control_latents = control_latents[:, :, start_idx: start_idx+sample_n_latent, ...]

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''

            return latents, control_latents, text, data_info.get('type', 'video')

        # Do not use vae_cache
        data_type = data_info.get('type', 'video')
        data_type = data_type if data_type is not None else "video"
        if data_type == 'video':
            # Process video data
            if 'file_path' in data_info:
                video_id, text = data_info['file_path'], data_info['text']
            else:
                video_id = data_info["path"]
                if isinstance(video_id, dict):
                    video_id = video_id["path"]

            if not self.data_root:
                video_dir = video_id
            else:
                if not video_id.startswith(self.data_root):
                    video_dir = os.path.join(self.data_root, video_id.lstrip("/"))
                else:
                    video_dir = video_id

            # Process video pixel_values
            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
            # Process control_pixel_values
            control_video_id = data_info.get("control_file_path") or data_info.get("control_path")
            control_video_type = ""
            if control_video_id is not None:
                if isinstance(control_video_id, dict):
                    control_video_type = control_video_id.get("type", "")
                    control_video_id = control_video_id["path"]

                if not self.data_root:
                    control_video_id = control_video_id
                else:
                    if not control_video_id.startswith(self.data_root):
                        control_video_id = os.path.join(self.data_root, control_video_id.lstrip("/"))
                    else:
                        control_video_id = control_video_id

                with VideoReader_contextmanager(control_video_id, num_threads=2) as control_video_reader:
                    try:
                        sample_args = (control_video_reader, batch_index)
                        control_pixel_values = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )
                        resized_frames = []
                        for i in range(len(control_pixel_values)):
                            frame = control_pixel_values[i]
                            resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            resized_frames.append(resized_frame)
                        control_pixel_values = np.array(resized_frames)
                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from control video. Error is {e}.")
                    # aug layout control video
                    if control_video_type is not None and "layout" in control_video_type:
                        control_pixel_values = self.images_augment(control_pixel_values, aug_opt_names=["color"])

                    # # sparse control mode
                    # if self.sparse_control_mode:
                    #     control_pixel_values, masked_control_frame_ids = sample_and_mask_frames(control_pixel_values)
                    #     if len(masked_control_frame_ids) > 0:
                    #         control_video_type += "_sparse"


                    # 仅对线稿/边缘类 control type 启用，且有 30% 概率跳过
                    _REPEAT_CONTROL_TYPES = {
                        "mlsd", "scribble_hed", "scribble_hedsafe",
                        "scribble_pidinet", "scribble_pidisafe"
                    }
                    if (self.sparse_control_mode
                            and control_video_type in _REPEAT_CONTROL_TYPES
                            and random.random() < 0.7):
                        control_pixel_values, _, replaced_control_frame_ids = repeat_ref_frame_to_neighbors(
                            control_pixel_values
                        )
                        if len(replaced_control_frame_ids) > 0:
                            control_video_type += "_sparse"
                            

                    if not self.enable_bucket:
                        control_pixel_values = torch.from_numpy(control_pixel_values).permute(0, 3, 1, 2).contiguous()
                        control_pixel_values = control_pixel_values / 255.
                        control_pixel_values = self.video_transforms(control_pixel_values)
                        del control_video_reader
                    else:
                        control_pixel_values = control_pixel_values
            else:
                control_pixel_values = np.zeros_like(pixel_values)
                if random.random() > 0.5:
                    control_pixel_values += 255

            # Process subject_ref_infos
            all_subject_ref_imgs_infos = data_info.get("ref_img_infos")
            subject_ref_pixel_values = np.empty(0)
            mask_pixel_values = np.empty(0)

            source_tag = data_info.get("source_tag") or data_info.get("data_source")
            if all_subject_ref_imgs_infos:
                max_subject_num = min(len(all_subject_ref_imgs_infos), self.max_subject_num)
                sample_subject_num = random.randint(1, max_subject_num)
                if source_tag is not None and "online" in source_tag:
                    sample_subject_num = len(all_subject_ref_imgs_infos)
                    sample_subject_ref_imgs_infos = all_subject_ref_imgs_infos
                else:
                    sample_subject_ref_imgs_infos = random.sample(all_subject_ref_imgs_infos, k=sample_subject_num)
                # for multi subjects, save every subject info
                sample_subject_ref_pixel_values_list = []
                sample_mask_pixel_values_list = []
                sample_subject_caption_list = []
                for subject_ref_imgs_info in sample_subject_ref_imgs_infos:
                    subject_caption = subject_ref_imgs_info.get("subject_caption", text)
                    # subject ref image
                    # NOTE(squirrelli): for verification of ref_imgs in future.
                    subject_valid_ref_imgs = [ref_img for ref_img in subject_ref_imgs_info.get("ref_imgs", [])]
                    if subject_valid_ref_imgs:
                        # subject caption
                        sample_subject_caption_list.append(subject_caption)
                        # subject image
                        max_ref_num_per_subject = min(len(subject_valid_ref_imgs), self.max_ref_num_per_subjct)
                        sample_ref_num = random.randint(1, max_ref_num_per_subject)
                        if subject_ref_imgs_info.get("subject_type") == "vidu_ref":
                            sample_ref_num = len(subject_valid_ref_imgs)
                        if source_tag is not None and "online" in source_tag:
                            sample_ref_num = len(subject_valid_ref_imgs)
                            subject_valid_ref_imgs = subject_valid_ref_imgs
                        else:
                            subject_valid_ref_imgs = random.sample(subject_valid_ref_imgs, k=sample_ref_num)
                        subject_ref_img_paths = []
                        for subject_valid_ref_img in subject_valid_ref_imgs:
                            ref_img_path_key = "img_path"
                            if random.random() < self.ref_rmbg_prob:
                                ref_img_path_key = "masked_img_path"
                            subject_ref_img_path = subject_valid_ref_img.get(
                                ref_img_path_key, subject_valid_ref_img["img_path"]
                            )
                            subject_ref_img_paths.append(subject_ref_img_path)
                        # for multi ref_imgs
                        target_size = self.video_sample_size if not self.enable_bucket else pixel_values.shape[-3:-1]
                        subject_ref_pixel_values = [
                            process_reference_images(subject_ref_img_path, target_size, self.data_root)
                            for subject_ref_img_path in subject_ref_img_paths
                        ]
                        subject_ref_pixel_values = np.array(subject_ref_pixel_values)
                        sample_subject_ref_pixel_values_list.append(subject_ref_pixel_values)
                    # subject mask video
                    mask_video_id = subject_ref_imgs_info.get("subject_mask_video_path")
                    if mask_video_id is not None:
                        if not self.data_root:
                            mask_video_id = mask_video_id
                        else:
                            mask_video_id = os.path.join(self.data_root, mask_video_id.lstrip("/"))
                        
                        with VideoReader_contextmanager(mask_video_id, num_threads=2) as mask_video_reader:
                            try:
                                sample_args = (mask_video_reader, batch_index)
                                mask_pixel_values = func_timeout(
                                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                                )
                                resized_frames = []
                                for i in range(len(mask_pixel_values)):
                                    frame = mask_pixel_values[i]
                                    resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                    resized_frames.append(resized_frame)
                                mask_pixel_values = np.array(resized_frames)
                                mask_pixel_values = mask_pixel_values[:, :, :, -1:]
                                if random.random() < self.dilated_masks_prob:
                                    mask_pixel_values = dilate_mask(mask_pixel_values)
                                if random.random() < self.rected_masks_prob:
                                    mask_pixel_values = rect_mask(mask_pixel_values)
                            except FunctionTimedOut:
                                raise ValueError(f"Read {idx} timeout.")
                            except Exception as e:
                                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                            if not self.enable_bucket:
                                mask_pixel_values = torch.from_numpy(mask_pixel_values).permute(0, 3, 1, 2).contiguous()
                                mask_pixel_values = mask_pixel_values / 255.
                                mask_pixel_values = torch.where(mask_pixel_values > 0.5, 1.0, 0.0)
                                del mask_video_reader
                            else:
                                mask_pixel_values = mask_pixel_values

                            if not self.enable_bucket:
                                mask_pixel_values = self.mask_transforms(mask_pixel_values)
                            sample_mask_pixel_values_list.append(mask_pixel_values)
                
                # multi subject infos fusion
                if sample_subject_ref_pixel_values_list:
                    subject_ref_pixel_values = np.concatenate(sample_subject_ref_pixel_values_list, axis=0)
                if sample_mask_pixel_values_list:
                    mask_pixel_values = reduce(lambda x, y: np.maximum(x, y), sample_mask_pixel_values_list)
                if sample_subject_caption_list:
                    if random.random() < 0.3 and len(sample_subject_caption_list) == 1:
                        text = sample_subject_caption_list[0]

            # random ref frame mode
            if self.random_ref_frame_mode and \
                (random.random() < 0.5 or len(subject_ref_pixel_values) == 0):
                remain_ref_num = self.max_ref_num_per_subjct * self.max_subject_num - len(subject_ref_pixel_values)
                if remain_ref_num > 0:
                    # If subject_ref_pixel_values is empty, first ref from pixel_values
                    if len(subject_ref_pixel_values) == 0:
                        first_ref_frame_id = random.choice(range(pixel_values.shape[0]))
                        first_ref_frame = pixel_values[first_ref_frame_id]
                        subject_ref_pixel_values = np.expand_dims(first_ref_frame, axis=0)
                        remain_ref_num -= 1
                    
                    # Subsequent refs from original video_reader
                    if remain_ref_num > 0:
                        additional_ref_num = random.randint(1, remain_ref_num)
                        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader_for_ref:
                            try:
                                # Sample random frames from the entire video
                                total_frames = len(video_reader_for_ref)
                                random_frame_indices = random.sample(range(total_frames), k=additional_ref_num)
                                random_frame_indices = sorted(random_frame_indices)
                                
                                sample_args = (video_reader_for_ref, random_frame_indices)
                                additional_ref_frames = func_timeout(
                                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                                )
                                
                                # Resize frames to match target size
                                resized_ref_frames = []
                                for i in range(len(additional_ref_frames)):
                                    frame = additional_ref_frames[i]
                                    resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                                    resized_ref_frames.append(resized_frame)
                                additional_ref_frames = np.array(resized_ref_frames)
                                
                                subject_ref_pixel_values = np.concatenate(
                                    [subject_ref_pixel_values, additional_ref_frames], axis=0
                                )
                            except FunctionTimedOut:
                                raise ValueError(f"Read ref frames timeout.")
                            except Exception as e:
                                raise ValueError(f"Failed to extract ref frames from video. Error is {e}.")
            
            # ref frame data augment
            subject_ref_pixel_values = self.images_augment(
                subject_ref_pixel_values,
                skip_first_frame=True,
                aug_opt_names=["geo"]
            )

            # conditinal prompt inject
            anime_type_prefix_dict = {
                "2d": "2d anime flat style, stepped beat, ",
                "3d": "3d CG realism style, smooth motion, ",
            }
            anime_type = data_info.get("anime_type")
            text = anime_type_prefix_dict.get(anime_type, "") + text

            # Random null text generation
            if random.random() < self.text_drop_ratio:
                text = ''

            # pixel_values format fhwc
            batch_data = {
                "pixel_values": pixel_values,
                "control_pixel_values": control_pixel_values,
                "text": text,
                "data_type": "video",
                "ref_pixel_values": subject_ref_pixel_values,
                "mask_pixel_values": mask_pixel_values,
                "control_type": control_video_type
            }
            return batch_data
        else:
            # Process Image data
            if 'file_path' in data_info:
                image_path, text = data_info['file_path'], data_info['text']
            else:
                image_path = data_info["path"]
            
            if not self.data_root:
                if not image_path.startswith(self.data_root):
                    image_path = os.path.join(self.data_root, image_path.lstrip("/"))


            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if not self.data_root:
                control_image_id = control_image_id
            else:
                if not control_image_id.startswith(self.data_root):
                    control_image_id = os.path.join(self.data_root, control_image_id.lstrip("/"))


            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            return image, control_image, text, 'image'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        data_type = data_type if data_type is not None else "video"
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                data_type_local = data_type_local if data_type_local is not None else "video"
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                # pixel_values, control_pixel_values, text, data_type, ref_pixel_values, mask_pixel_values
                sample = self.get_batch(idx)
                sample["idx"] = idx
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(sample["pixel_values"].size())
            mask_pixel_values = sample["pixel_values"] * (1 - mask) + torch.ones_like(sample["pixel_values"]) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        return sample
