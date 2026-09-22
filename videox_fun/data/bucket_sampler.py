# Copyright (c) OpenMMLab. All rights reserved.
import os
import traceback
from typing import (Generic, Iterable, Iterator, List, Optional, Sequence,
                    Sized, TypeVar, Union)

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler, DistributedSampler

ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}

ASPECT_RATIO_RANDOM_CROP_512 = {
    '0.42': [320.0, 768.0], '0.5': [352.0, 704.0], 
    '0.57': [384.0, 672.0], '0.68': [416.0, 608.0], '0.78': [448.0, 576.0], '0.88': [480.0, 544.0], 
    '0.94': [480.0, 512.0], '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], 
    '1.13': [544.0, 480.0], '1.29': [576.0, 448.0], '1.46': [608.0, 416.0], '1.75': [672.0, 384.0], 
    '2.0': [704.0, 352.0],  '2.4': [768.0, 320.0]
}

ASPECT_RATIO_720P = {
    "0.38": (588, 1568),
    "0.43": (628, 1466),
    "0.48": (666, 1388),
    "0.50": (678, 1356),
    "0.53": (698, 1318),
    "0.54": (706, 1306),
    "0.56": (720, 1280),  # base
    "0.62": (758, 1212),
    "0.67": (784, 1176),
    "0.75": (832, 1110),
    "1.00": (960, 960),
    "1.33": (1108, 832),
    "1.50": (1176, 784),
    "1.78": (1280, 720),
    "1.89": (1320, 698),
    "2.00": (1358, 680),
    "2.08": (1386, 666),
}

ASPECT_RATIO_RANDOM_CROP_720P = {
    "0.48": (666, 1388), "0.50": (678, 1356),
    "0.53": (698, 1318), "0.54": (706, 1306), "0.56": (720, 1280), "0.62": (758, 1212),
    "0.67": (784, 1176), "0.75": (832, 1110), "1.00": (960, 960),
    "1.33": (1108, 832), "1.50": (1176, 784), "1.78": (1280, 720), "1.89": (1320, 698),
    "2.00": (1358, 680), "2.08": (1386, 666),
}

ASPECT_RATIO_768 = {
    '0.25': [384.0, 1536.0], '0.26': [384.0, 1488.0], '0.27': [384.0, 1440.0], '0.28': [384.0, 1392.0],
    '0.32': [432.0, 1344.0], '0.33': [432.0, 1296.0], '0.35': [432.0, 1248.0], '0.4': [480.0, 1200.0],
    '0.42': [480.0, 1152.0], '0.48': [528.0, 1104.0], '0.5': [528.0, 1056.0], '0.52': [528.0, 1008.0],
    '0.57': [576.0, 1008.0], '0.6': [576.0, 960.0], '0.68': [624.0, 912.0], '0.72': [624.0, 864.0],
    '0.78': [672.0, 864.0], '0.82': [672.0, 816.0], '0.88': [720.0, 816.0], '0.94': [720.0, 768.0],
    '1.0': [768.0, 768.0], '1.07': [768.0, 720.0], '1.13': [816.0, 720.0], '1.21': [816.0, 672.0],
    '1.29': [864.0, 672.0], '1.38': [864.0, 624.0], '1.46': [912.0, 624.0], '1.67': [960.0, 576.0],
    '1.75': [1008.0, 576.0], '2.0': [1056.0, 528.0], '2.09': [1104.0, 528.0], '2.4': [1152.0, 480.0],
    '2.5': [1200.0, 480.0], '2.89': [1332.0, 432.0], '3.0': [1296.0, 432.0], '3.11': [1344.0, 432.0],
    '3.62': [1408.0, 384.0], '3.75': [1440.0, 384.0], '3.88': [1488.0, 384.0], '4.0': [1536.0, 384.0]
}

ASPECT_RATIO_RANDOM_CROP_768 = {
    '0.42': [480.0, 1152.0], '0.5': [528.0, 1056.0],
    '0.57': [576.0, 1008.0], '0.68': [624.0, 912.0], '0.78': [672.0, 864.0], '0.88': [720.0, 816.0],
    '0.94': [720.0, 768.0], '1.0': [768.0, 768.0], '1.07': [768.0, 720.0],
    '1.13': [816.0, 720.0], '1.29': [864.0, 672.0], '1.46': [912.0, 624.0], '1.75': [1008.0, 576.0],
    '2.0': [1056.0, 528.0], '2.4': [1152.0, 480.0]
}

ASPECT_RATIO_1024 = {
    "0.25": (512, 2048),
    "0.26": (512, 1984),
    "0.27": (512, 1920),
    "0.28": (512, 1856),
    "0.32": (576, 1792),
    "0.33": (576, 1728),
    "0.35": (576, 1664),
    "0.40": (640, 1600),
    "0.42": (640, 1536),
    "0.48": (704, 1472),
    "0.50": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.60": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.00": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "2.00": (1408, 704),
    "2.09": (1472, 704),
    "2.40": (1536, 640),
    "2.50": (1600, 640),
    "2.89": (1664, 576),
    "3.00": (1728, 576),
    "3.11": (1792, 576),
    "3.62": (1856, 512),
    "3.75": (1920, 512),
    "3.88": (1984, 512),
    "4.00": (2048, 512),
}

ASPECT_RATIO_1080P = {
    "0.38": (882, 2352),
    "0.43": (942, 2198),
    "0.48": (998, 2080),
    "0.50": (1018, 2036),
    "0.53": (1048, 1980),
    "0.54": (1058, 1958),
    "0.56": (1080, 1920),  # base
    "0.62": (1138, 1820),
    "0.67": (1176, 1764),
    "0.75": (1248, 1664),
    "1.00": (1440, 1440),
    "1.33": (1662, 1246),
    "1.50": (1764, 1176),
    "1.78": (1920, 1080),
    "1.89": (1980, 1048),
    "2.00": (2036, 1018),
    "2.08": (2078, 998),
}

ASPECT_RATIO_RANDOM_CROP_PROB = [
    1, 2,
    4, 4, 4, 4,
    8, 8, 8,
    4, 4, 4, 4,
    2, 1
]

ASPECT_RATIO_RANDOM_CROP_PROB = np.array(ASPECT_RATIO_RANDOM_CROP_PROB) / sum(ASPECT_RATIO_RANDOM_CROP_PROB)

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def get_image_size_without_loading(path):
    with Image.open(path) as img:
        return img.size  # (width, height)

class GeneratorDistributedSampler(DistributedSampler):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        generator=None
    ) -> None:
        self.generator = generator

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            if self.generator is None:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
            else:
                g = self.generator
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._pos_start = 0

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=generator).tolist()
                if self._pos_start >= n:
                    self._pos_start = 0
                # print("xx top 10", xx[:10], self._pos_start)
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

class AspectRatioBatchImageSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        train_folder: str = None,
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio 
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios] 
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                image_dict = self.dataset[idx]

                width, height = image_dict.get("width", None), image_dict.get("height", None)
                if width is None or height is None:
                    image_id, name = image_dict['file_path'], image_dict['text']
                    if not self.train_folder:
                        image_dir = image_id
                    else:
                        if not image_id.startswith(self.train_folder):
                            image_dir = os.path.join(self.train_folder, image_id.lstrip("/"))
                        else:
                            image_dir = image_dir

                    width, height = get_image_size_without_loading(image_dir)

                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        video_folder: str = None,
        train_data_format: str = "webvid",
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.video_folder = video_folder
        self.train_data_format = train_data_format
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio 
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios] 
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                video_dict = self.dataset[idx]
                width, more = video_dict.get("width", None), video_dict.get("height", None)

                if width is None or height is None:
                    if self.train_data_format == "normal":
                        video_id, name = video_dict['file_path'], video_dict['text']
                        if self.video_folder is None:
                            video_dir = video_id
                        else:
                            if not video_id.startswith(self.train_folder):
                                video_dir = os.path.join(self.video_folder, video_id)
                            else:
                                video_dir = video_id
                    else:
                        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
                        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
                    cap = cv2.VideoCapture(video_dir)

                    # 获取视频尺寸
                    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 浮点数转换为整数
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数
                    
                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e, self.dataset[idx], "This item is error, please check it.")
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchImageVideoSampler(BatchSampler):
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
                 train_folder: str = None,
                 aspect_ratios: dict = ASPECT_RATIO_512,
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
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            'image':{ratio: [] for ratio in aspect_ratios}, 
            'video':{ratio: [] for ratio in aspect_ratios}
        }

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset[idx].get('type', 'video')
            if content_type == 'image':
                try:
                    image_dict = self.dataset[idx]
                    width, height = image_dict.get("width", image_dict.get("w", None)), image_dict.get("height", image_dict.get("h", None))

                    if width is None or height is None:
                        if "file_path" in image_dict:
                            image_id = image_dict["file_path"]
                        else:
                            image_id = image_dict["path"]
                        if not self.train_folder:
                            image_dir = image_id
                        else:
                            if not image_id.startswith(self.train_folder):
                                image_dir = os.path.join(self.train_folder, image_id.lstrip("/"))
                            else:
                                image_dir = image_dir

                        width, height = get_image_size_without_loading(image_dir)

                        ratio = height / width # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width # self.dataset[idx]
                except Exception as e:
                    traceback.print_exc()
                    print(e, self.dataset[idx], "This item is error, please check it.")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['image'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
            else:
                try:
                    video_dict = self.dataset[idx]
                    width, height = video_dict.get("width", video_dict.get("w", None)), video_dict.get("height", video_dict.get("h", None))

                    if width is None or height is None:
                        if "file_path" in video_dict:
                            video_id = video_dict["file_path"]
                        else:
                            video_id = video_dict["path"]
                        if not self.train_folder:
                            video_dir = video_id
                        else:
                            if not video_id.startswith(self.train_folder):
                                video_dir = os.path.join(self.train_folder, video_id.lstrip("/"))
                            else:
                                video_dir = video_id
                                
                        cap = cv2.VideoCapture(video_dir)


                        # 获取视频尺寸
                        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 浮点数转换为整数
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数
                        
                        ratio = height / width # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width # self.dataset[idx]
                except Exception as e:
                    traceback.print_exc()
                    print(e, self.dataset[idx], "This item is error, please check it.")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['video'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
