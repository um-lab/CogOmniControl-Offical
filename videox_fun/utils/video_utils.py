from contextlib import contextmanager
import gc

from decord import VideoReader
import imageio


def write_video(frames, output_path, fps, output_params=["-crf", "17"]):
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1, codec="libx264", output_params=output_params)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()
