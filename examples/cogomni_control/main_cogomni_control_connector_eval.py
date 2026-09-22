from glob import glob
import yaml
import os
import jsonlines

from videox_fun.inference import CogOmniControl_Connector_Inferencer
from videox_fun.utils import write_video


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CogOmniControl")
    parser.add_argument("--eval_anno_path", required=True, nargs="+")
    parser.add_argument("--infer_cfg_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def parse_eval_anno(eval_anno_path):
    # Get data_infos for visualization
    data_infos = []
    if eval_anno_path.endswith(".jsonl"):
        with jsonlines.open(eval_anno_path) as reader:
            data_infos = list(reader)
    elif eval_anno_path.endswith(".txt"):
        with open(eval_anno_path, "r") as f:
            data_list = [line.strip() for line in f]
        for data_dir in data_list:
            data_input_paths = sorted(glob(os.path.join(data_dir, "*")))
            control_path = None
            video_path = None
            ref_img_infos = []
            caption = ""
            for data_input_path in data_input_paths:
                if "control" in os.path.basename(data_input_path):
                    control_path = data_input_path
                if "win" in data_input_path:
                    video_path = data_input_path
                if "ref" in os.path.basename(data_input_path):
                    ref_img_infos.append({
                        "ref_imgs": [
                            {
                                "img_path": os.path.realpath(data_input_path)
                            }
                        ]
                    })
                if "caption" in data_input_path or "prompt" in data_input_path:
                    with open(data_input_path, "r") as f:
                        caption = f.read().strip()
            assert control_path is not None, f"Control path not found in {data_dir}"
            assert len(ref_img_infos) > 0, f"Reference image path not found in {data_dir}"

            data_info = {
                "control_path": os.path.realpath(control_path),
                "ref_img_infos": ref_img_infos,
                "dense_caption": [
                    {"content": caption, "dense_caption_type": "manual"}
                ],
                "path": os.path.realpath(video_path) if video_path else os.path.realpath(control_path)
            }
            data_infos.append(data_info)
    else:
        raise ValueError("Unsupported annotation format: {}".format(eval_anno_path))
    # convert data_infos to task_infos for eval
    task_infos = []
    for data_info in data_infos:
        ref_imgs = []
        for subject_img_info in data_info["ref_img_infos"]:
            ref_img_metas = subject_img_info["ref_imgs"]
            for ref_img_meta in ref_img_metas:
                ref_imgs.append(ref_img_meta["img_path"])
        # print(ref_imgs)
        task_info = {
            "control_video": data_info["control_path"],
            "prompt": data_info["dense_caption"][0]["content"],
            "ref_images": ref_imgs,
        }
        task_infos.append(task_info)
        print(data_info["control_path"])
    
    return task_infos, data_infos


def main(args):
    # parse eval annotation to get task_infos and data_infos
    eval_anno_paths = args.eval_anno_path
    if not isinstance(eval_anno_paths, list):
        eval_anno_paths = [eval_anno_paths]
    task_infos = []
    data_infos = []
    for eval_anno_path in eval_anno_paths:
        print(f"Parsing eval annotation from {eval_anno_path}...")
        task_infos_part, data_infos_part = parse_eval_anno(eval_anno_path)
        task_infos.extend(task_infos_part)
        data_infos.extend(data_infos_part)
    print(len(task_infos), len(data_infos))

    # load infer_cfg from yml
    with open(args.infer_cfg_path, "r") as f:
        infer_cfg = yaml.safe_load(f)
    inferencer = CogOmniControl_Connector_Inferencer()

    ouput_pred_info_path = os.path.join(args.output_dir, "pred_infos.jsonl")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pred_info_mode = "w" if not os.path.exists(ouput_pred_info_path) else "a"
    with jsonlines.open(ouput_pred_info_path, mode=pred_info_mode, flush=True) as pred_info_writer:
        for task_id, task_info in enumerate(task_infos):
            task_name = os.path.basename(task_info["control_video"]).split(".")[0]
            output_task_dir = os.path.join(args.output_dir, task_name)
            # Avoid name collision when multiple tasks share the same control_video basename
            if os.path.exists(output_task_dir):
                suffix_id = 1
                while os.path.exists(f"{output_task_dir}_{suffix_id}"):
                    suffix_id += 1
                output_task_dir = f"{output_task_dir}_{suffix_id}"
            os.makedirs(output_task_dir, exist_ok=True)
            pred_video = inferencer.predict(**infer_cfg, **task_info)
            # save pred_video and task_info & infer_cfg to yaml
            output_task_info = {
                "task_info": task_info,
                "infer_cfg": infer_cfg
            }        
            with open(os.path.join(output_task_dir, "task_info.yaml"), "w") as f:
                yaml.dump(output_task_info, f, indent=4, default_flow_style=False, allow_unicode=True)

            output_video_path = os.path.join(output_task_dir, "pred_video.mp4")
            write_video(pred_video, output_video_path, fps=24)

            pred_info = {
                **data_infos[task_id],
                "lose_path": os.path.realpath(output_video_path),
            }
            pred_info_writer.write(pred_info)
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
