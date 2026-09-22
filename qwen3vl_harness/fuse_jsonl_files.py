#!/usr/bin/env python3
"""
融合 tool.json(由 convert_tool_json.py 从 thinking_results 转换得到) 与 pred_infos(推理产物) 两个 JSON/JSONL 文件。

用法:
    python fuse_jsonl_files.py --tool tool.json --input pred_infos.jsonl --output fused.jsonl
"""

import json
import argparse
import os
from pathlib import Path

def load_jsonl_file(file_path):
    """加载JSONL文件"""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue
    
    return entries

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    else:
        return [data]

def load_file(file_path):
    """根据文件扩展名加载文件"""
    if file_path.endswith('.json'):
        return load_json_file(file_path)
    elif file_path.endswith('.jsonl'):
        return load_jsonl_file(file_path)
    else:
        # 尝试自动检测格式
        with open(file_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                return load_json_file(file_path)
            else:
                return load_jsonl_file(file_path)

def create_entry_key(entry):
    """创建条目的唯一标识键

    合并规则：基于 dense_caption 下的 content 内容是否相同来判定匹配。
    将 dense_caption 列表中所有元素的 content 拼接后作为唯一标识。
    """
    dense_caption = entry.get('dense_caption', [])

    contents = []
    if isinstance(dense_caption, list):
        for item in dense_caption:
            if isinstance(item, dict):
                content = item.get('content', '')
            else:
                # 如果不是 dict，直接转成字符串
                content = str(item)
            # 规范化：去除首尾空白，避免因空白差异导致匹配失败
            contents.append(content.strip())
    elif isinstance(dense_caption, dict):
        contents.append(str(dense_caption.get('content', '')).strip())
    else:
        contents.append(str(dense_caption).strip())

    # 用一个不会出现在正常文本中的分隔符拼接
    return '\u0001'.join(contents)

def fuse_entries(thinking_entries, pred_entries):
    """融合两个文件中的条目"""
    
    # 为thinking_entries创建索引
    thinking_dict = {}
    for entry in thinking_entries:
        key = create_entry_key(entry)
        thinking_dict[key] = entry
    
    # 为pred_entries创建索引
    pred_dict = {}
    for entry in pred_entries:
        key = create_entry_key(entry)
        pred_dict[key] = entry
    
    # 融合结果
    fused_entries = []
    matched_count = 0
    unmatched_thinking = []
    unmatched_pred = []
    
    # 首先匹配thinking_entries中的条目
    for key, thinking_entry in thinking_dict.items():
        if key in pred_dict:
            # 找到匹配项，进行融合
            pred_entry = pred_dict[key]
            fused_entry = thinking_entry.copy()
            
            # 从pred_entry中添加推理相关的字段
            fused_entry['pred_path'] = pred_entry.get('path', '')
            fused_entry['pred_lose_path'] = pred_entry.get('lose_path', '')
            fused_entry['pred_ref_img_infos'] = pred_entry.get('ref_img_infos', [])
            fused_entry['pred_dense_caption'] = pred_entry.get('dense_caption', [])
            fused_entry['pred_source_tag'] = pred_entry.get('source_tag', '')
            fused_entry['pred_caption'] = pred_entry.get('caption', [])
            
            fused_entries.append(fused_entry)
            matched_count += 1
            
            # 从pred_dict中移除已匹配的条目
            del pred_dict[key]
        # else:
        #     # 没有匹配的pred_entry，保留thinking_entry
        #     fused_entry = thinking_entry.copy()
        #     fused_entry['pred_path'] = ''
        #     fused_entry['pred_lose_path'] = ''
        #     fused_entry['pred_ref_img_infos'] = []
        #     fused_entry['pred_dense_caption'] = []
        #     fused_entry['pred_source_tag'] = ''
        #     fused_entry['pred_caption'] = []
        #     fused_entries.append(fused_entry)
        #     unmatched_thinking.append(key)
    
    # # 处理剩余的pred_entries（在thinking_entries中没有匹配的）
    # for key, pred_entry in pred_dict.items():
    #     fused_entry = {
    #         'control_path': pred_entry.get('control_path', ''),
    #         'path': pred_entry.get('path', ''),
    #         'lose_path': pred_entry.get('lose_path', ''),
    #         'ref_img_infos': pred_entry.get('ref_img_infos', []),
    #         'dense_caption': pred_entry.get('dense_caption', []),
    #         'source_tag': pred_entry.get('source_tag', ''),
    #         'caption': pred_entry.get('caption', []),
    #         'final_answer': '',
    #         'completion_length': 0,
    #         'final_answer_length': 0,
    #         'evaluator': [],
    #         'num_evaluator': 0,
    #         'pred_path': pred_entry.get('path', ''),
    #         'pred_lose_path': pred_entry.get('lose_path', ''),
    #         'pred_ref_img_infos': pred_entry.get('ref_img_infos', []),
    #         'pred_dense_caption': pred_entry.get('dense_caption', []),
    #         'pred_source_tag': pred_entry.get('source_tag', ''),
    #         'pred_caption': pred_entry.get('caption', []),
    #         'is_pred_only': True  # 标记为仅存在于pred_infos的条目
    #     }
    #     fused_entries.append(fused_entry)
    #     unmatched_pred.append(key)
    
    return fused_entries, matched_count, len(unmatched_thinking), len(unmatched_pred)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="融合 tool.json(思考/工具结果) 与 pred_infos(推理产物) 两个 JSON/JSONL 文件"
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="tool.json 文件（由 convert_tool_json.py 从 thinking_results 转换得到）",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="pred_infos 推理产物文件（JSON/JSONL），例如 pred_infos.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="融合后输出文件路径，例如 fused.jsonl",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    tool_file = args.tool
    pred_file = args.input
    output_file = args.output

    print("开始融合JSON文件...")
    print(f"tool.json文件: {tool_file}")
    print(f"pred_infos文件: {pred_file}")
    print(f"输出文件: {output_file}")
    print()

    # 检查文件是否存在
    if not Path(tool_file).exists():
        print(f"错误: tool.json文件不存在: {tool_file}")
        return 1

    if not Path(pred_file).exists():
        print(f"错误: pred_infos文件不存在: {pred_file}")
        return 1

    # 加载文件
    print("正在加载tool.json文件...")
    tool_entries = load_file(tool_file)
    print(f"tool.json条目数: {len(tool_entries)}")

    print("正在加载pred_infos文件...")
    pred_entries = load_file(pred_file)
    print(f"pred_infos条目数: {len(pred_entries)}")
    print()

    # 融合条目
    print("正在融合条目...")
    fused_entries, matched_count, unmatched_tool_count, unmatched_pred_count = fuse_entries(tool_entries, pred_entries)

    print("融合结果:")
    print(f"总融合条目数: {len(fused_entries)}")
    print(f"成功匹配条目数: {matched_count}")
    print(f"tool.json中未匹配条目数: {unmatched_tool_count}")
    print(f"pred_infos中未匹配条目数: {unmatched_pred_count}")
    print()


    # 写入融合结果（JSONL 格式，每行一个 JSON 对象，兼容下游 load_input_data）
    print("正在写入融合结果...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in fused_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"融合完成!")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {Path(output_file).stat().st_size / 1024:.2f} KB")

    return 0

if __name__ == "__main__":
    exit(main())