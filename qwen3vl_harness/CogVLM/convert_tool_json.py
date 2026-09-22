#!/usr/bin/env python3
"""
将JSONL文件转换为JSON格式并进行字段处理
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 从evaluator.md文件中提取的正确评估器名称映射
CORRECT_EVALUATOR_NAMES = {
    # 基础评估器 (1-11)
    "文本遵循验证器": "文本遵循验证器",
    "主体ID保持验证器": "主体ID保持验证器", 
    "参考图像视觉特征验证器": "参考图像视觉特征验证器",
    "参考图像像素对齐验证器": "参考图像像素对齐验证器",
    "控制视频遵循验证器": "控制视频遵循验证器",
    "物理动态特效验证器": "物理动态特效验证器",
    "多模态隐含因果验证器": "多模态隐含因果验证器",
    "时空平滑度验证器": "时空平滑度验证器",
    "交互逻辑性验证器": "交互逻辑性验证器",
    "负面伪影检测验证器": "负面伪影检测验证器",
    "Storyboard标注遵循验证器": "Storyboard标注遵循验证器",
    
    # 预训练模型评估器 (M1-M3)
    "美学评分器": "美学评分器",
    "动态程度评估器": "动态程度评估器",
    "运动平滑度评估器": "运动平滑度评估器"
}

# 简写名称到全称的映射
SHORT_NAME_MAPPING = {
    "负面伪影检测器": "负面伪影检测验证器",
    "负面检测器": "负面伪影检测验证器",
    "伪影检测器": "负面伪影检测验证器",
    # 可以根据需要添加其他简写映射
}

def extract_evaluators_from_final_answer(final_answer):
    """从final_answer中提取评估器列表，并验证名称正确性"""
    if not final_answer:
        return []
    
    # 查找[tools]标记后的所有评估器
    tools_pattern = r'\[tools\]\s*(\[.*?\])+'
    match = re.search(tools_pattern, final_answer, re.DOTALL)
    
    if match:
        # 提取所有[评估器]格式的内容
        evaluator_pattern = r'\[(.*?)\]'
        evaluator_matches = re.findall(evaluator_pattern, match.group(0))
        
        # 第一个匹配是[tools]，所以跳过
        raw_evaluators = []
        for i, evaluator_match in enumerate(evaluator_matches):
            if i == 0:  # 跳过[tools]本身
                continue
            if evaluator_match.strip():
                raw_evaluators.append(evaluator_match.strip())
        
        # 验证并修正评估器名称
        validated_evaluators = []
        for evaluator in raw_evaluators:
            # 首先检查简写名称映射
            if evaluator in SHORT_NAME_MAPPING:
                correct_name = SHORT_NAME_MAPPING[evaluator]
                validated_evaluators.append(correct_name)
                print(f"信息: 简写名称 '{evaluator}' 已映射为 '{correct_name}'")
                continue
                
            # 检查是否在正确名称映射中
            if evaluator in CORRECT_EVALUATOR_NAMES:
                validated_evaluators.append(evaluator)
            else:
                # 尝试模糊匹配
                matched = False
                for correct_name in CORRECT_EVALUATOR_NAMES:
                    if evaluator in correct_name or correct_name in evaluator:
                        validated_evaluators.append(correct_name)
                        matched = True
                        print(f"信息: 评估器名称 '{evaluator}' 已修正为 '{correct_name}'")
                        break
                
                if not matched:
                    print(f"警告: 未知的评估器名称 '{evaluator}'，已跳过")
        
        return validated_evaluators
    
    return []

def process_jsonl_file(input_path, output_path):
    """处理JSONL文件"""
    
    # 读取JSONL文件
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # 处理每个条目
                processed_entry = process_entry(entry)
                entries.append(processed_entry)
                
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue
    
    # 写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    
    return len(entries)

def process_entry(entry):
    """处理单个JSON条目"""
    
    # 创建新条目，移除completion字段
    new_entry = {}
    for key, value in entry.items():
        if key != "completion":
            new_entry[key] = value
    
    # 处理final_answer字段，提取评估器
    if "final_answer" in entry:
        final_answer = entry["final_answer"]
        evaluators = extract_evaluators_from_final_answer(final_answer)
        
        # 添加evaluator字段
        new_entry["evaluator"] = evaluators
        
        # 添加num_evaluator字段，记录评估器数量
        new_entry["num_evaluator"] = len(evaluators)
    
    return new_entry

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将JSONL文件转换为JSON格式并提取评估器(tools)信息"
    )
    parser.add_argument(
        "input_file",
        help="输入的JSONL文件路径",
    )
    parser.add_argument(
        "output_file",
        help="输出的JSON文件路径 (例如 tool.json)",
    )
    return parser.parse_args()

def main():
    """主函数"""
    
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    
    # 确保输出目录存在
    output_dir = Path(output_file).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始处理文件: {input_file}")
    print(f"使用正确的评估器名称列表:")
    for name in CORRECT_EVALUATOR_NAMES:
        print(f"  - {name}")
    print()
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return 1
    
    try:
        # 处理文件
        processed_count = process_jsonl_file(input_file, output_file)
        
        print(f"处理完成!")
        print(f"成功处理 {processed_count} 个条目")
        print(f"输出文件: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())