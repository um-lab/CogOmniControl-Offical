#!/bin/bash

INPUT_FILES=(
   /CogControlBench/annotations.jsonl
)

# 输出目录
OUTPUT_DIR="./thinking_results"

# tool.json 输出路径
TOOL_JSON_OUTPUT="./qwen3vl_harness/inputs/tool.json"

# 模型配置
BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
LORA_PATH="weights/CogVLM"

BACKEND="transformers"  # vllm 或 transformers
BATCH_SIZE=1
MAX_NEW_TOKENS=4096
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=50


# GPU配置
GPU_MEMORY_UTILIZATION=0.85
TENSOR_PARALLEL_SIZE=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"

# 创建临时目录
TEMP_DIR="/tmp/cogomni_thinking_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TEMP_DIR}"
echo "临时目录: ${TEMP_DIR}"

TEMP_INPUT="${TEMP_DIR}/combined_input.jsonl"
echo "合并输入文件..."
valid_files=0
for input_file in "${INPUT_FILES[@]}"; do
    if [ -f "${input_file}" ]; then
        file_size=$(wc -c < "${input_file}")
        if [ "$file_size" -gt 10 ]; then  # 至少10字节的文件才认为是有效的
            echo "添加文件: ${input_file} (大小: ${file_size} 字节)"
            if [[ "${input_file}" == *.json ]]; then
                # JSON数组格式转换为JSONL
                python3 -c "
import json
with open('${input_file}', 'r') as f:
    data = json.load(f)
with open('${TEMP_INPUT}', 'a') as out:
    for item in data:
        json.dump(item, out, ensure_ascii=False)
        out.write('\n')
"
            elif [[ "${input_file}" == *.txt ]]; then
                python3 -c "
import os, json
from glob import glob

with open('${input_file}', 'r') as f:
    data_list = [line.strip() for line in f if line.strip()]

with open('${TEMP_INPUT}', 'a') as out:
    for data_dir in data_list:
        if not os.path.isdir(data_dir):
            continue
        data_input_paths = sorted(glob(os.path.join(data_dir, '*')))
        control_path = None
        ref_img_paths = []
        caption = ''
        for p in data_input_paths:
            basename = os.path.basename(p)
            if 'control' in basename:
                control_path = p
            if 'ref' in basename:
                ref_img_paths.append(os.path.realpath(p))
            if 'caption' in basename or 'prompt' in basename:
                with open(p, 'r') as cf:
                    caption = cf.read().strip()
        if control_path is None:
            continue
        item = {
            'control_path': os.path.realpath(control_path),
            'ref_img_infos': ref_img_paths,
            'dense_caption': [{'content': caption, 'dense_caption_type': 'manual'}],
        }
        json.dump(item, out, ensure_ascii=False)
        out.write('\n')
"
            else
                # JSONL格式直接追加
                cat "${input_file}" >> "${TEMP_INPUT}"
            fi
            ((valid_files++))
        else
            echo "警告: 文件过小或为空: ${input_file}"
        fi
    else
        echo "警告: 文件不存在: ${input_file}"
    fi
done

# 检查是否有有效的输入文件
if [ "$valid_files" -eq 0 ]; then
    echo "错误: 没有找到有效的输入文件！"
    echo "请检查 INPUT_FILES 数组中的路径是否正确"
    rm -rf "${TEMP_DIR}"
    exit 1
fi

# 检查合并后的文件是否为空
combined_size=$(wc -c < "${TEMP_INPUT}")
if [ "$combined_size" -le 10 ]; then
    echo "错误: 合并后的输入文件为空或过小！"
    echo "请检查输入文件是否包含有效的JSON/JSONL数据"
    rm -rf "${TEMP_DIR}"
    exit 1
fi

# 获取数据总条数
TOTAL=$(python3 -c "
import json
with open('${TEMP_INPUT}', 'r') as f:
    content = f.read().strip()
if content.startswith('['):
    data = json.loads(content)
    print(len(data))
else:
    print(sum(1 for line in content.split('\n') if line.strip()))
")

# 检查数据条数是否有效
if [ "$TOTAL" -eq 0 ]; then
    echo "错误: 没有找到有效的数据条目！"
    echo "请检查输入文件的格式（应为JSON数组或JSONL）"
    rm -rf "${TEMP_DIR}"
    exit 1
fi

echo "数据总条数: ${TOTAL}"

# 生成输出文件名（包含时间戳）
TIMESTAMP=$(date +"%Y.%m.%d-%H.%M.%S")
OUTPUT_FILE="${OUTPUT_DIR}/thinking_results_${TIMESTAMP}.jsonl"

echo "输出文件: ${OUTPUT_FILE}"

# 构建推理命令基础部分
INFERENCE_SCRIPT="./qwen3vl_harness/CogVLM/cogomni_inference.py"
INFERENCE_BASE_CMD="python3 ${INFERENCE_SCRIPT}"
INFERENCE_BASE_CMD+=" --input ${TEMP_INPUT}"
INFERENCE_BASE_CMD+=" --base_model ${BASE_MODEL}"

# 可选LoRA
if [ -n "${LORA_PATH}" ] && [ "${LORA_PATH}" != "None" ] && [ "${LORA_PATH}" != "none" ]; then
    INFERENCE_BASE_CMD+=" --lora_path ${LORA_PATH}"
    echo "使用LoRA: ${LORA_PATH}"
else
    echo "不使用LoRA"
fi

INFERENCE_BASE_CMD+=" --backend ${BACKEND}"
INFERENCE_BASE_CMD+=" --batch_size ${BATCH_SIZE}"
INFERENCE_BASE_CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
INFERENCE_BASE_CMD+=" --temperature ${TEMPERATURE}"
INFERENCE_BASE_CMD+=" --top_p ${TOP_P}"
INFERENCE_BASE_CMD+=" --top_k ${TOP_K}"
INFERENCE_BASE_CMD+=" --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION}"
INFERENCE_BASE_CMD+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"

# ============================================================
# 数据分片和并行推理（新增部分）
# ============================================================

Q1=$((TOTAL / 4))
Q2=$((TOTAL / 4 * 2))
Q3=$((TOTAL / 4 * 3))
Q4=${TOTAL}

echo "数据分片:"
echo "  Part 0 (GPU 0): [0, ${Q1})"
echo "  Part 1 (GPU 0): [${Q1}, ${Q2})"
echo "  Part 2 (GPU 1): [${Q2}, ${Q3})"
echo "  Part 3 (GPU 1): [${Q3}, ${Q4})"

# 生成分片输出文件名
PART0_OUTPUT="${OUTPUT_DIR}/thinking_results_${TIMESTAMP}_part0.jsonl"
PART1_OUTPUT="${OUTPUT_DIR}/thinking_results_${TIMESTAMP}_part1.jsonl"
PART2_OUTPUT="${OUTPUT_DIR}/thinking_results_${TIMESTAMP}_part2.jsonl"
PART3_OUTPUT="${OUTPUT_DIR}/thinking_results_${TIMESTAMP}_part3.jsonl"

# 4个进程全部并行启动
echo "[GPU 0] 启动 Part 0: [0, ${Q1})"
CUDA_VISIBLE_DEVICES=0 ${INFERENCE_BASE_CMD} \
    --output ${PART0_OUTPUT} \
    --start_index 0 \
    --end_index ${Q1} \
    --skip_existing &

echo "[GPU 0] 启动 Part 1: [${Q1}, ${Q2})"
CUDA_VISIBLE_DEVICES=0 ${INFERENCE_BASE_CMD} \
    --output ${PART1_OUTPUT} \
    --start_index ${Q1} \
    --end_index ${Q2} \
    --skip_existing &

echo "[GPU 1] 启动 Part 2: [${Q2}, ${Q3})"
CUDA_VISIBLE_DEVICES=1 ${INFERENCE_BASE_CMD} \
    --output ${PART2_OUTPUT} \
    --start_index ${Q2} \
    --end_index ${Q3} \
    --skip_existing &

echo "[GPU 1] 启动 Part 3: [${Q3}, ${Q4})"
CUDA_VISIBLE_DEVICES=1 ${INFERENCE_BASE_CMD} \
    --output ${PART3_OUTPUT} \
    --start_index ${Q3} \
    --end_index ${Q4} \
    --skip_existing &

echo "4个推理进程已全部并行启动，等待完成..."
wait
echo "所有推理完成！"

# 合并4份结果
cat ${PART0_OUTPUT} ${PART1_OUTPUT} ${PART2_OUTPUT} ${PART3_OUTPUT} > ${OUTPUT_FILE}
echo "结果已合并到 ${OUTPUT_FILE}（共4个分片）"

# 清理临时文件
echo "清理临时文件..."
rm -rf "${TEMP_DIR}"

# 最终结果统计
echo ""
echo "=================================================="
echo " 推理完成！"
echo "=================================================="
echo "输入文件: ${#INPUT_FILES[@]} 个"
echo "数据条数: ${TOTAL} 条"
echo "输出文件: ${OUTPUT_FILE}"
echo ""

# 显示输出文件信息
if [ -f "${OUTPUT_FILE}" ]; then
    OUTPUT_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    OUTPUT_LINES=$(wc -l < "${OUTPUT_FILE}")
    echo "输出文件大小: ${OUTPUT_SIZE}"
    echo "输出文件行数: ${OUTPUT_LINES}"
    
    # 显示前几行作为示例
    echo ""
    echo "输出文件前3行示例:"
    head -3 "${OUTPUT_FILE}" | while read line; do
        echo "  $(echo ${line} | cut -c1-100)..."
    done
else
    echo "警告: 输出文件未生成！"
fi

echo "=================================================="
echo "=================================================="


# 将推理结果转换为 tool.json
echo ""
echo "转换推理结果为 tool.json..."
python ./qwen3vl_harness/CogVLM/convert_tool_json.py "${OUTPUT_FILE}" "${TOOL_JSON_OUTPUT}"
echo "tool.json 已生成: ${TOOL_JSON_OUTPUT}"