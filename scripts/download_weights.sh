# pip install hf

# mkdir weights

# hf download yang1232009/CogVLM --local-dir weights/CogVLM

# hf download yang1232009/CogOmniControl --local-dir weights

# for Wan Video
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir weights/Wan2.2-T2V-A14B
hf download Kijai/WanVideo_comfy Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors --local-dir weights/loras

# for Benchmark
mkdir benchmark
hf download yang1232009/CogControlBench --local-dir benchmark --repo-type dataset