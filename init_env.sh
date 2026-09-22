
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
# for H20 cu126_torch2.8.0
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu126_torch280
pip install psutil
pip install flash_attn==2.7.4.post1 --no-build-isolation


pip install accelerate peft
pip install -r ./requirements_torch-2.8.0_cu126.txt --no-deps
# pip uninstall xformers -y