export HF_ENDPOINT=https://hf-mirror.com

python -m huggingface_hub.commands.huggingface_cli download --resume-download a8cheng/navila-llama3-8b-8f --local-dir /home/nvme04/public_data/xzs_data/navila

python -m huggingface_hub.commands.huggingface_cli download --resume-download a8cheng/navila-siglip-llama3-8b-v1.5-pretrain --local-dir /home/nvme04/public_data/xzs_data/navila_base

python -m huggingface_hub.commands.huggingface_cli download --repo-type dataset --resume-download a8cheng/NaVILA-Dataset --local-dir /home/nvme04/public_data/xzs_data/navila_dataset