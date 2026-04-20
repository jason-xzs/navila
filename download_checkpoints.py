import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="a8cheng/navila-llama3-8b-8f",
    local_dir="/home/nvme04/public_data/xzs_data/navila",
    resume_download=True
)

snapshot_download(
    repo_id="a8cheng/navila-siglip-llama3-8b-v1.5-pretrain",
    local_dir="/home/nvme04/public_data/xzs_data/navila_base",
    resume_download=True
)

snapshot_download(
    repo_id="a8cheng/NaVILA-Dataset",
    repo_type="dataset",
    local_dir="/home/nvme04/public_data/xzs_data/navila_dataset",
    resume_download=True
)