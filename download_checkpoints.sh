export HF_ENDPOINT=https://hf-mirror.com

hf download a8cheng/navila-llama3-8b-8f --local-dir /root/autodl-tmp/navila

huggingface-cli download --repo-type dataset --resume-download a8cheng/NaVILA-Dataset --local-dir /root/autodl-tmp/navila_data