#!/bin/bash

# Update package list and install required tools
apt update
apt install -y python3 python3-venv python3-pip python3-dev nano screen

# Create a virtual environment named "venv"
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the required Python packages
pip install bitsandbytes==0.45.3 \
            datasets==3.4.1 \
            galore==0.9.2 \
            huggingface-hub==0.29.3 \
            loguru==0.7.3 \
            matplotlib==3.10.1 \
            numpy==2.2.4 \
            nvidia-cublas-cu12==12.4.5.8 \
            nvidia-cuda-cupti-cu12==12.4.127 \
            nvidia-cuda-nvrtc-cu12==12.4.127 \
            nvidia-cuda-runtime-cu12==12.4.127 \
            nvidia-cudnn-cu12==9.1.0.70 \
            nvidia-cufft-cu12==11.2.1.3 \
            nvidia-curand-cu12==10.3.5.147 \
            nvidia-cusolver-cu12==11.6.1.9 \
            nvidia-cusparse-cu12==12.3.1.170 \
            nvidia-cusparselt-cu12==0.6.2 \
            nvidia-nccl-cu12==2.21.5 \
            nvidia-nvjitlink-cu12==12.4.127 \
            nvidia-nvtx-cu12==12.4.127 \
            safetensors==0.5.3 \
            tensorly==0.9.0 \
            tokenizers==0.21.1 \
            torch==2.6.0 \
            tqdm==4.67.1 \
            transformers==4.49.0 \
            wandb==0.19.8 \
            nvitop

# Deactivate the virtual environment after installation
deactivate

echo "Setup complete! Virtual environment 'venv' has been created and packages installed."

echo "Run: wandb login and start a screen to run the experiments."
