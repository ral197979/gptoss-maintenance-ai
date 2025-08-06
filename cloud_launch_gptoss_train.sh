
# ======================================================
# 🚀 Lambda Labs: GPT-OSS-20B Fine-Tuning Environment
# ======================================================
# Step 1: Launch Lambda Labs A100 instance (Ubuntu 22.04)
# Step 2: SSH into your server

ssh ubuntu@your_lambda_public_ip

# Step 3: Setup environment
sudo apt update && sudo apt install -y git wget unzip python3-pip
pip install torch transformers datasets peft accelerate bitsandbytes

# Step 4: Upload your files (from local machine)
# Use SCP or rsync:
scp -i your-key.pem fine_tune_gptoss_lora.py train_data.json test_data.json ubuntu@your_lambda_ip:~/

# Step 5: Run training
python3 fine_tune_gptoss_lora.py



# ======================================================
# 🚀 RunPod: Auto-Deploy GPT-OSS Fine-Tuning Container
# ======================================================
# Step 1: Create a pod at https://www.runpod.io/ with:
# - Image: runpod/pytorch
# - GPU: A100 (20B fits with QLoRA)
# - Volumes: upload fine_tune_gptoss_lora.py, train/test JSON

# Step 2: Terminal Access → Open Shell

# Step 3: Install dependencies
pip install torch transformers datasets peft accelerate bitsandbytes

# Step 4: Launch training
python3 fine_tune_gptoss_lora.py
