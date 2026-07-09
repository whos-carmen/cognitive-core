#!/bin/bash
set -euxo pipefail

exec > /var/log/user-data.log 2>&1

# 1. NVIDIA drivers + Docker
apt-get update -qq
apt-get install -y -qq nvidia-driver-550 nvidia-utils-550 nvidia-cuda-toolkit docker.io

systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

sleep 5
nvidia-smi

# 2. Pull training image
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# 3. Set up workspace with scripts
mkdir -p /workspace/models/code/logs /workspace/configs /workspace/train/outputs

# Download scripts from S3 (needs instance role or manual)
# Using AWS CLI (v2):
aws s3 cp s3://cognitive-core-training-337476940767/scripts/cognitive-core-scripts.tar.gz /tmp/scripts.tar.gz
tar xzf /tmp/scripts.tar.gz -C /workspace/

chown -R ubuntu:ubuntu /workspace

# 4. Create NVIDIA run script
cat > /workspace/run_sft.sh << 'SCRIPT'
#!/bin/bash
# Usage: bash run_sft.sh [smoke|full|dpo]
MODE="${1:-full}"
DOCKER="docker run --gpus all -it --rm -v /workspace:/workspace -w /workspace -e TOKENIZERS_PARALLELISM=false pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
case "$MODE" in
  smoke)
    $DOCKER python models/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/models/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_smoke \
      --max_steps 5 --bsz 1 --accum 1 --max_len 4096 --train_cap 4096
    ;;
  full)
    $DOCKER python models/code/train/sft.py \
      --model /workspace/models/merged \
      --train_file /workspace/models/dataset/train_v4.jsonl \
      --out /workspace/train/outputs/sft_claude_agent \
      --epochs 3 --neftune 5 --bsz 1 --accum 24 --lr 1e-5 --max_len 24576 --train_cap 24576
    ;;
  dpo)
    $DOCKER python models/code/train/dpo.py \
      --model /workspace/train/outputs/sft_claude_agent \
      --data /workspace/models/dataset/dpo_onpolicy_claude.jsonl \
      --out /workspace/train/outputs/final-cognitive-core \
      --beta 0.1 --lr 1e-6 --epochs 3 --accum 8
    ;;
esac
SCRIPT
chmod +x /workspace/run_sft.sh
chown ubuntu:ubuntu /workspace/run_sft.sh

echo "=== EC2 SETUP COMPLETE ==="
echo "SSH command: ssh ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Then: cd /workspace && bash run_sft.sh smoke  # test first"
echo "      bash run_sft.sh full                    # full training"
echo ""
echo "Before training, download models:"
echo "  cd /workspace && mkdir -p models/merged"
echo "  # Download merged model + dataset from HuggingFace or S3"
echo "  pip install huggingface-hub && huggingface-cli download ..."
