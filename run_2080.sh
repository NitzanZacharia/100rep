#!/bin/bash
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=mamba_2080
#SBATCH --output=logs/mamba_2080_%j.out
#SBATCH --error=logs/mamba_2080_%j.err

# Init Conda
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/miniconda3/etc/profile.d/conda.sh

# Activate the env
conda activate /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/envs/zamba-env

# Verify GPU (just to be sure)
python -c "import torch; print(f'Running on GPU: {torch.cuda.get_device_name(0)}')"

# Run the actual experiment
python run_layer_experiments.py \
    --model-id "tiiuae/falcon-mamba-7b-instruct" \
    --num-samples 50 \
    --trust-remote-code
