#!/bin/bash
#SBATCH --partition=killable
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=mamba_2080
#SBATCH --output=logs/mamba_2080_%j.out
#SBATCH --error=logs/mamba_2080_%j.err

# הפעלת הסביבה
source /vol/joberant_nobck/data/NLP_368307701_2526a/inbalmoryles/envs/zamba-env/bin/activate

# שלב הקומפילציה (חד פעמי - אפשר להוריד את זה אחרי שהריצה הראשונה מצליחה)
echo "[+] Re-compiling Mamba kernels for RTX 3090..."
pip install --no-cache-dir causal-conv1d mamba-ssm --force-reinstall

# בדיקה ש-PyTorch רואה את הכרטיס הנכון
python -c "import torch; print(f'Using GPU: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})')"

# הרצת הניסוי
python run_layer_experiments.py \
    --model-id "tiiuae/falcon-mamba-7b-instruct" \
    --num-samples 50 \
    --trust-remote-code
