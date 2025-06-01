#!/bin/bash
#SBATCH --job-name=jsp_gpu_train
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jsp_train_%j.out

echo "== Lade Python- & CUDA-Module =="
module purge
module load python/3.10  # oder die Python-Version, die auf deinem Cluster verfügbar ist
module load cuda/11.8    # passend zu PyTorch-Version

echo "== Wechsle ins Projektverzeichnis =="
cd ~/Reinforcement-Learning

echo "== Installiere notwendige Pakete (falls nötig) =="
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
[ -f requirements.txt ] && pip install --user -r requirements.txt

echo "== Verifiziere CUDA-Zugriff =="
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"

echo "== Starte Training =="
python train_gym_ppo.py --episodes 10

echo "== Training beendet =="