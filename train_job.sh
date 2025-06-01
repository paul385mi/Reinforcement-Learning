#!/bin/bash
#SBATCH --job-name=jsp_gpu_train
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jsp_train_%j.out

echo "== Lade Module =="
module purge
module load python

echo "== Wechsle in das Projektverzeichnis =="
cd ~/Reinforcement-Learning

echo "== (Re)Erstelle venv auf GPU-Node =="
rm -rf venv
python3 -m venv venv
source venv/bin/activate

echo "== Installiere CUDA-kompatibles PyTorch =="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "== Installiere restliche Abhängigkeiten =="
pip install -r requirements.txt

echo "== Starte Training =="
python train_gym_ppo.py --episodes 1000