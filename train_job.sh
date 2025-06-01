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
module load python/3.10
module load cuda/11.8

echo "== Erstelle/aktiviere virtuelle Umgebung =="
cd ~/Reinforcement-Learning
python3 -m venv venv
source venv/bin/activate

echo "== Installiere Pakete =="
pip install --upgrade pip
pip install numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

echo "== Starte Training =="
python train_gym_ppo.py --episodes 10

echo "== Job beendet =="