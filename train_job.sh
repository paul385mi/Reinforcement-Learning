#!/bin/bash
#SBATCH --job-name=jsp_gpu_train
#SBATCH --partition=scc-gpu
#SBATCH --gpus=A100:1                 # Spezifische GPU anfordern
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jsp_train_%j.out

# Umgebung vorbereiten
module purge
module load python

cd ~/Reinforcement-Learning

# Virtuelle Umgebung aktivieren
source venv/bin/activate

# CUDA-Verfügbarkeit prüfen
python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"

# Training starten
python train_gym_ppo.py --episodes 10

echo "== Job beendet =="