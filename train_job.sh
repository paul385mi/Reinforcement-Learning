#!/bin/bash
#SBATCH --job-name=jsp_gpu_train
#SBATCH --partition=scc-gpu         # GPU-Partition der GWDG
#SBATCH --gres=gpu:1                # 1 GPU anfordern
#SBATCH --cpus-per-task=4           # Anzahl CPU-Kerne
#SBATCH --mem=16G                   # RAM
#SBATCH --time=04:00:00             # Maximale Laufzeit (hh:mm:ss)
#SBATCH --output=jsp_train_%j.out   # Logdatei (%j = Job-ID)

# Umgebung vorbereiten
module load python
source ~/Reinforcement-Learning/venv/bin/activate
cd ~/Reinforcement-Learning

# Training starten
python train_gym_ppo.py --episodes 1000
