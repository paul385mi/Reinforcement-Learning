#!/bin/bash
#SBATCH -J ppo-training
#SBATCH -p scc-gpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log

# Aktivieren der virtuellen Umgebung mit absolutem Pfad
source /user/paul.mill/u17597/Reinforcement-Learning/venv/bin/activate

# Debug: Ausgabe des verwendeten Python-Interpreters
echo "Python path:" $(which python)
python --version

# Start des Trainings
python train_gym_ppo.py --episodes 1
