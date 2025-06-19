#!/bin/bash
#SBATCH -J ppo-training
#SBATCH -p scc-gpu
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log

source venv/bin/activate  # falls du eine virtuelle Umgebung nutzt
python3 train_gym_ppo.py --episodes 1
