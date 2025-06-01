#!/bin/bash
#SBATCH --job-name=jsp_gpu_train
#SBATCH --partition=scc-gpu             # GPU-Partition der GWDG
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --cpus-per-task=4               # CPU-Kerne pro Task
#SBATCH --mem=16G                       # Arbeitsspeicher
#SBATCH --time=04:00:00                 # Max. Laufzeit
#SBATCH --output=jsp_train_%j.out       # Log-Datei mit Job-ID im Namen

# ------------------------------------------------------------------------------
# Vorbereitung
# ------------------------------------------------------------------------------

echo "== Lade Python-Modul =="
module purge
module load python

echo "== Wechsle ins Projektverzeichnis =="
cd ~/Reinforcement-Learning

echo "== Erstelle und aktiviere virtuelle Umgebung (falls nicht vorhanden) =="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "== Aktualisiere pip und installiere PyTorch mit CUDA-Unterstützung =="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "== Installiere restliche Abhängigkeiten =="
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "WARNUNG: requirements.txt nicht gefunden – überspringe."
fi

# ------------------------------------------------------------------------------
# Training starten
# ------------------------------------------------------------------------------

echo "== Starte Training mit GPU =="
python train_gym_ppo.py --episodes 1000

echo "== Job beendet =="