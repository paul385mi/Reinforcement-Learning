# JSP Reinforcement Learning (PPO)

PPO-Agent für Job Shop Scheduling Probleme.

## Setup

```bash
# Python & Umgebung
brew install python3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

### Training
```bash
# Standard Training
python train_gym_ppo.py

# Mit Parametern
python train_gym_ppo.py --episodes 100 --batch-size 64

# Langes Training
python train_gym_ppo.py --episodes 500
```

### Testing
```bash
# Modell testen
python train_gym_ppo.py --test-only --model-path results/models/gym_ppo_model_*.pt
```

### Evaluation
```bash
# PPO vs. Heuristiken Vergleich
cd auswertung
python combined.py
```

### Cluster
```bash
# SLURM Job
sbatch run_training.sh

# Background
nohup python train_gym_ppo.py --episodes 1000 > training.log 2>&1 &
```

## Parameter

| Flag | Default | Beschreibung |
|------|---------|-------------|
| `--episodes` | 50 | Trainingsepisoden |
| `--batch-size` | 32 | Batch-Größe |
| `--save-interval` | 50 | Speicherintervall |
| `--test-only` | False | Nur testen |
| `--model-path` | - | Modellpfad |

## Outputs

- `results/models/` - Trainierte Modelle
- `results/images/` - Lernkurven
- `auswertung/images/` - Evaluierungsdiagramme
- `logs/` - Training Logs

## Troubleshooting

```bash
# Python Check
python3 --version

# Umgebung Reset
rm -rf venv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# GPU Check
python -c "import torch; print(torch.cuda.is_available())"

# Cleanup
rm -rf logs/old_* results/models/gym_ppo_model_202*.pt
```
    