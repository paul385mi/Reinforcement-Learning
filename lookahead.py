import gym
import numpy as np
from torch_ppo_agent import TorchPPOAgent
from gym_environment import JSPGymEnvironment
import json
import os

# Pfad zu deinem vortrainierten Modell
MODEL_PATH = "/Users/paulmill/Desktop/2025/Reinforcement Learning/Reinforcement/results/models/gym_ppo_model_20250414_144908.pt"

# Lade das JSP-Daten (ersetze dies durch deine tatsächlichen Daten)
def load_jsp_data():
    # Lade JSP-Daten aus einer JSON-Datei
    with open('/Users/paulmill/Desktop/2025/Reinforcement Learning/Reinforcement/data.json', 'r') as f:
        return json.load(f)

def main():
    # Lade JSP-Daten
    jsp_data = load_jsp_data()
    
    # Erstelle die Umgebung
    env = JSPGymEnvironment(jsp_data, enable_logging=True)
    
    # Überprüfe, ob das Modell existiert
    if not os.path.exists(MODEL_PATH):
        print(f"FEHLER: Modell nicht gefunden unter {MODEL_PATH}")
        available_models = [f for f in os.listdir(os.path.dirname(MODEL_PATH)) if f.endswith('.pt') or f.endswith('.zip')]
        if available_models:
            print(f"Verfügbare Modelle: {available_models}")
        return
    
    # Erstelle eine Instanz des TorchPPOAgent
    pretrained_model = TorchPPOAgent(len(jsp_data["jobs"]), jsp_data)
    
    # Lade das vortrainierte Modell
    try:
        pretrained_model.load_model(MODEL_PATH)
        print(f"Modell erfolgreich geladen: {MODEL_PATH}")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return
    
    # Setze die Umgebung zurück
    obs = env.reset()
    
    done = False
    total_reward = 0
    
    # Führe Schritte mit dem Lookahead-Mechanismus aus
    while not done:
        # Wähle eine Aktion basierend auf der aktuellen Beobachtung
        action, _ = pretrained_model.select_action(obs)
        
        # Führe die Aktion aus und übergebe das Modell für den Lookahead
        obs, reward, done, info = env.step(action, model=pretrained_model)
        
        total_reward += reward
        
        # Zeige Informationen an
        print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(f"Makespan: {info['makespan']}, Completed Jobs: {info['completed_jobs']}/{env.num_jobs}")
        
        if done:
            print("\nFinal Statistics:")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Makespan: {info['makespan']}")
            print(f"Completed Jobs: {info['completed_jobs']}/{env.num_jobs}")
            print(f"Met Deadlines: {info['met_deadlines']}/{env.num_jobs}")
            
            # Zeige detaillierte Reward-Statistiken an
            if 'reward_stats' in info:
                print("\nReward Components:")
                for component, value in info['reward_stats'].items():
                    print(f"  {component}: {value:.2f}")
            
            # Zeige detaillierte Statistiken an
            machine_stats = env.get_machine_utilization_stats()
            print("\nMachine Utilization:")
            for machine_id, stats in machine_stats.items():
                print(f"  Machine {machine_id}: {stats['utilization']:.2f} (Setup: {stats['setup_time_ratio']:.2f}, Idle: {stats['idle_time_ratio']:.2f})")
            
            job_stats = env.get_job_completion_stats()
            print("\nJob Completion:")
            print(f"  Average Completion Time: {job_stats['avg_completion_time']:.2f}")
            print(f"  Deadline Ratio: {job_stats['deadline_ratio']:.2f}")
            print(f"  High Priority Met Ratio: {job_stats['high_priority_met_ratio']:.2f}")

if __name__ == "__main__":
    main()