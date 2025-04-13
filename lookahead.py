import gym
import numpy as np
from stable_baselines3 import PPO
from gym_environment import JSPGymEnvironment
import json
import os

# Pfad zu deinem vortrainierten Modell
MODEL_PATH = "/Users/timoelkers/Desktop/Transformer_Graph/Reinforcement-Learning/results/models/gym_ppo_model_20250407_214258.pt"

# Lade das JSP-Daten (ersetze dies durch deine tatsächlichen Daten)
def load_jsp_data():
    # Lade JSP-Daten aus einer JSON-Datei
    with open('/Users/timoelkers/Desktop/Transformer_Graph/Reinforcement-Learning/data/jsp_data.json', 'r') as f:
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
    
    # Lade das vortrainierte Modell
    try:
        pretrained_model = PPO.load(MODEL_PATH)
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
        action, _ = pretrained_model.predict(obs, deterministic=True)
        
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