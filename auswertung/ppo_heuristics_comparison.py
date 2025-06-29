import json
import numpy as np
import sys
import os
import random

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_environment import JSPGymEnvironment
from torch_ppo_agent import TorchPPOAgent
import torch

class HeuristicDispatcher:
    """Einfache Implementierung klassischer Dispatching-Heuristiken"""
    
    def __init__(self, jsp_data):
        self.jsp_data = jsp_data
        self.jobs = jsp_data["jobs"]
        
    def fifo(self, valid_jobs, state):
        """First In, First Out - wähle ersten verfügbaren Job"""
        return valid_jobs[0] if valid_jobs else 0
    
    def filo(self, valid_jobs, state):
        """First In, Last Out - wähle letzten verfügbaren Job"""
        return valid_jobs[-1] if valid_jobs else 0
    
    def spt(self, valid_jobs, state):
        """Shortest Processing Time - wähle Job mit kürzester nächster Operation"""
        if not valid_jobs:
            return 0
        
        min_time = float('inf')
        best_job = valid_jobs[0]
        
        for job_idx in valid_jobs:
            progress = state['job_progress'][job_idx]
            if progress < len(self.jobs[job_idx]["operations"]):
                proc_time = self.jobs[job_idx]["operations"][progress]["processingTime"]
                if proc_time < min_time:
                    min_time = proc_time
                    best_job = job_idx
        
        return best_job
    
    def lpt(self, valid_jobs, state):
        """Longest Processing Time - wähle Job mit längster nächster Operation"""
        if not valid_jobs:
            return 0
        
        max_time = -1
        best_job = valid_jobs[0]
        
        for job_idx in valid_jobs:
            progress = state['job_progress'][job_idx]
            if progress < len(self.jobs[job_idx]["operations"]):
                proc_time = self.jobs[job_idx]["operations"][progress]["processingTime"]
                if proc_time > max_time:
                    max_time = proc_time
                    best_job = job_idx
        
        return best_job
    
    def earliest_due_date(self, valid_jobs, state):
        """Earliest Due Date - wähle Job mit frühester Deadline"""
        if not valid_jobs:
            return 0
        
        earliest_deadline = float('inf')
        best_job = valid_jobs[0]
        
        for job_idx in valid_jobs:
            deadline = self.jobs[job_idx]["deadline"]
            if deadline < earliest_deadline:
                earliest_deadline = deadline
                best_job = job_idx
        
        return best_job
    
    def critical_ratio(self, valid_jobs, state):
        """Critical Ratio - wähle Job mit niedrigstem Verhältnis von verbleibender Zeit zu verbleibender Arbeit"""
        if not valid_jobs:
            return 0
        
        current_time = state['current_time'][0]
        best_ratio = float('inf')
        best_job = valid_jobs[0]
        
        for job_idx in valid_jobs:
            deadline = self.jobs[job_idx]["deadline"]
            progress = state['job_progress'][job_idx]
            
            # Berechne verbleibende Bearbeitungszeit
            remaining_ops = self.jobs[job_idx]["operations"][progress:]
            remaining_time = sum(op["processingTime"] for op in remaining_ops)
            
            if remaining_time > 0:
                ratio = (deadline - current_time) / remaining_time
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_job = job_idx
        
        return best_job
    
    def slack_time(self, valid_jobs, state):
        """Slack Time - wähle Job mit geringster Pufferzeit"""
        if not valid_jobs:
            return 0
        
        current_time = state['current_time'][0]
        min_slack = float('inf')
        best_job = valid_jobs[0]
        
        for job_idx in valid_jobs:
            deadline = self.jobs[job_idx]["deadline"]
            progress = state['job_progress'][job_idx]
            
            # Berechne verbleibende Bearbeitungszeit
            remaining_ops = self.jobs[job_idx]["operations"][progress:]
            remaining_time = sum(op["processingTime"] for op in remaining_ops)
            
            slack = deadline - current_time - remaining_time
            if slack < min_slack:
                min_slack = slack
                best_job = job_idx
        
        return best_job
    
    def random_choice(self, valid_jobs, state):
        """Random - wähle zufällig einen verfügbaren Job"""
        if not valid_jobs:
            return 0
        return random.choice(valid_jobs)


def run_heuristic(env, heuristic_func, heuristic_name):
    """Führe eine Heuristik aus und sammle Ergebnisse"""
    # Erstelle JSP-Daten Dictionary für Dispatcher
    jsp_data = {"jobs": env.jobs, "machines": env.machines}
    dispatcher = HeuristicDispatcher(jsp_data)
    
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 1000:  # Sicherheitsgrenze
        # Finde gültige Jobs
        valid_jobs = []
        for job_idx in range(len(env.jobs)):
            if (state['job_progress'][job_idx] < len(env.jobs[job_idx]["operations"]) and
                env._check_predecessors(job_idx, state['job_progress'][job_idx])):
                valid_jobs.append(job_idx)
        
        if not valid_jobs:
            break
            
        # Wähle Aktion mit Heuristik
        action = heuristic_func(valid_jobs, state)
        
        # Führe Aktion aus
        next_state, reward, done, info = env.step(action)
        state = next_state
        steps += 1
    
    # Sammle Metriken
    makespan = max(state['machine_times'])
    
    # Maschinenauslastung
    if state['current_time'][0] > 0:
        utilization = sum(state['machine_times']) / (state['current_time'][0] * len(state['machine_times']))
    else:
        utilization = 0
    
    # Jobs vor Deadline abgeschlossen
    met_deadlines = 0
    total_delay = 0
    delayed_jobs = 0
    
    for job_idx, progress in enumerate(state['job_progress']):
        if progress >= len(env.jobs[job_idx]["operations"]):
            deadline = env.jobs[job_idx]["deadline"]
            if state['current_time'][0] <= deadline:
                met_deadlines += 1
            else:
                delay = state['current_time'][0] - deadline
                total_delay += delay
                delayed_jobs += 1
    
    avg_delay = total_delay / delayed_jobs if delayed_jobs > 0 else 0
    
    return {
        'heuristic': heuristic_name,
        'makespan': makespan,
        'utilization': utilization,
        'met_deadlines': met_deadlines,
        'total_jobs': len(env.jobs),
        'deadline_ratio': met_deadlines / len(env.jobs),
        'avg_delay': avg_delay,
        'steps': steps
    }


def run_ppo_agent(env, model_path):
    """Führe den trainierten PPO-Agent aus"""
    # Lade PPO-Agent mit korrekter JSP-Daten-Struktur
    jsp_data = {"jobs": env.jobs, "machines": env.machines}
    agent = TorchPPOAgent(len(env.jobs), jsp_data)
    agent.load_model(model_path)
    
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 1000:
        action, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        steps += 1
    
    # Sammle gleiche Metriken wie bei Heuristiken
    makespan = max(state['machine_times'])
    
    if state['current_time'][0] > 0:
        utilization = sum(state['machine_times']) / (state['current_time'][0] * len(state['machine_times']))
    else:
        utilization = 0
    
    met_deadlines = 0
    total_delay = 0
    delayed_jobs = 0
    
    for job_idx, progress in enumerate(state['job_progress']):
        if progress >= len(env.jobs[job_idx]["operations"]):
            deadline = env.jobs[job_idx]["deadline"]
            if state['current_time'][0] <= deadline:
                met_deadlines += 1
            else:
                delay = state['current_time'][0] - deadline
                total_delay += delay
                delayed_jobs += 1
    
    avg_delay = total_delay / delayed_jobs if delayed_jobs > 0 else 0
    
    return {
        'heuristic': 'PPO Agent',
        'makespan': makespan,
        'utilization': utilization,
        'met_deadlines': met_deadlines,
        'total_jobs': len(env.jobs),
        'deadline_ratio': met_deadlines / len(env.jobs),
        'avg_delay': avg_delay,
        'steps': steps
    }


def compare_methods(jsp_data_path, model_path, num_runs=5):
    """Vergleiche PPO-Agent mit Heuristiken über mehrere Läufe"""
    
    # Lade JSP-Daten
    with open(jsp_data_path, 'r') as f:
        jsp_data = json.load(f)
    
    # Erstelle Environment
    env = JSPGymEnvironment(jsp_data, enable_logging=False)
    
    # Definiere Heuristiken - verwende jsp_data statt env.jobs
    dispatcher = HeuristicDispatcher(jsp_data)
    
    # Definiere Heuristiken (Random hinzugefügt)
    heuristics = [
        (dispatcher.fifo, "FIFO"),
        (dispatcher.filo, "FILO"),
        (dispatcher.spt, "SPT"),
        (dispatcher.lpt, "LPT"),
        (dispatcher.earliest_due_date, "Earliest Due Date"),
        (dispatcher.critical_ratio, "Critical Ratio"),
        (dispatcher.slack_time, "Slack Time"),
        (dispatcher.random_choice, "Random")  # Neue Random-Heuristik
    ]
    
    results = []
    
    print("Vergleiche PPO-Agent mit Dispatching-Heuristiken (inkl. Random)...\n")
    
    # Teste PPO-Agent
    print("Teste PPO-Agent...")
    ppo_results = []
    for run in range(num_runs):
        result = run_ppo_agent(env, model_path)
        ppo_results.append(result)
    
    # Berechne Durchschnitt für PPO
    ppo_avg = {
        'heuristic': 'PPO Agent',
        'makespan': np.mean([r['makespan'] for r in ppo_results]),
        'utilization': np.mean([r['utilization'] for r in ppo_results]),
        'deadline_ratio': np.mean([r['deadline_ratio'] for r in ppo_results]),
        'avg_delay': np.mean([r['avg_delay'] for r in ppo_results]),
        'steps': np.mean([r['steps'] for r in ppo_results])
    }
    results.append(ppo_avg)
    
    # Teste jede Heuristik
    for heuristic_func, name in heuristics:
        print(f"Teste {name}...")
        heuristic_results = []
        for run in range(num_runs):
            # Setze Seed für Random-Heuristik für reproduzierbare Ergebnisse
            if name == "Random":
                random.seed(run * 42)  # Unterschiedliche Seeds pro Run
            result = run_heuristic(env, heuristic_func, name)
            heuristic_results.append(result)
        
        # Berechne Durchschnitt für Heuristik
        heuristic_avg = {
            'heuristic': name,
            'makespan': np.mean([r['makespan'] for r in heuristic_results]),
            'utilization': np.mean([r['utilization'] for r in heuristic_results]),
            'deadline_ratio': np.mean([r['deadline_ratio'] for r in heuristic_results]),
            'avg_delay': np.mean([r['avg_delay'] for r in heuristic_results]),
            'steps': np.mean([r['steps'] for r in heuristic_results])
        }
        results.append(heuristic_avg)
    
    # Zeige Ergebnisse
    print("\n" + "="*80)
    print("VERGLEICHSERGEBNISSE:")
    print("="*80)
    
    print(f"{'Methode':<20} {'Makespan':<12} {'Auslastung':<12} {'Deadline %':<12} {'Avg Delay':<12}")
    print("-" * 80)
    
    ppo_makespan = results[0]['makespan']
    ppo_utilization = results[0]['utilization']
    ppo_deadline_ratio = results[0]['deadline_ratio']
    
    for result in results:
        print(f"{result['heuristic']:<20} "
              f"{result['makespan']:<12.1f} "
              f"{result['utilization']:<12.2f} "
              f"{result['deadline_ratio']*100:<12.1f} "
              f"{result['avg_delay']:<12.1f}")
    
    print("\n" + "="*80)
    print("PPO vs HEURISTIKEN VERGLEICH:")
    print("="*80)
    
    for result in results[1:]:  # Skip PPO (first result)
        makespan_improvement = ((result['makespan'] - ppo_makespan) / result['makespan']) * 100
        utilization_improvement = ((ppo_utilization - result['utilization']) / result['utilization']) * 100 if result['utilization'] > 0 else 0
        deadline_improvement = ((ppo_deadline_ratio - result['deadline_ratio']) / result['deadline_ratio']) * 100 if result['deadline_ratio'] > 0 else 0
        
        print(f"\nPPO vs {result['heuristic']}:")
        print(f"  Makespan: {makespan_improvement:+.1f}% {'(besser)' if makespan_improvement < 0 else '(schlechter)'}")
        print(f"  Auslastung: {utilization_improvement:+.1f}% {'(besser)' if utilization_improvement > 0 else '(schlechter)'}")
        print(f"  Deadline-Rate: {deadline_improvement:+.1f}% {'(besser)' if deadline_improvement > 0 else '(schlechter)'}")
    
    return results


if __name__ == "__main__":
    # Pfade definieren - relativ zum Hauptverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    # Prüfe ob Dateien existieren
    if not os.path.exists(jsp_data_path):
        print(f"Fehler: data.json nicht gefunden unter {jsp_data_path}")
        print("Bitte stelle sicher, dass data.json im Hauptverzeichnis liegt.")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Fehler: PPO-Modell nicht gefunden unter {model_path}")
        print("Verfügbare Modelle:")
        models_dir = os.path.join(main_dir, "results/models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pt'):
                    print(f"  {file}")
        sys.exit(1)
    
    print(f"Verwende JSP-Daten: {jsp_data_path}")
    print(f"Verwende PPO-Modell: {model_path}")
    print()
    
    # Vergleich durchführen
    results = compare_methods(jsp_data_path, model_path, num_runs=3)
    
    print(f"\nVergleich abgeschlossen! Ergebnisse basieren auf {3} Läufen pro Methode.")