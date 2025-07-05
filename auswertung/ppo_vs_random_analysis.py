#!/usr/bin/env python3
"""
Korrigierter Vergleich: PPO Agent vs. Random Strategy
Behebt JSON-Fehler und zeigt realistische Ergebnisse
"""

import json
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_environment import JSPGymEnvironment
from torch_ppo_agent import TorchPPOAgent
import torch

class RandomDispatcher:
    """Einfache Random-Strategie für Vergleich"""
    
    def __init__(self, jsp_data):
        self.jsp_data = jsp_data
        self.jobs = jsp_data["jobs"]
        
    def random_choice(self, valid_jobs, state):
        """Random - wähle zufällig einen verfügbaren Job"""
        if not valid_jobs:
            return 0
        return random.choice(valid_jobs)

def convert_numpy_types(obj):
    """Konvertiere NumPy-Typen für JSON-Serialisierung"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def run_random_strategy(env, num_runs=10):
    """Führe Random-Strategie mehrfach aus"""
    jsp_data = {"jobs": env.jobs, "machines": env.machines}
    dispatcher = RandomDispatcher(jsp_data)
    
    results = []
    
    for run in range(num_runs):
        # Setze verschiedene Seeds für echte Randomness
        random.seed(run * 123 + 42)
        
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            # Finde gültige Jobs
            valid_jobs = []
            for job_idx in range(len(env.jobs)):
                if (state['job_progress'][job_idx] < len(env.jobs[job_idx]["operations"]) and
                    env._check_predecessors(job_idx, state['job_progress'][job_idx])):
                    valid_jobs.append(job_idx)
            
            if not valid_jobs:
                break
                
            # Wähle zufällige Aktion
            action = dispatcher.random_choice(valid_jobs, state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
        
        # Sammle Metriken
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
        
        results.append({
            'makespan': float(makespan),
            'utilization': float(utilization),
            'deadline_ratio': float(met_deadlines / len(env.jobs)),
            'avg_delay': float(avg_delay),
            'steps': int(steps)
        })
    
    # Berechne Statistiken
    return {
        'method': 'Random Strategy',
        'makespan_mean': float(np.mean([r['makespan'] for r in results])),
        'makespan_std': float(np.std([r['makespan'] for r in results])),
        'utilization_mean': float(np.mean([r['utilization'] for r in results])),
        'utilization_std': float(np.std([r['utilization'] for r in results])),
        'deadline_ratio_mean': float(np.mean([r['deadline_ratio'] for r in results])),
        'deadline_ratio_std': float(np.std([r['deadline_ratio'] for r in results])),
        'avg_delay_mean': float(np.mean([r['avg_delay'] for r in results])),
        'avg_delay_std': float(np.std([r['avg_delay'] for r in results])),
        'all_results': results
    }

def run_ppo_agent(env, model_path, num_runs=10):
    """Führe PPO-Agent mehrfach aus"""
    jsp_data = {"jobs": env.jobs, "machines": env.machines}
    agent = TorchPPOAgent(len(env.jobs), jsp_data)
    agent.load_model(model_path)
    
    results = []
    
    for run in range(num_runs):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1
        
        # Sammle Metriken
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
        
        results.append({
            'makespan': float(makespan),
            'utilization': float(utilization),
            'deadline_ratio': float(met_deadlines / len(env.jobs)),
            'avg_delay': float(avg_delay),
            'steps': int(steps)
        })
    
    return {
        'method': 'PPO Agent',
        'makespan_mean': float(np.mean([r['makespan'] for r in results])),
        'makespan_std': float(np.std([r['makespan'] for r in results])),
        'utilization_mean': float(np.mean([r['utilization'] for r in results])),
        'utilization_std': float(np.std([r['utilization'] for r in results])),
        'deadline_ratio_mean': float(np.mean([r['deadline_ratio'] for r in results])),
        'deadline_ratio_std': float(np.std([r['deadline_ratio'] for r in results])),
        'avg_delay_mean': float(np.mean([r['avg_delay'] for r in results])),
        'avg_delay_std': float(np.std([r['avg_delay'] for r in results])),
        'all_results': results
    }

def create_realistic_comparison(ppo_results, random_results):
    """Erstelle realistische Visualisierung basierend auf tatsächlichen Ergebnissen"""
    
    # Berechne Verbesserungen (können negativ sein)
    makespan_improvement = ((random_results['makespan_mean'] - ppo_results['makespan_mean']) / random_results['makespan_mean']) * 100
    utilization_improvement = ((ppo_results['utilization_mean'] - random_results['utilization_mean']) / random_results['utilization_mean']) * 100
    delay_improvement = ((random_results['avg_delay_mean'] - ppo_results['avg_delay_mean']) / random_results['avg_delay_mean']) * 100 if random_results['avg_delay_mean'] > 0 else 0
    
    # Bestimme welche Methode besser ist
    ppo_better_makespan = makespan_improvement > 0
    ppo_better_utilization = utilization_improvement > 0
    ppo_better_delay = delay_improvement > 0
    
    # Setze Stil ohne Emojis
    plt.style.use('default')
    
    # Erstelle 2x2 Layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Agent vs. Random Strategy - Objektiver Vergleich', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Farben: Dynamisch basierend auf Performance
    success_color = '#2E8B57'  # Grün für bessere Performance
    poor_color = '#CD5C5C'     # Rot für schlechtere Performance
    
    # 1. Makespan Vergleich
    methods = ['PPO Agent', 'Random Strategy']
    makespans = [ppo_results['makespan_mean'], random_results['makespan_mean']]
    makespan_errors = [ppo_results['makespan_std'], random_results['makespan_std']]
    
    colors1 = [success_color if ppo_better_makespan else poor_color, 
               poor_color if ppo_better_makespan else success_color]
    
    bars1 = ax1.bar(methods, makespans, yerr=makespan_errors, capsize=10,
                    color=colors1, alpha=0.8, edgecolor='black', linewidth=2)
    
    title1 = f'Makespan Vergleich\n'
    if ppo_better_makespan:
        title1 += f'PPO {abs(makespan_improvement):.1f}% besser'
    else:
        title1 += f'Random {abs(makespan_improvement):.1f}% besser'
    
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Makespan (Minuten)', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Werte anzeigen
    for bar, value, error in zip(bars1, makespans, makespan_errors):
        ax1.text(bar.get_x() + bar.get_width()/2., value + error + 20, 
                f'{value:.0f}±{error:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Winner Badge
    winner_idx = 0 if ppo_better_makespan else 1
    ax1.text(winner_idx, makespans[winner_idx] + makespan_errors[winner_idx] + 100, 
             'WINNER', ha='center', va='bottom', 
             fontweight='bold', color='white', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=success_color, alpha=0.9))
    
    # 2. Maschinenauslastung
    utilizations = [ppo_results['utilization_mean'], random_results['utilization_mean']]
    util_errors = [ppo_results['utilization_std'], random_results['utilization_std']]
    
    colors2 = [success_color if ppo_better_utilization else poor_color, 
               poor_color if ppo_better_utilization else success_color]
    
    bars2 = ax2.bar(methods, utilizations, yerr=util_errors, capsize=10,
                    color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
    
    title2 = f'Maschinenauslastung\n'
    if ppo_better_utilization:
        title2 += f'PPO {abs(utilization_improvement):.2f}% effizienter'
    else:
        title2 += f'Random {abs(utilization_improvement):.2f}% effizienter'
    
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.set_ylabel('Auslastung (0-1)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Dynamische y-Achse
    min_util = min(utilizations) - max(util_errors) - 0.01
    max_util = max(utilizations) + max(util_errors) + 0.01
    ax2.set_ylim(min_util, max_util)
    
    # Werte anzeigen
    for bar, value, error in zip(bars2, utilizations, util_errors):
        ax2.text(bar.get_x() + bar.get_width()/2., value + error + (max_util-min_util)*0.02, 
                f'{value:.3f}±{error:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Winner Badge
    winner_idx = 0 if ppo_better_utilization else 1
    ax2.text(winner_idx, utilizations[winner_idx] + util_errors[winner_idx] + (max_util-min_util)*0.05, 
             'WINNER', ha='center', va='bottom', 
             fontweight='bold', color='white', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=success_color, alpha=0.9))
    
    # 3. Verspätung
    delays = [ppo_results['avg_delay_mean'], random_results['avg_delay_mean']]
    delay_errors = [ppo_results['avg_delay_std'], random_results['avg_delay_std']]
    
    colors3 = [success_color if ppo_better_delay else poor_color, 
               poor_color if ppo_better_delay else success_color]
    
    bars3 = ax3.bar(methods, delays, yerr=delay_errors, capsize=10,
                    color=colors3, alpha=0.8, edgecolor='black', linewidth=2)
    
    title3 = f'Durchschnittliche Verspätung\n'
    if ppo_better_delay:
        title3 += f'PPO {abs(delay_improvement):.1f}% weniger Verspätung'
    else:
        title3 += f'Random {abs(delay_improvement):.1f}% weniger Verspätung'
    
    ax3.set_title(title3, fontsize=14, fontweight='bold')
    ax3.set_ylabel('Verspätung (Minuten)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Werte anzeigen
    for bar, value, error in zip(bars3, delays, delay_errors):
        ax3.text(bar.get_x() + bar.get_width()/2., value + error + 20, 
                f'{value:.0f}±{error:.0f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Winner Badge
    winner_idx = 0 if ppo_better_delay else 1
    ax3.text(winner_idx, delays[winner_idx] + delay_errors[winner_idx] + 100, 
             'WINNER', ha='center', va='bottom', 
             fontweight='bold', color='white', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=success_color, alpha=0.9))
    
    # 4. Objektive Zusammenfassung
    ax4.axis('off')
    
    # Berechne Gesamtperformance
    ppo_wins = sum([ppo_better_makespan, ppo_better_utilization, ppo_better_delay])
    
    if ppo_wins >= 2:
        overall_winner = "PPO Agent"
        winner_color = success_color
        conclusion = "PPO übertrifft Random in der Mehrheit der Metriken"
    else:
        overall_winner = "Random Strategy"
        winner_color = poor_color
        conclusion = "Random übertrifft PPO in der Mehrheit der Metriken"
    
    summary_text = f"""OBJEKTIVE ANALYSE

Makespan: {'PPO' if ppo_better_makespan else 'Random'} ({abs(makespan_improvement):.1f}% besser)

Effizienz: {'PPO' if ppo_better_utilization else 'Random'} ({abs(utilization_improvement):.2f}% besser)

Verspätung: {'PPO' if ppo_better_delay else 'Random'} ({abs(delay_improvement):.1f}% besser)

GESAMTSIEGER: {overall_winner}
({ppo_wins}/3 Metriken gewonnen)

FAZIT: {conclusion}

Basiert auf {len(ppo_results['all_results'])} Testläufen pro Methode
    """
    
    ax4.text(0.1, 0.9, summary_text, fontsize=13, fontweight='bold',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    # Champion Symbol
    ax4.text(0.8, 0.5, 'WINNER', fontsize=40, ha='center', va='center', 
             color=winner_color, fontweight='bold')
    ax4.text(0.8, 0.3, overall_winner, fontsize=14, fontweight='bold', 
             ha='center', va='center', color=winner_color)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

def save_results(ppo_results, random_results):
    """Speichere Ergebnisse und Visualisierung"""
    
    # Erstelle Verzeichnis
    os.makedirs('auswertung/images', exist_ok=True)
    
    # Erstelle Visualisierung
    fig = create_realistic_comparison(ppo_results, random_results)
    
    # Speichere Bild
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = f'auswertung/images/ppo_vs_random_objective_{timestamp}.png'
    fig.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Konvertiere alle NumPy-Typen für JSON
    ppo_clean = convert_numpy_types({k: v for k, v in ppo_results.items() if k != 'all_results'})
    random_clean = convert_numpy_types({k: v for k, v in random_results.items() if k != 'all_results'})
    
    # Speichere auch Daten als JSON
    results_data = {
        'timestamp': timestamp,
        'ppo_results': ppo_clean,
        'random_results': random_clean,
        'improvements': {
            'makespan_improvement_percent': float(((random_results['makespan_mean'] - ppo_results['makespan_mean']) / random_results['makespan_mean']) * 100),
            'utilization_improvement_percent': float(((ppo_results['utilization_mean'] - random_results['utilization_mean']) / random_results['utilization_mean']) * 100),
            'delay_improvement_percent': float(((random_results['avg_delay_mean'] - ppo_results['avg_delay_mean']) / random_results['avg_delay_mean']) * 100) if random_results['avg_delay_mean'] > 0 else 0.0
        }
    }
    
    json_path = f'auswertung/images/ppo_vs_random_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return image_path, json_path

def main():
    """Hauptfunktion"""
    print("OBJEKTIVER VERGLEICH: PPO Agent vs. Random Strategy")
    print("=" * 60)
    
    # Pfade definieren
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    # Prüfe Dateien
    if not os.path.exists(jsp_data_path):
        print(f"FEHLER: data.json nicht gefunden unter {jsp_data_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"FEHLER: PPO-Modell nicht gefunden unter {model_path}")
        return
    
    print(f"JSP-Daten: {jsp_data_path}")
    print(f"PPO-Modell: {model_path}")
    print()
    
    # Lade Daten und erstelle Environment
    with open(jsp_data_path, 'r') as f:
        jsp_data = json.load(f)
    
    env = JSPGymEnvironment(jsp_data, enable_logging=False)
    
    num_runs = 20  # Mehr Läufe für bessere Statistiken
    
    print(f"Teste PPO Agent ({num_runs} Läufe)...")
    ppo_results = run_ppo_agent(env, model_path, num_runs)
    
    print(f"Teste Random Strategy ({num_runs} Läufe)...")
    random_results = run_random_strategy(env, num_runs)
    
    print("\nOBJEKTIVE ERGEBNISSE:")
    print("=" * 60)
    
    # Berechne Verbesserungen
    makespan_improvement = ((random_results['makespan_mean'] - ppo_results['makespan_mean']) / random_results['makespan_mean']) * 100
    utilization_improvement = ((ppo_results['utilization_mean'] - random_results['utilization_mean']) / random_results['utilization_mean']) * 100
    delay_improvement = ((random_results['avg_delay_mean'] - ppo_results['avg_delay_mean']) / random_results['avg_delay_mean']) * 100 if random_results['avg_delay_mean'] > 0 else 0
    
    print(f"Makespan:")
    print(f"  PPO:    {ppo_results['makespan_mean']:.1f}±{ppo_results['makespan_std']:.1f}")
    print(f"  Random: {random_results['makespan_mean']:.1f}±{random_results['makespan_std']:.1f}")
    print(f"  Unterschied: {makespan_improvement:+.1f}% ({'PPO besser' if makespan_improvement > 0 else 'Random besser'})")
    print()
    
    print(f"Auslastung:")
    print(f"  PPO:    {ppo_results['utilization_mean']:.3f}±{ppo_results['utilization_std']:.3f}")
    print(f"  Random: {random_results['utilization_mean']:.3f}±{random_results['utilization_std']:.3f}")
    print(f"  Unterschied: {utilization_improvement:+.2f}% ({'PPO besser' if utilization_improvement > 0 else 'Random besser'})")
    print()
    
    print(f"Verspätung:")
    print(f"  PPO:    {ppo_results['avg_delay_mean']:.1f}±{ppo_results['avg_delay_std']:.1f}")
    print(f"  Random: {random_results['avg_delay_mean']:.1f}±{random_results['avg_delay_std']:.1f}")
    print(f"  Unterschied: {delay_improvement:+.1f}% ({'PPO besser' if delay_improvement > 0 else 'Random besser'})")
    print()
    
    # Speichere Ergebnisse
    print("Speichere objektive Analyse...")
    image_path, json_path = save_results(ppo_results, random_results)
    
    print("ANALYSE ABGESCHLOSSEN!")
    print("=" * 60)
    print(f"Visualisierung: {image_path}")
    print(f"Daten: {json_path}")
    print()
    print("Die Ergebnisse zeigen die tatsächliche Performance ohne Verzerrung.")

if __name__ == "__main__":
    main()