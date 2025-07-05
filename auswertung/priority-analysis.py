import json
import os
import pandas as pd
import torch
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_ppo_agent import TorchPPOAgent
from gym_environment import JSPGymEnvironment
from datetime import datetime



def analyze_ppo_priority_behavior(jsp_data_path, model_path, output_dir="analysis"):
    """
    Analysiert das Prioritätenverhalten des PPO-Agenten Schritt für Schritt.
    
    Args:
        jsp_data_path: Pfad zur JSP-Datendatei
        model_path: Pfad zum trainierten Modell
        output_dir: Ausgabeverzeichnis für die CSV-Datei
    """
    
    # Daten laden
    with open(jsp_data_path, 'r') as f:
        jsp_data = json.load(f)
    
    # Environment und Agent initialisieren
    env = JSPGymEnvironment(jsp_data, enable_logging=False)
    agent = TorchPPOAgent(len(jsp_data["jobs"]), jsp_data)
    
    # Modell laden
    agent.load_model(model_path)
    print(f"Modell geladen: {model_path}")
    
    # Analyse durchführen
    state = env.reset()
    done = False
    step = 0
    
    # Datensammlung für CSV
    analysis_data = []
    
    # Job-Informationen für schnellen Zugriff
    job_info = {job["id"]: {"priority": job["priority"], "deadline": job["deadline"]} 
                for job in jsp_data["jobs"]}
    
    print("Starte Schritt-für-Schritt Analyse...")
    print("=" * 80)
    
    while not done:
        step += 1
        
        # Verfügbare Jobs ermitteln
        valid_jobs = []
        for job_idx in range(len(jsp_data["jobs"])):
            if state['valid_actions_mask'][job_idx] == 1:
                job_id = agent.idx_to_job_id[job_idx]
                job_priority = job_info[job_id]["priority"]
                job_deadline = job_info[job_id]["deadline"]
                valid_jobs.append({
                    'job_idx': job_idx,
                    'job_id': job_id,
                    'priority': job_priority,
                    'deadline': job_deadline
                })
        
        # Sortiere verfügbare Jobs nach Priorität (höchste zuerst)
        valid_jobs.sort(key=lambda x: x['priority'], reverse=True)
        
        # Agent-Entscheidung
        action, action_prob = agent.select_action(state)
        selected_job_id = agent.idx_to_job_id[action]
        selected_job_info = job_info[selected_job_id]
        
        # Aktuelle Operation ermitteln
        current_op_idx = state['job_progress'][action]
        current_op = jsp_data["jobs"][action]["operations"][current_op_idx]
        
        # Prioritätsrang des gewählten Jobs
        priority_rank = next((i+1 for i, job in enumerate(valid_jobs) 
                            if job['job_id'] == selected_job_id), len(valid_jobs))
        
        # Ist die höchste Priorität gewählt?
        is_highest_priority = priority_rank == 1
        highest_priority_available = valid_jobs[0]['priority'] if valid_jobs else 0
        
        # Schritt ausführen
        next_state, reward, done, info = env.step(action)
        
        # Daten für CSV sammeln
        row_data = {
            'step': step,
            'selected_job_id': selected_job_id,
            'selected_job_priority': selected_job_info['priority'],
            'selected_job_deadline': selected_job_info['deadline'],
            'operation_id': current_op['id'],
            'machine_id': current_op['machineId'],
            'processing_time': current_op['processingTime'],
            'material': current_op['material'],
            'action_probability': action_prob,
            'reward': reward,
            'current_time': state['current_time'][0],
            'makespan': max(state['machine_times']),
            'total_available_jobs': len(valid_jobs),
            'highest_priority_available': highest_priority_available,
            'priority_rank_selected': priority_rank,
            'is_highest_priority_selected': is_highest_priority,
            'setup_time': info.get('setup_time', 0),
            'job_completed': info.get('job_completed', False)
        }
        
        # Verfügbare Jobs als String (für Übersicht)
        available_jobs_str = "; ".join([f"{job['job_id']}(P{job['priority']})" 
                                       for job in valid_jobs])
        row_data['available_jobs'] = available_jobs_str
        
        analysis_data.append(row_data)
        
        # Konsolen-Ausgabe
        print(f"Schritt {step:2d}: Gewählt: {selected_job_id} (P{selected_job_info['priority']}) "
              f"- Rang: {priority_rank}/{len(valid_jobs)} "
              f"- Höchste Priorität: {'✓' if is_highest_priority else '✗'}")
        print(f"          Verfügbar: {available_jobs_str}")
        print(f"          Operation: {current_op['id']} auf {current_op['machineId']} "
              f"({current_op['processingTime']}min)")
        print(f"          Belohnung: {reward:.2f}, Setup: {info.get('setup_time', 0):.1f}min")
        print("-" * 80)
        
        # State für nächsten Schritt
        state = next_state
    
    # DataFrame erstellen und speichern
    df = pd.DataFrame(analysis_data)
    
    # Ausgabeverzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV-Datei speichern
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(output_dir, f'ppo_priority_analysis_{timestamp}.csv')
    df.to_csv(csv_filename, index=False)
    
    # Zusammenfassung erstellen
    print("\n" + "=" * 80)
    print("ANALYSE ZUSAMMENFASSUNG")
    print("=" * 80)
    
    total_steps = len(analysis_data)
    highest_priority_selected = df['is_highest_priority_selected'].sum()
    priority_selection_rate = (highest_priority_selected / total_steps) * 100
    
    print(f"Gesamte Schritte: {total_steps}")
    print(f"Höchste Priorität gewählt: {highest_priority_selected}/{total_steps} ({priority_selection_rate:.1f}%)")
    print(f"Finale Makespan: {df['makespan'].iloc[-1]:.1f}")
    print(f"Durchschnittliche Belohnung: {df['reward'].mean():.2f}")
    
    # Prioritätsverteilung der Auswahlen
    print("\nPrioritätsverteilung der Auswahlen:")
    priority_distribution = df['selected_job_priority'].value_counts().sort_index(ascending=False)
    for priority, count in priority_distribution.items():
        percentage = (count / total_steps) * 100
        print(f"  Priorität {priority}: {count} mal ({percentage:.1f}%)")
    
    # Durchschnittlicher Prioritätsrang
    avg_priority_rank = df['priority_rank_selected'].mean()
    print(f"\nDurchschnittlicher Prioritätsrang: {avg_priority_rank:.2f}")
    print(f"(1 = höchste Priorität, höhere Zahlen = niedrigere Priorität)")
    
    print(f"\nDetaillierte Analyse gespeichert in: {csv_filename}")
    
    return df, csv_filename

def main():
    """Hauptfunktion"""
    # Pfade definieren
    main_dir = os.getcwd()
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    # Prüfen ob Dateien existieren
    if not os.path.exists(jsp_data_path):
        print(f"Fehler: JSP-Datendatei nicht gefunden: {jsp_data_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Fehler: Modelldatei nicht gefunden: {model_path}")
        return
    
    # Analyse durchführen
    try:
        df, csv_path = analyze_ppo_priority_behavior(jsp_data_path, model_path)
        print(f"\n✓ Analyse erfolgreich abgeschlossen!")
        print(f"✓ CSV-Datei erstellt: {csv_path}")
        
    except Exception as e:
        print(f"Fehler bei der Analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()