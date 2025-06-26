import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch_ppo_agent import TorchPPOAgent
from gym_environment import JSPGymEnvironment
import warnings
warnings.filterwarnings('ignore')

class JSPHeuristicSolver:
    """
    Implementiert verschiedene Heuristiken für Job-Shop Scheduling
    """
    
    def __init__(self, jsp_data):
        self.jsp_data = jsp_data
        self.jobs = jsp_data["jobs"]
        self.machines = jsp_data["machines"]
        self.num_jobs = len(self.jobs)
        self.num_machines = len(self.machines)
        
        # Create mappings
        self.job_id_to_idx = {job["id"]: idx for idx, job in enumerate(self.jobs)}
        self.machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(self.machines)}
        
    def _get_valid_jobs(self, job_progress, current_time=None):
        """Gibt alle aktuell verfügbaren Jobs zurück"""
        valid_jobs = []
        for job_idx in range(self.num_jobs):
            if job_progress[job_idx] < len(self.jobs[job_idx]["operations"]):
                # Prüfe Vorgänger
                op_idx = job_progress[job_idx]
                op = self.jobs[job_idx]["operations"][op_idx]
                
                predecessors_completed = True
                for pred in op.get("predecessors", []):
                    pred_job_id, pred_op_id = pred.split(":")
                    pred_job_idx = self.job_id_to_idx[pred_job_id]
                    
                    # Finde Vorgänger-Operation Index
                    pred_op_idx = None
                    for i, pred_op in enumerate(self.jobs[pred_job_idx]["operations"]):
                        if pred_op["id"] == pred_op_id:
                            pred_op_idx = i
                            break
                    
                    if pred_op_idx is None or job_progress[pred_job_idx] <= pred_op_idx:
                        predecessors_completed = False
                        break
                
                if predecessors_completed:
                    valid_jobs.append(job_idx)
        
        return valid_jobs
    
    def fifo(self, env):
        """First In, First Out - Jobs in der Reihenfolge ihrer IDs"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Wähle den Job mit der niedrigsten ID (FIFO)
            action = min(valid_jobs)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        return actions, state
    
    def lifo(self, env):
        """Last In, First Out - Jobs in umgekehrter Reihenfolge ihrer IDs"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Wähle den Job mit der höchsten ID (LIFO)
            action = max(valid_jobs)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        return actions, state
    
    def spt(self, env):
        """Shortest Processing Time - Job mit kürzester Bearbeitungszeit"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Finde Job mit kürzester aktueller Operation
            min_time = float('inf')
            best_job = None
            
            for job_idx in valid_jobs:
                op_idx = state['job_progress'][job_idx]
                if op_idx < len(self.jobs[job_idx]["operations"]):
                    proc_time = self.jobs[job_idx]["operations"][op_idx]["processingTime"]
                    if proc_time < min_time:
                        min_time = proc_time
                        best_job = job_idx
            
            if best_job is not None:
                actions.append(best_job)
                state, _, done, _ = env.step(best_job)
                if done:
                    break
            else:
                break
        
        return actions, state
    
    def lpt(self, env):
        """Longest Processing Time - Job mit längster Bearbeitungszeit"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Finde Job mit längster aktueller Operation
            max_time = -1
            best_job = None
            
            for job_idx in valid_jobs:
                op_idx = state['job_progress'][job_idx]
                if op_idx < len(self.jobs[job_idx]["operations"]):
                    proc_time = self.jobs[job_idx]["operations"][op_idx]["processingTime"]
                    if proc_time > max_time:
                        max_time = proc_time
                        best_job = job_idx
            
            if best_job is not None:
                actions.append(best_job)
                state, _, done, _ = env.step(best_job)
                if done:
                    break
            else:
                break
        
        return actions, state
    
    def hpf(self, env):
        """Highest Priority First - Job mit höchster Priorität"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Finde Job mit höchster Priorität
            max_priority = -1
            best_job = None
            
            for job_idx in valid_jobs:
                priority = self.jobs[job_idx]["priority"]
                if priority > max_priority:
                    max_priority = priority
                    best_job = job_idx
            
            if best_job is not None:
                actions.append(best_job)
                state, _, done, _ = env.step(best_job)
                if done:
                    break
            else:
                break
        
        return actions, state
    
    def edf(self, env):
        """Earliest Deadline First - Job mit frühester Deadline"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Finde Job mit frühester Deadline
            earliest_deadline = float('inf')
            best_job = None
            
            for job_idx in valid_jobs:
                deadline = self.jobs[job_idx]["deadline"]
                if deadline < earliest_deadline:
                    earliest_deadline = deadline
                    best_job = job_idx
            
            if best_job is not None:
                actions.append(best_job)
                state, _, done, _ = env.step(best_job)
                if done:
                    break
            else:
                break
        
        return actions, state
    
    def cr(self, env):
        """Critical Ratio - Verhältnis von verbleibender Zeit zu verbleibender Arbeit"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            current_time = state['current_time'][0]
            min_ratio = float('inf')
            best_job = None
            
            for job_idx in valid_jobs:
                deadline = self.jobs[job_idx]["deadline"]
                remaining_time = max(0, deadline - current_time)
                
                # Berechne verbleibende Arbeit
                remaining_work = 0
                for op_idx in range(state['job_progress'][job_idx], len(self.jobs[job_idx]["operations"])):
                    remaining_work += self.jobs[job_idx]["operations"][op_idx]["processingTime"]
                
                if remaining_work > 0:
                    ratio = remaining_time / remaining_work
                else:
                    ratio = float('inf')
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    best_job = job_idx
            
            if best_job is not None:
                actions.append(best_job)
                state, _, done, _ = env.step(best_job)
                if done:
                    break
            else:
                break
        
        return actions, state
    
    def random_choice(self, env):
        """Zufällige Auswahl"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            valid_jobs = self._get_valid_jobs(state['job_progress'])
            if not valid_jobs:
                break
            
            # Zufällige Auswahl
            action = np.random.choice(valid_jobs)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        return actions, state


class JSPComparator:
    """
    Vergleicht verschiedene JSP-Lösungsansätze
    """
    
    def __init__(self, jsp_data_path, ppo_model_path):
        # Lade JSP-Daten
        with open(jsp_data_path, 'r') as f:
            self.jsp_data = json.load(f)
        
        # Analysiere und korrigiere unrealistische Deadlines
        self._analyze_and_fix_deadlines()
        
        # Initialisiere Umgebung und Agent
        self.env = JSPGymEnvironment(self.jsp_data, enable_logging=False)
        self.ppo_agent = TorchPPOAgent(len(self.jsp_data["jobs"]), self.jsp_data)
        
        # Lade PPO-Modell
        self.ppo_agent.load_model(ppo_model_path)
        
        # Initialisiere Heuristik-Solver
        self.heuristic_solver = JSPHeuristicSolver(self.jsp_data)
        
        # Definiere alle Heuristiken
        self.heuristics = {
            'PPO_Agent': self._run_ppo,
            'FIFO': self.heuristic_solver.fifo,
            'LIFO': self.heuristic_solver.lifo,
            'SPT': self.heuristic_solver.spt,
            'LPT': self.heuristic_solver.lpt,
            'HPF': self.heuristic_solver.hpf,
            'EDF': self.heuristic_solver.edf,
            'CR': self.heuristic_solver.cr,
            'Random': self.heuristic_solver.random_choice
        }
        
        self.results = {}
    
    def _analyze_and_fix_deadlines(self):
        """Analysiert und korrigiert unrealistische Deadlines"""
        # Berechne theoretische Mindestbearbeitungszeit pro Job
        total_processing_times = []
        deadlines = []
        
        for job in self.jsp_data["jobs"]:
            total_proc_time = sum(op["processingTime"] for op in job["operations"])
            total_processing_times.append(total_proc_time)
            deadlines.append(job["deadline"])
        
        total_work = sum(total_processing_times)
        avg_proc_time = np.mean(total_processing_times)
        max_deadline = max(deadlines)
        min_deadline = min(deadlines)
        
        print(f"Deadline-Analyse:")
        print(f"- Gesamtarbeit: {total_work} Minuten")
        print(f"- Durchschnittliche Job-Zeit: {avg_proc_time:.1f} Minuten")
        print(f"- Deadline-Bereich: {min_deadline} - {max_deadline}")
        print(f"- Anzahl Maschinen: {len(self.jsp_data['machines'])}")
        
        # Schätze realistische Makespan
        estimated_makespan = total_work / len(self.jsp_data["machines"]) * 1.5  # 50% Puffer für Setup/Idle
        
        print(f"- Geschätzte realistische Makespan: {estimated_makespan:.1f}")
        
        # Prüfe, ob Deadlines unrealistisch sind
        unrealistic_deadlines = [d for d in deadlines if d < estimated_makespan * 0.8]
        
        if len(unrealistic_deadlines) > len(deadlines) * 0.5:  # Mehr als 50% unrealistisch
            print(f"⚠️  {len(unrealistic_deadlines)} von {len(deadlines)} Deadlines scheinen unrealistisch kurz zu sein!")
            print("   Deadlines werden für bessere Vergleichbarkeit angepasst...")
            
            # Adjustiere Deadlines basierend auf Job-Komplexität
            for job in self.jsp_data["jobs"]:
                job_proc_time = sum(op["processingTime"] for op in job["operations"])
                job_complexity = job_proc_time / avg_proc_time  # Relative Komplexität
                priority_factor = (11 - job["priority"]) / 10.0  # Niedrigere Priorität = mehr Zeit
                
                # Neue Deadline basierend auf Komplexität und Priorität
                new_deadline = estimated_makespan * (0.6 + 0.4 * job_complexity) * priority_factor
                
                # Stelle sicher, dass hohe Priorität nicht zu unrealistischen Deadlines führt
                min_reasonable_deadline = job_proc_time * 2.0  # Mindestens doppelte Bearbeitungszeit
                new_deadline = max(new_deadline, min_reasonable_deadline)
                
                job["deadline"] = int(new_deadline)
            
            # Aktualisierte Statistiken
            new_deadlines = [job["deadline"] for job in self.jsp_data["jobs"]]
            print(f"   Neue Deadline-Bereich: {min(new_deadlines)} - {max(new_deadlines)}")
        else:
            print("✅ Deadlines erscheinen realistisch.")

    
    def _run_ppo(self, env):
        """Führt PPO-Agent aus"""
        state = env.reset()
        actions = []
        
        while not env.completed_jobs >= env.num_jobs:
            action, _ = self.ppo_agent.select_action(state)
            actions.append(action)
            
            state, _, done, _ = env.step(action)
            if done:
                break
        
        return actions, state
    
    def _calculate_metrics(self, final_state, actions, algorithm_name):
        """Berechnet umfassende Metriken für eine Lösung"""
        metrics = {}
        
        # Grundlegende Metriken
        metrics['algorithm'] = algorithm_name
        metrics['makespan'] = float(max(final_state['machine_times']))
        metrics['total_actions'] = len(actions)
        metrics['completed_jobs'] = sum([1 for i, progress in enumerate(final_state['job_progress']) 
                                       if progress >= len(self.jsp_data["jobs"][i]["operations"])])
        
        current_time = float(final_state['current_time'][0])
        
        # Deadline-Performance mit Debug-Info
        met_deadlines = 0
        total_tardiness = 0.0
        total_earliness = 0.0
        deadline_violations = 0
        
        # Debug: Zeige Deadline-Verteilung
        deadlines = [job["deadline"] for job in self.jsp_data["jobs"]]
        if algorithm_name == 'PPO_Agent':  # Nur einmal ausgeben
            print(f"Debug - Current time: {current_time:.1f}")
            print(f"Debug - Deadlines: min={min(deadlines)}, max={max(deadlines)}, avg={np.mean(deadlines):.1f}")
        
        for i, progress in enumerate(final_state['job_progress']):
            if progress >= len(self.jsp_data["jobs"][i]["operations"]):
                deadline = float(self.jsp_data["jobs"][i]["deadline"])
                completion_time = current_time
                
                if completion_time <= deadline:
                    met_deadlines += 1
                    total_earliness += (deadline - completion_time)
                else:
                    deadline_violations += 1
                    total_tardiness += (completion_time - deadline)
        
        metrics['met_deadlines'] = met_deadlines
        metrics['deadline_ratio'] = float(met_deadlines / len(self.jsp_data["jobs"]) if len(self.jsp_data["jobs"]) > 0 else 0)
        metrics['total_tardiness'] = float(total_tardiness)
        metrics['avg_tardiness'] = float(total_tardiness / len(self.jsp_data["jobs"]) if len(self.jsp_data["jobs"]) > 0 else 0)
        metrics['total_earliness'] = float(total_earliness)
        metrics['avg_earliness'] = float(total_earliness / len(self.jsp_data["jobs"]) if len(self.jsp_data["jobs"]) > 0 else 0)
        metrics['deadline_violations'] = deadline_violations
        
        # Alternative Timeliness-Berechnung wenn alle Deadlines verfehlt werden
        if met_deadlines == 0 and len(self.jsp_data["jobs"]) > 0:
            # Verwende relative Tardiness als inverse Timeliness
            relative_tardiness_scores = []
            for i, job in enumerate(self.jsp_data["jobs"]):
                progress = final_state['job_progress'][i]
                if progress >= len(job["operations"]):
                    deadline = float(job["deadline"])
                    priority = float(job["priority"])
                    
                    if current_time > deadline:
                        # Je geringer die relative Verspätung, desto besser
                        relative_tardiness = (current_time - deadline) / deadline if deadline > 0 else 1.0
                        # Score von 0 (sehr schlecht) bis 1 (wenig verspätet)
                        job_score = max(0.0, 1.0 - relative_tardiness) * (priority / 10.0)
                    else:
                        # Sollte nicht auftreten, da met_deadlines == 0
                        job_score = 1.0 * (priority / 10.0)
                    
                    relative_tardiness_scores.append(job_score)
            
            if relative_tardiness_scores:
                metrics['timeliness_score'] = float(np.mean(relative_tardiness_scores))
            else:
                metrics['timeliness_score'] = 0.0
        else:
            # Prioritäts-Performance
            priority_weighted_completion = 0.0
            high_priority_met = 0
            high_priority_total = 0
            
            for i, progress in enumerate(final_state['job_progress']):
                job = self.jsp_data["jobs"][i]
                priority = float(job["priority"])
                
                if priority >= 7:
                    high_priority_total += 1
                    if (progress >= len(job["operations"]) and 
                        current_time <= float(job["deadline"])):
                        high_priority_met += 1
                
                if progress >= len(job["operations"]):
                    priority_weighted_completion += priority * current_time
            
            total_priority = sum(float(job["priority"]) for job in self.jsp_data["jobs"])
            metrics['priority_weighted_avg_completion'] = float(priority_weighted_completion / total_priority 
                                                             if total_priority > 0 else 0)
            metrics['high_priority_ratio'] = float(high_priority_met / high_priority_total 
                                                  if high_priority_total > 0 else 0)
            
            # Timeliness Score (wie vorher)
            timeliness_score = 0.0
            total_jobs = len(self.jsp_data["jobs"])
            
            for i, job in enumerate(self.jsp_data["jobs"]):
                progress = final_state['job_progress'][i]
                if progress >= len(job["operations"]):
                    deadline = float(job["deadline"])
                    completion_time = current_time
                    priority = float(job["priority"])
                    
                    if completion_time <= deadline:
                        slack = deadline - completion_time
                        normalized_slack = slack / deadline if deadline > 0 else 0
                        job_timeliness = (1.0 + normalized_slack) * (priority / 10.0)
                    else:
                        tardiness = completion_time - deadline
                        normalized_tardiness = tardiness / deadline if deadline > 0 else 1
                        job_timeliness = max(0, 1.0 - normalized_tardiness) * (priority / 10.0)
                    
                    timeliness_score += job_timeliness
            
            metrics['timeliness_score'] = float(timeliness_score / total_jobs if total_jobs > 0 else 0)
        
        # Füge Fallback-Werte hinzu, falls sie nicht gesetzt wurden
        if 'high_priority_ratio' not in metrics:
            metrics['high_priority_ratio'] = 0.0
        if 'priority_weighted_avg_completion' not in metrics:
            metrics['priority_weighted_avg_completion'] = float(current_time)
        
        # Maschinen-Utilization
        if current_time > 0:
            total_machine_time = sum(final_state['machine_times'])
            max_possible_time = current_time * len(final_state['machine_times'])
            metrics['machine_utilization'] = float(total_machine_time / max_possible_time)
            
            # Maschinen-Balance (Standardabweichung der Maschinenzeiten)
            machine_times = [float(t) for t in final_state['machine_times']]
            mean_time = sum(machine_times) / len(machine_times)
            variance = sum((t - mean_time) ** 2 for t in machine_times) / len(machine_times)
            metrics['machine_balance'] = float(1.0 / (1.0 + np.sqrt(variance) / mean_time) if mean_time > 0 else 0)
        else:
            metrics['machine_utilization'] = 0.0
            metrics['machine_balance'] = 0.0
        
        # Effizienz-Metriken
        total_processing_time = sum(float(op["processingTime"]) 
                                  for job in self.jsp_data["jobs"] 
                                  for op in job["operations"])
        metrics['efficiency_ratio'] = float(total_processing_time / metrics['makespan'] if metrics['makespan'] > 0 else 0)
        
        # Flowtime (Durchlaufzeit)
        total_flowtime = 0.0
        for i, job in enumerate(self.jsp_data["jobs"]):
            if final_state['job_progress'][i] >= len(job["operations"]):
                total_flowtime += current_time
        
        metrics['avg_flowtime'] = float(total_flowtime / metrics['completed_jobs'] if metrics['completed_jobs'] > 0 else 0)
        
        return metrics
    
    def run_comparison(self, num_runs=5):
        """Führt Vergleich aller Heuristiken durch"""
        print("Starte umfassenden Vergleich der JSP-Heuristiken...")
        print(f"Anzahl Jobs: {len(self.jsp_data['jobs'])}")
        print(f"Anzahl Maschinen: {len(self.jsp_data['machines'])}")
        print(f"Anzahl Durchläufe pro Heuristik: {num_runs}")
        print("-" * 60)
        
        all_results = []
        
        for alg_name, alg_func in self.heuristics.items():
            print(f"Teste {alg_name}...")
            alg_results = []
            
            for run in range(num_runs):
                try:
                    # Setze Seed für reproduzierbare Ergebnisse bei Random
                    if alg_name == 'Random':
                        np.random.seed(42 + run)
                    
                    # Führe Algorithmus aus
                    actions, final_state = alg_func(self.env)
                    
                    # Berechne Metriken
                    metrics = self._calculate_metrics(final_state, actions, alg_name)
                    metrics['run'] = run + 1
                    
                    alg_results.append(metrics)
                    all_results.append(metrics)
                    
                except Exception as e:
                    print(f"   Fehler in Run {run + 1}: {str(e)}")
                    continue
            
            if alg_results:
                avg_makespan = np.mean([r['makespan'] for r in alg_results])
                avg_deadline_ratio = np.mean([r['deadline_ratio'] for r in alg_results])
                avg_timeliness = np.mean([r['timeliness_score'] for r in alg_results])
                print(f"   Durchschnitt - Makespan: {avg_makespan:.1f}, "
                      f"Deadline-Ratio: {avg_deadline_ratio:.3f}, "
                      f"Timeliness: {avg_timeliness:.3f}")
            
            self.results[alg_name] = alg_results
        
        print("\nVergleich abgeschlossen!")
        return all_results
    
    def generate_comprehensive_report(self, results):
        """Generiert umfassenden Bericht mit allen Metriken"""
        # Erstelle DataFrame
        df = pd.DataFrame(results)
        
        # Berechne Aggregationen pro Algorithmus
        summary_stats = df.groupby('algorithm').agg({
            'makespan': ['mean', 'std', 'min', 'max'],
            'deadline_ratio': ['mean', 'std', 'min', 'max'],
            'timeliness_score': ['mean', 'std', 'min', 'max'],
            'machine_utilization': ['mean', 'std'],
            'machine_balance': ['mean', 'std'],
            'avg_tardiness': ['mean', 'std'],
            'avg_earliness': ['mean', 'std'],
            'high_priority_ratio': ['mean', 'std'],
            'efficiency_ratio': ['mean', 'std'],
            'avg_flowtime': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        
        # Erstelle Ranking basierend auf verschiedenen Kriterien
        rankings = pd.DataFrame(index=summary_stats.index)
        
        # Ranking nach Makespan (niedriger ist besser)
        rankings['makespan_rank'] = summary_stats['makespan_mean'].rank(method='min')
        
        # Ranking nach Deadline-Ratio (höher ist besser)
        rankings['deadline_ratio_rank'] = summary_stats['deadline_ratio_mean'].rank(method='min', ascending=False)
        
        # Ranking nach Timeliness Score (höher ist besser)
        rankings['timeliness_rank'] = summary_stats['timeliness_score_mean'].rank(method='min', ascending=False)
        
        # Ranking nach Machine Utilization (höher ist besser)
        rankings['utilization_rank'] = summary_stats['machine_utilization_mean'].rank(method='min', ascending=False)
        
        # Kombiniertes Ranking (gewichteter Durchschnitt)
        rankings['combined_rank'] = (
            rankings['makespan_rank'] * 0.3 +
            rankings['deadline_ratio_rank'] * 0.25 +
            rankings['timeliness_rank'] * 0.25 +
            rankings['utilization_rank'] * 0.2
        )
        
        rankings['overall_rank'] = rankings['combined_rank'].rank(method='min')
        
        return df, summary_stats, rankings
    
    def create_visualizations(self, df, summary_stats, output_dir='results/comparison'):
        """Erstellt klare, aussagekräftige Visualisierungen"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtere nur Metriken mit aussagekräftigen Daten (nicht nur Nullen)
        meaningful_metrics = []
        for metric in ['makespan', 'deadline_ratio', 'timeliness_score', 'machine_utilization', 'avg_tardiness']:
            metric_values = df[metric].values
            if not np.all(metric_values == 0) and not np.all(np.isnan(metric_values)):
                metric_range = np.max(metric_values) - np.min(metric_values)
                if metric_range > 1e-10:  # Nur wenn es Variation gibt
                    meaningful_metrics.append(metric)
        
        print(f"Aussagekräftige Metriken für Visualisierung: {meaningful_metrics}")
        
        # Aggregiere Daten pro Algorithmus
        avg_data = df.groupby('algorithm')[meaningful_metrics].mean()
        std_data = df.groupby('algorithm')[meaningful_metrics].std().fillna(0)
        
        # Erstelle eine große, übersichtliche Grafik
        num_metrics = len(meaningful_metrics)
        if num_metrics == 0:
            print("⚠️  Keine aussagekräftigen Metriken gefunden für Visualisierung")
            return
        
        # 1. Hauptvergleichsgrafik
        fig, axes = plt.subplots(1, min(num_metrics, 4), figsize=(16, 6))
        if num_metrics == 1:
            axes = [axes]
        
        # Farben definieren
        colors = []
        for alg in avg_data.index:
            if alg == 'PPO_Agent':
                colors.append('#E74C3C')  # Rot für PPO
            else:
                colors.append('#3498DB')  # Blau für andere
        
        for i, metric in enumerate(meaningful_metrics[:4]):  # Maximal 4 Metriken
            ax = axes[i] if i < len(axes) else axes[-1]
            
            # Sortiere nach Performance (je nach Metrik aufsteigend oder absteigend)
            if metric in ['makespan', 'avg_tardiness']:
                # Niedriger ist besser
                sorted_data = avg_data[metric].sort_values()
            else:
                # Höher ist besser
                sorted_data = avg_data[metric].sort_values(ascending=False)
            
            # Erstelle Balkendiagramm mit Fehlerbalken
            y_pos = range(len(sorted_data))
            bars = ax.barh(y_pos, sorted_data.values, 
                          color=[colors[list(avg_data.index).index(alg)] for alg in sorted_data.index],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Füge Fehlerbalken hinzu wenn vorhanden
            if metric in std_data.columns:
                error_values = [std_data.loc[alg, metric] for alg in sorted_data.index]
                ax.errorbar(sorted_data.values, y_pos, xerr=error_values, 
                           fmt='none', ecolor='black', capsize=3, alpha=0.7)
            
            # Beschriftung
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_data.index, fontsize=10)
            ax.set_xlabel(self._get_metric_label(metric), fontsize=11, fontweight='bold')
            ax.set_title(self._get_metric_title(metric), fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Werte auf Balken anzeigen
            for j, (bar, value) in enumerate(zip(bars, sorted_data.values)):
                ax.text(value + max(sorted_data.values) * 0.01, j, f'{value:.0f}' if metric == 'makespan' else f'{value:.3f}',
                       va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/jsp_comparison_main.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Direkter PPO-Vergleich (nur wenn PPO vorhanden ist)
        if 'PPO_Agent' in avg_data.index and len(meaningful_metrics) > 1:
            self._create_ppo_comparison_chart(avg_data, meaningful_metrics, output_dir)
        
        # 3. Ranking-Übersicht
        if len(meaningful_metrics) >= 2:
            self._create_ranking_overview(avg_data, meaningful_metrics, output_dir)
    
    def _get_metric_label(self, metric):
        """Gibt benutzerfreundliche Labels für Metriken zurück"""
        labels = {
            'makespan': 'Zeit (Minuten)',
            'deadline_ratio': 'Anteil (0-1)',
            'timeliness_score': 'Score (0-1)',
            'machine_utilization': 'Auslastung (0-1)',
            'avg_tardiness': 'Verspätung (Minuten)',
            'efficiency_ratio': 'Effizienz (0-1)'
        }
        return labels.get(metric, metric.replace('_', ' ').title())
    
    def _get_metric_title(self, metric):
        """Gibt benutzerfreundliche Titel für Metriken zurück"""
        titles = {
            'makespan': 'Gesamtdauer (Makespan)',
            'deadline_ratio': 'Deadline-Einhaltung',
            'timeliness_score': 'Pünktlichkeits-Score',
            'machine_utilization': 'Maschinenauslastung',
            'avg_tardiness': 'Durchschn. Verspätung',
            'efficiency_ratio': 'Effizienz-Ratio'
        }
        return titles.get(metric, metric.replace('_', ' ').title())
    
    def _create_ppo_comparison_chart(self, avg_data, meaningful_metrics, output_dir):
        """Erstellt eine spezielle PPO-Vergleichsgrafik"""
        plt.figure(figsize=(12, 8))
        
        ppo_values = avg_data.loc['PPO_Agent', meaningful_metrics].values
        
        # Finde für jede Metrik den besten Nicht-PPO-Wert
        best_others = []
        best_other_names = []
        
        for metric in meaningful_metrics:
            other_algos = avg_data.drop('PPO_Agent', errors='ignore')
            if len(other_algos) > 0:
                if metric in ['makespan', 'avg_tardiness']:
                    best_value = other_algos[metric].min()
                    best_algo = other_algos[metric].idxmin()
                else:
                    best_value = other_algos[metric].max()
                    best_algo = other_algos[metric].idxmax()
                
                best_others.append(best_value)
                best_other_names.append(best_algo)
            else:
                best_others.append(0)
                best_other_names.append('N/A')
        
        # Erstelle Vergleichsbalken
        x_pos = range(len(meaningful_metrics))
        width = 0.35
        
        bars1 = plt.bar([x - width/2 for x in x_pos], ppo_values, width, 
                       label='PPO Agent', color='#E74C3C', alpha=0.8)
        bars2 = plt.bar([x + width/2 for x in x_pos], best_others, width,
                       label='Beste Heuristik', color='#3498DB', alpha=0.8)
        
        plt.xlabel('Metriken', fontweight='bold')
        plt.ylabel('Werte', fontweight='bold')
        plt.title('PPO Agent vs. Beste klassische Heuristik', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, [self._get_metric_title(m) for m in meaningful_metrics], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Werte auf Balken
        for bar, value in zip(bars1, ppo_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                    f'{value:.3f}' if abs(value) < 100 else f'{value:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, value in zip(bars2, best_others):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                    f'{value:.3f}' if abs(value) < 100 else f'{value:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ppo_vs_best_heuristics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_ranking_overview(self, avg_data, meaningful_metrics, output_dir):
        """Erstellt eine Ranking-Übersicht"""
        plt.figure(figsize=(10, 6))
        
        # Berechne Rankings
        rankings = pd.DataFrame(index=avg_data.index)
        
        for metric in meaningful_metrics:
            if metric in ['makespan', 'avg_tardiness']:
                # Niedriger ist besser
                rankings[metric] = avg_data[metric].rank(method='min')
            else:
                # Höher ist besser
                rankings[metric] = avg_data[metric].rank(method='min', ascending=False)
        
        # Durchschnittsrang
        rankings['avg_rank'] = rankings[meaningful_metrics].mean(axis=1)
        rankings_sorted = rankings.sort_values('avg_rank')
        
        # Erstelle Heatmap
        sns.heatmap(rankings_sorted[meaningful_metrics], 
                   annot=True, cmap='RdYlBu_r', fmt='.1f',
                   xticklabels=[self._get_metric_title(m) for m in meaningful_metrics],
                   yticklabels=rankings_sorted.index,
                   cbar_kws={'label': 'Rang (1 = Beste)'})
        
        plt.title('Ranking-Übersicht aller Algorithmen\n(1 = Beste Performance)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Algorithmus', fontweight='bold')
        
        # Markiere PPO Agent
        if 'PPO_Agent' in rankings_sorted.index:
            ppo_pos = list(rankings_sorted.index).index('PPO_Agent')
            plt.axhline(y=ppo_pos+0.5, color='red', linewidth=3, alpha=0.7)
            plt.axhline(y=ppo_pos-0.5, color='red', linewidth=3, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/algorithm_rankings_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_comprehensive_results(self, df, summary_stats, rankings, output_dir='results/comparison'):
        """Speichert alle Ergebnisse in verschiedenen Formaten"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Detaillierte Ergebnisse als CSV
        df.to_csv(f'{output_dir}/detailed_results_{timestamp}.csv', index=False)
        
        # 2. Zusammenfassung als CSV
        summary_stats.to_csv(f'{output_dir}/summary_statistics_{timestamp}.csv')
        
        # 3. Rankings als CSV
        rankings.to_csv(f'{output_dir}/algorithm_rankings_{timestamp}.csv')
        
        # 4. Umfassender JSON-Report für weitere Verarbeitung
        comprehensive_report = {
            'metadata': {
                'timestamp': timestamp,
                'num_jobs': len(self.jsp_data['jobs']),
                'num_machines': len(self.jsp_data['machines']),
                'jsp_data_summary': {
                    'job_priorities': [job['priority'] for job in self.jsp_data['jobs']],
                    'job_deadlines': [job['deadline'] for job in self.jsp_data['jobs']],
                    'total_operations': sum(len(job['operations']) for job in self.jsp_data['jobs']),
                    'machines': [machine['id'] for machine in self.jsp_data['machines']]
                }
            },
            'algorithm_performance': {},
            'rankings': rankings.to_dict(),
            'statistical_tests': {}
        }
        
        # Füge detaillierte Performance-Daten hinzu
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            comprehensive_report['algorithm_performance'][alg] = {
                'runs': len(alg_data),
                'metrics': {
                    'makespan': {
                        'mean': float(alg_data['makespan'].mean()),
                        'std': float(alg_data['makespan'].std()),
                        'min': float(alg_data['makespan'].min()),
                        'max': float(alg_data['makespan'].max()),
                        'values': alg_data['makespan'].tolist()
                    },
                    'deadline_ratio': {
                        'mean': float(alg_data['deadline_ratio'].mean()),
                        'std': float(alg_data['deadline_ratio'].std()),
                        'values': alg_data['deadline_ratio'].tolist()
                    },
                    'timeliness_score': {
                        'mean': float(alg_data['timeliness_score'].mean()),
                        'std': float(alg_data['timeliness_score'].std()),
                        'values': alg_data['timeliness_score'].tolist()
                    },
                    'machine_utilization': {
                        'mean': float(alg_data['machine_utilization'].mean()),
                        'std': float(alg_data['machine_utilization'].std()),
                        'values': alg_data['machine_utilization'].tolist()
                    },
                    'high_priority_ratio': {
                        'mean': float(alg_data['high_priority_ratio'].mean()),
                        'std': float(alg_data['high_priority_ratio'].std()),
                        'values': alg_data['high_priority_ratio'].tolist()
                    },
                    'avg_tardiness': {
                        'mean': float(alg_data['avg_tardiness'].mean()),
                        'std': float(alg_data['avg_tardiness'].std()),
                        'values': alg_data['avg_tardiness'].tolist()
                    }
                }
            }
        
        # Statistische Tests (falls scipy verfügbar)
        try:
            from scipy import stats
            
            ppo_makespan = df[df['algorithm'] == 'PPO_Agent']['makespan'].values
            
            for alg in df['algorithm'].unique():
                if alg != 'PPO_Agent':
                    alg_makespan = df[df['algorithm'] == alg]['makespan'].values
                    
                    # Wilcoxon-Test für Makespan-Vergleich
                    if len(ppo_makespan) > 0 and len(alg_makespan) > 0:
                        try:
                            stat, p_value = stats.mannwhitneyu(ppo_makespan, alg_makespan, alternative='two-sided')
                            comprehensive_report['statistical_tests'][f'PPO_vs_{alg}_makespan'] = {
                                'test': 'Mann-Whitney U',
                                'statistic': float(stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except:
                            pass
        
        except ImportError:
            comprehensive_report['statistical_tests']['note'] = 'scipy not available for statistical tests'
        
        # Speichere JSON-Report mit Konvertierung für numpy-Typen
        def convert_numpy_types(obj):
            """Konvertiert numpy-Typen zu JSON-serialisierbaren Python-Typen"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Konvertiere den gesamten Report
        serializable_report = convert_numpy_types(comprehensive_report)
        
        with open(f'{output_dir}/comprehensive_report_{timestamp}.json', 'w') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        # 5. Erstelle Markdown-Report für Dokumentation
        markdown_content = self._generate_markdown_report(df, summary_stats, rankings, comprehensive_report)
        with open(f'{output_dir}/analysis_report_{timestamp}.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Alle Ergebnisse gespeichert in: {output_dir}")
        print(f"Timestamp: {timestamp}")
        
        return comprehensive_report
    
    def _generate_markdown_report(self, df, summary_stats, rankings, comprehensive_report):
        """Generiert einen detaillierten Markdown-Report"""
        timestamp = comprehensive_report['metadata']['timestamp']
        
        md_content = f"""# Job-Shop Scheduling: Vergleich von Heuristiken vs. PPO-Agent

**Analysezeitpunkt:** {timestamp}  
**Anzahl Jobs:** {comprehensive_report['metadata']['num_jobs']}  
**Anzahl Maschinen:** {comprehensive_report['metadata']['num_machines']}  
**Gesamtoperationen:** {comprehensive_report['metadata']['jsp_data_summary']['total_operations']}

## Executive Summary

Dieser Bericht vergleicht die Leistung eines trainierten PPO-Reinforcement Learning Agenten mit klassischen Job-Shop Scheduling Heuristiken.

### Top-3 Algorithmen nach Gesamtperformance:
"""
        
        # Add top 3 rankings
        top_3 = rankings.sort_values('overall_rank').head(3)
        for i, (alg, row) in enumerate(top_3.iterrows()):
            md_content += f"{i+1}. **{alg}** (Gesamtrang: {row['overall_rank']:.1f})\n"
        
        md_content += f"""

## Detaillierte Metriken

### 1. Makespan (Gesamtdauer)
"""
        
        makespan_ranking = summary_stats['makespan_mean'].sort_values()
        for i, (alg, value) in enumerate(makespan_ranking.items()):
            ppo_marker = " 🎯" if alg == "PPO_Agent" else ""
            md_content += f"{i+1}. {alg}: {value:.2f} ± {summary_stats.loc[alg, 'makespan_std']:.2f}{ppo_marker}\n"
        
        md_content += f"""
**PPO-Performance:** {"✅ Top-3" if "PPO_Agent" in makespan_ranking.head(3).index else "❌ Nicht Top-3"}

### 2. Deadline-Einhaltung
"""
        
        deadline_ranking = summary_stats['deadline_ratio_mean'].sort_values(ascending=False)
        for i, (alg, value) in enumerate(deadline_ranking.items()):
            ppo_marker = " 🎯" if alg == "PPO_Agent" else ""
            md_content += f"{i+1}. {alg}: {value:.3f} ({value*100:.1f}%) ± {summary_stats.loc[alg, 'deadline_ratio_std']:.3f}{ppo_marker}\n"
        
        md_content += f"""
**PPO-Performance:** {"✅ Top-3" if "PPO_Agent" in deadline_ranking.head(3).index else "❌ Nicht Top-3"}

### 3. Timeliness Score
"""
        
        timeliness_ranking = summary_stats['timeliness_score_mean'].sort_values(ascending=False)
        for i, (alg, value) in enumerate(timeliness_ranking.items()):
            ppo_marker = " 🎯" if alg == "PPO_Agent" else ""
            md_content += f"{i+1}. {alg}: {value:.3f} ± {summary_stats.loc[alg, 'timeliness_score_std']:.3f}{ppo_marker}\n"
        
        md_content += f"""
**PPO-Performance:** {"✅ Top-3" if "PPO_Agent" in timeliness_ranking.head(3).index else "❌ Nicht Top-3"}

### 4. Maschinenauslastung
"""
        
        util_ranking = summary_stats['machine_utilization_mean'].sort_values(ascending=False)
        for i, (alg, value) in enumerate(util_ranking.items()):
            ppo_marker = " 🎯" if alg == "PPO_Agent" else ""
            md_content += f"{i+1}. {alg}: {value:.3f} ({value*100:.1f}%) ± {summary_stats.loc[alg, 'machine_utilization_std']:.3f}{ppo_marker}\n"
        
        # PPO-spezifische Analyse
        if 'PPO_Agent' in comprehensive_report['algorithm_performance']:
            ppo_data = comprehensive_report['algorithm_performance']['PPO_Agent']
            
            md_content += f"""

## PPO-Agent Detailanalyse

### Leistungsübersicht:
- **Durchschnittliche Makespan:** {ppo_data['metrics']['makespan']['mean']:.2f} (Std: {ppo_data['metrics']['makespan']['std']:.2f})
- **Beste Makespan:** {ppo_data['metrics']['makespan']['min']:.2f}
- **Schlechteste Makespan:** {ppo_data['metrics']['makespan']['max']:.2f}
- **Deadline-Einhaltung:** {ppo_data['metrics']['deadline_ratio']['mean']*100:.1f}%
- **Timeliness Score:** {ppo_data['metrics']['timeliness_score']['mean']:.3f}
- **Maschinenauslastung:** {ppo_data['metrics']['machine_utilization']['mean']*100:.1f}%

### Stärken des PPO-Agenten:
"""
            
            # Analyze strengths
            ppo_ranks = rankings.loc['PPO_Agent'] if 'PPO_Agent' in rankings.index else None
            if ppo_ranks is not None:
                if ppo_ranks['makespan_rank'] <= 3:
                    md_content += "- ✅ Exzellente Makespan-Performance (Top-3)\n"
                if ppo_ranks['deadline_ratio_rank'] <= 3:
                    md_content += "- ✅ Sehr gute Deadline-Einhaltung (Top-3)\n"
                if ppo_ranks['timeliness_rank'] <= 3:
                    md_content += "- ✅ Hoher Timeliness Score (Top-3)\n"
                if ppo_ranks['utilization_rank'] <= 3:
                    md_content += "- ✅ Effiziente Maschinennutzung (Top-3)\n"
        
        md_content += """

## Vergleich mit klassischen Heuristiken

### Beobachtungen:
1. **FIFO/LIFO:** Einfache Implementierung, aber oft suboptimale Ergebnisse
2. **SPT (Shortest Processing Time):** Gut für Durchsatz, problematisch für lange Jobs
3. **EDF (Earliest Deadline First):** Speziell für Deadline-Einhaltung optimiert
4. **HPF (Highest Priority First):** Fokus auf wichtige Jobs
5. **CR (Critical Ratio):** Balanciert Zeit und verbleibende Arbeit

### Empfehlungen:
"""
        
        # Generate recommendations based on rankings
        if 'PPO_Agent' in rankings.index:
            overall_rank = rankings.loc['PPO_Agent', 'overall_rank']
            if overall_rank <= 2:
                md_content += "- 🌟 **PPO-Agent wird empfohlen** - Beste Gesamtperformance\n"
            elif overall_rank <= 4:
                md_content += "- ⭐ **PPO-Agent ist wettbewerbsfähig** - Gute Gesamtperformance\n"
            else:
                md_content += "- ⚠️ **PPO-Agent benötigt Verbesserung** - Klassische Heuristiken überlegen\n"
        
        md_content += """
- Für produktive Umgebungen: Kombination aus PPO und fallback auf bewährte Heuristiken
- Weitere Optimierung der PPO-Reward-Funktion für spezifische Anwendungsfälle
- Berücksichtigung von Setup-Zeiten und Materialwechseln in der Bewertung

## Dateien und weitere Analyse

Alle detaillierten Daten sind verfügbar in:
- `detailed_results_*.csv` - Rohdaten aller Testläufe
- `summary_statistics_*.csv` - Aggregierte Statistiken
- `algorithm_rankings_*.csv` - Ranking-Übersicht
- `comprehensive_report_*.json` - Vollständiger maschinenlesbarer Report

Für tiefergehende statistische Analysen können die bereitgestellten Daten in R, Python oder anderen Analysewerkzeugen weiterverwendet werden.
"""
        
        return md_content


def main():
    """Hauptfunktion für den Vergleich"""
    # Konfiguration
    jsp_data_path = "data.json"
    ppo_model_path = "results/models/gym_ppo_model_20250625_100324.pt"
    
    # Überprüfe, ob Dateien existieren
    if not os.path.exists(jsp_data_path):
        print(f"Fehler: {jsp_data_path} nicht gefunden!")
        print("Führen Sie zuerst data_generator.py aus, um Testdaten zu generieren.")
        return
    
    if not os.path.exists(ppo_model_path):
        print(f"Fehler: PPO-Modell {ppo_model_path} nicht gefunden!")
        print("Trainieren Sie zuerst ein PPO-Modell mit train_gym_ppo.py")
        return
    
    # Erstelle Comparator
    print("Initialisiere JSP-Vergleichsanalyse...")
    comparator = JSPComparator(jsp_data_path, ppo_model_path)
    
    # Führe Vergleich durch
    print("Starte Vergleichstests...")
    results = comparator.run_comparison(num_runs=10)  # 10 Runs für statistische Signifikanz
    
    # Generiere Berichte
    print("Generiere Analyseberichte...")
    df, summary_stats, rankings = comparator.generate_comprehensive_report(results)
    
    # Erstelle Visualisierungen
    print("Erstelle Visualisierungen...")
    comparator.create_visualizations(df, summary_stats)
    
    # Speichere alle Ergebnisse
    print("Speichere Ergebnisse...")
    comprehensive_report = comparator.save_comprehensive_results(df, summary_stats, rankings)
    
    # Ausgabe der wichtigsten Ergebnisse
    print("\n" + "="*60)
    print("ZUSAMMENFASSUNG DER ERGEBNISSE")
    print("="*60)
    
    print(f"\nGesamtranking (kombinierte Bewertung):")
    for i, (alg, rank) in enumerate(rankings.sort_values('overall_rank').head(5)['overall_rank'].items()):
        marker = " 🏆" if i == 0 else " 🥈" if i == 1 else " 🥉" if i == 2 else ""
        print(f"{i+1}. {alg}: {rank:.2f}{marker}")
    
    if 'PPO_Agent' in summary_stats.index:
        print(f"\nPPO-Agent Performance:")
        ppo_stats = summary_stats.loc['PPO_Agent']
        print(f"- Makespan: {ppo_stats['makespan_mean']:.2f} ± {ppo_stats['makespan_std']:.2f}")
        print(f"- Deadline-Ratio: {ppo_stats['deadline_ratio_mean']:.3f} ({ppo_stats['deadline_ratio_mean']*100:.1f}%)")
        print(f"- Timeliness Score: {ppo_stats['timeliness_score_mean']:.3f}")
        print(f"- Machine Utilization: {ppo_stats['machine_utilization_mean']:.3f} ({ppo_stats['machine_utilization_mean']*100:.1f}%)")
    
    print(f"\nAlle Ergebnisse wurden gespeichert in: results/comparison/")
    print("Verwenden Sie die generierten CSV- und JSON-Dateien für weitere Analysen.")


if __name__ == "__main__":
    main()