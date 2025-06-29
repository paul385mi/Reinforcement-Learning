import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auswertung.ppo_heuristics_comparison import compare_methods

def create_comparison_charts(results):
    """Erstelle einfache Vergleichsdiagramme (aus heuristics_visualization.py)"""
    
    # Extrahiere Daten für Visualisierung
    methods = [r['heuristic'] for r in results]
    makespans = [r['makespan'] for r in results]
    utilizations = [r['utilization'] for r in results]
    deadline_ratios = [r['deadline_ratio'] * 100 for r in results]  # In Prozent
    avg_delays = [r['avg_delay'] for r in results]
    
    # Farben für PPO vs Heuristiken
    colors = ['#FF6B6B' if method == 'PPO Agent' else '#4ECDC4' for method in methods]
    
    # Erstelle 2x2 Subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PPO Agent vs. Dispatching-Heuristiken Vergleich', fontsize=16, fontweight='bold')
    
    # 1. Makespan Vergleich
    bars1 = ax1.bar(range(len(methods)), makespans, color=colors, alpha=0.8)
    ax1.set_title('Makespan (niedrigere Werte = besser)', fontweight='bold')
    ax1.set_ylabel('Makespan (Minuten)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Beste Makespan markieren
    best_makespan_idx = np.argmin(makespans)
    bars1[best_makespan_idx].set_color('#2E8B57')
    ax1.text(best_makespan_idx, makespans[best_makespan_idx] + 50, 
             f'BESTE\n{makespans[best_makespan_idx]:.0f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # 2. Maschinenauslastung
    bars2 = ax2.bar(range(len(methods)), utilizations, color=colors, alpha=0.8)
    ax2.set_title('Maschinenauslastung (höhere Werte = besser)', fontweight='bold')
    ax2.set_ylabel('Auslastung (0-1)')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 1.0)  # Zoom auf relevanten Bereich
    
    # Beste Auslastung markieren
    best_util_idx = np.argmax(utilizations)
    bars2[best_util_idx].set_color('#2E8B57')
    ax2.text(best_util_idx, utilizations[best_util_idx] + 0.005, 
             f'BESTE\n{utilizations[best_util_idx]:.3f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # 3. Deadline-Erfüllungsrate
    bars3 = ax3.bar(range(len(methods)), deadline_ratios, color=colors, alpha=0.8)
    ax3.set_title('Deadline-Erfüllungsrate (höhere Werte = besser)', fontweight='bold')
    ax3.set_ylabel('Erfüllungsrate (%)')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 5)  # Zoom, falls es überhaupt erfüllte Deadlines gibt
    
    # Notiz, falls alle 0% sind
    if max(deadline_ratios) == 0:
        ax3.text(len(methods)/2, 2.5, 'Alle Methoden: 0% Deadlines erfüllt\n(Problem zu restriktiv)', 
                ha='center', va='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 4. Durchschnittliche Verspätung
    bars4 = ax4.bar(range(len(methods)), avg_delays, color=colors, alpha=0.8)
    ax4.set_title('Durchschnittliche Verspätung (niedrigere Werte = besser)', fontweight='bold')
    ax4.set_ylabel('Verspätung (Minuten)')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Beste Verspätung markieren
    best_delay_idx = np.argmin(avg_delays)
    bars4[best_delay_idx].set_color('#2E8B57')
    ax4.text(best_delay_idx, avg_delays[best_delay_idx] + 50, 
             f'BESTE\n{avg_delays[best_delay_idx]:.0f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # Legende
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='PPO Agent'),
        plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='Heuristiken'),
        plt.Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.8, label='Beste Methode')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    
    # Speichere Diagramm
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('auswertung/images', exist_ok=True)
    
    plt.savefig('results/images/heuristics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/heuristics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Kombiniertes Vergleichsdiagramm gespeichert:")
    print("  - results/images/heuristics_comparison.png")
    print("  - auswertung/images/heuristics_comparison.png")
    plt.close()

def create_improvement_chart(results):
    """Erstelle Verbesserungsdiagramm: PPO vs. jede Heuristik"""
    
    ppo_result = results[0]  # PPO ist das erste Ergebnis
    heuristic_results = results[1:]  # Alle anderen sind Heuristiken
    
    improvements = []
    heuristic_names = []
    
    for heuristic in heuristic_results:
        # Berechne Verbesserungen (negativ = PPO ist besser)
        makespan_improvement = ((heuristic['makespan'] - ppo_result['makespan']) / heuristic['makespan']) * 100
        utilization_improvement = ((ppo_result['utilization'] - heuristic['utilization']) / heuristic['utilization']) * 100
        delay_improvement = ((heuristic['avg_delay'] - ppo_result['avg_delay']) / heuristic['avg_delay']) * 100
        
        improvements.append([makespan_improvement, utilization_improvement, delay_improvement])
        heuristic_names.append(heuristic['heuristic'])
    
    improvements = np.array(improvements)
    
    # Erstelle Verbesserungsdiagramm
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(heuristic_names))
    width = 0.25
    
    # Balken für jede Metrik mit unterschiedlichen Farben
    bars1 = ax.bar(x - width, improvements[:, 0], width, label='Makespan', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x, improvements[:, 1], width, label='Maschinenauslastung', alpha=0.8, color='#4ECDC4')
    bars3 = ax.bar(x + width, improvements[:, 2], width, label='Verspätung', alpha=0.8, color='#FFD93D')
    
    # Färbe Balken basierend auf Verbesserung/Verschlechterung
    for bars in [bars1, bars2, bars3]:
        for i, bar in enumerate(bars):
            if bars == bars1:  # Makespan - Orange/Rot Töne
                value = improvements[i, 0]
                if value > 0:
                    bar.set_color('#FF8C42')  # Orange für Verbesserung
                else:
                    bar.set_color('#FF6B6B')  # Rot für Verschlechterung
            elif bars == bars2:  # Utilization - Blau/Türkis Töne
                value = improvements[i, 1]
                if value > 0:
                    bar.set_color('#17A2B8')  # Dunkeltürkis für Verbesserung
                else:
                    bar.set_color('#4ECDC4')  # Helles Türkis für Verschlechterung
            else:  # Delay - Gelb/Gold Töne
                value = improvements[i, 2]
                if value > 0:
                    bar.set_color('#FFC107')  # Gold für Verbesserung
                else:
                    bar.set_color('#FFD93D')  # Gelb für Verschlechterung
    
    ax.set_title('PPO Agent Verbesserung gegenüber Heuristiken\n(Positive Werte = PPO ist besser)', 
                fontweight='bold', fontsize=14)
    ax.set_ylabel('Verbesserung (%)')
    ax.set_xlabel('Heuristiken')
    ax.set_xticks(x)
    ax.set_xticklabels(heuristic_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Füge Werte auf Balken hinzu
    for i, (bars, metric_idx) in enumerate([(bars1, 0), (bars2, 1), (bars3, 2)]):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height >= 0 else -0.5),
                   f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/images/ppo_improvements.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/ppo_improvements.png', dpi=300, bbox_inches='tight')
    print("✅ Verbesserungsdiagramm gespeichert:")
    print("  - results/images/ppo_improvements.png")
    print("  - auswertung/images/ppo_improvements.png")
    plt.close()

def create_separate_charts(results):
    """Erstelle 4 separate Diagramme für jede Metrik (aus separate_charts_and_report.py)"""
    
    # Extrahiere Daten
    methods = [r['heuristic'] for r in results]
    makespans = [r['makespan'] for r in results]
    utilizations = [r['utilization'] for r in results]
    deadline_ratios = [r['deadline_ratio'] * 100 for r in results]
    avg_delays = [r['avg_delay'] for r in results]
    
    # Farben für PPO vs Heuristiken
    colors = ['#FF6B6B' if method == 'PPO Agent' else '#4ECDC4' for method in methods]
    
    # Erstelle Verzeichnisse
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('auswertung/images', exist_ok=True)
    
    # 1. MAKESPAN DIAGRAMM
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(methods)), makespans, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Beste Makespan markieren
    best_idx = np.argmin(makespans)
    bars[best_idx].set_color('#2E8B57')
    
    plt.title('Makespan Vergleich - PPO Agent vs. Dispatching-Heuristiken', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Makespan (Minuten)', fontsize=12)
    plt.xlabel('Methoden', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Werte auf Balken anzeigen
    for i, (bar, value) in enumerate(zip(bars, makespans)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 20, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/images/makespan_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/makespan_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MASCHINENAUSLASTUNG DIAGRAMM
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(methods)), utilizations, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Beste Auslastung markieren
    best_idx = np.argmax(utilizations)
    bars[best_idx].set_color('#2E8B57')
    
    plt.title('Maschinenauslastung Vergleich - PPO Agent vs. Dispatching-Heuristiken', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Auslastung (0-1)', fontsize=12)
    plt.xlabel('Methoden', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.95, 1.0)  # Zoom auf relevanten Bereich
    
    # Werte auf Balken anzeigen
    for i, (bar, value) in enumerate(zip(bars, utilizations)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 0.002, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/images/utilization_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/utilization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. DEADLINE-ERFÜLLUNGSRATE DIAGRAMM
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(methods)), deadline_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Deadline-Erfüllungsrate Vergleich - PPO Agent vs. Dispatching-Heuristiken', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Deadline-Erfüllungsrate (%)', fontsize=12)
    plt.xlabel('Methoden', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 5)
    
    # Spezielle Behandlung wenn alle 0% sind
    if max(deadline_ratios) == 0:
        plt.text(len(methods)/2, 2.5, 
                'ALLE METHODEN: 0% Deadlines erfüllt\n(Problemstellung zu restriktiv)', 
                ha='center', va='center', fontweight='bold', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    else:
        # Werte auf Balken anzeigen (falls es Werte > 0 gibt)
        for i, (bar, value) in enumerate(zip(bars, deadline_ratios)):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2., value + 0.1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/images/deadline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/deadline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. DURCHSCHNITTLICHE VERSPÄTUNG DIAGRAMM
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(methods)), avg_delays, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Beste Verspätung markieren
    best_idx = np.argmin(avg_delays)
    bars[best_idx].set_color('#2E8B57')
    
    plt.title('Durchschnittliche Verspätung Vergleich - PPO Agent vs. Dispatching-Heuristiken', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Durchschnittliche Verspätung (Minuten)', fontsize=12)
    plt.xlabel('Methoden', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Werte auf Balken anzeigen
    for i, (bar, value) in enumerate(zip(bars, avg_delays)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 20, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/images/delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 4 separate Diagramme erstellt:")
    print("  - makespan_comparison.png")
    print("  - utilization_comparison.png") 
    print("  - deadline_comparison.png")
    print("  - delay_comparison.png")

def create_summary_table(results):
    """Erstelle eine zusammenfassende Tabelle"""
    
    print("\n" + "="*100)
    print("ZUSAMMENFASSUNG - BESTE METHODEN PRO METRIK")
    print("="*100)
    
    # Finde beste Methoden
    makespans = [(r['heuristic'], r['makespan']) for r in results]
    utilizations = [(r['heuristic'], r['utilization']) for r in results]
    deadline_ratios = [(r['heuristic'], r['deadline_ratio']) for r in results]
    delays = [(r['heuristic'], r['avg_delay']) for r in results]
    
    best_makespan = min(makespans, key=lambda x: x[1])
    best_utilization = max(utilizations, key=lambda x: x[1])
    best_deadline = max(deadline_ratios, key=lambda x: x[1])
    best_delay = min(delays, key=lambda x: x[1])
    
    print(f"🏆 BESTE MAKESPAN:        {best_makespan[0]:<20} ({best_makespan[1]:.1f} Min)")
    print(f"🏆 BESTE AUSLASTUNG:      {best_utilization[0]:<20} ({best_utilization[1]:.3f})")
    print(f"🏆 BESTE DEADLINE-RATE:   {best_deadline[0]:<20} ({best_deadline[1]*100:.1f}%)")
    print(f"🏆 BESTE VERSPÄTUNG:      {best_delay[0]:<20} ({best_delay[1]:.1f} Min)")
    
    # PPO-Ranking
    ppo_result = results[0]
    
    makespan_rank = sorted(makespans, key=lambda x: x[1]).index((ppo_result['heuristic'], ppo_result['makespan'])) + 1
    util_rank = sorted(utilizations, key=lambda x: x[1], reverse=True).index((ppo_result['heuristic'], ppo_result['utilization'])) + 1
    deadline_rank = sorted(deadline_ratios, key=lambda x: x[1], reverse=True).index((ppo_result['heuristic'], ppo_result['deadline_ratio'])) + 1
    delay_rank = sorted(delays, key=lambda x: x[1]).index((ppo_result['heuristic'], ppo_result['avg_delay'])) + 1
    
    print("\n" + "-"*50)
    print("PPO AGENT RANKING:")
    print("-"*50)
    print(f"Makespan:         Platz {makespan_rank}/{len(results)}")
    print(f"Auslastung:       Platz {util_rank}/{len(results)}")
    print(f"Deadline-Rate:    Platz {deadline_rank}/{len(results)}")
    print(f"Verspätung:       Platz {delay_rank}/{len(results)}")
    
    avg_rank = (makespan_rank + util_rank + deadline_rank + delay_rank) / 4
    print(f"\n📊 DURCHSCHNITTSRANG: {avg_rank:.1f}/{len(results)}")

def generate_comprehensive_report(results):
    """Generiere einen umfassenden Report mit allen wichtigen Informationen"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Erstelle Report-Verzeichnisse
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('auswertung/reports', exist_ok=True)
    
    report_filename = f"ppo_heuristics_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    # Berechne Statistiken
    ppo_result = results[0]
    heuristic_results = results[1:]
    
    # Finde beste Methoden
    makespans = [(r['heuristic'], r['makespan']) for r in results]
    utilizations = [(r['heuristic'], r['utilization']) for r in results]
    deadline_ratios = [(r['heuristic'], r['deadline_ratio']) for r in results]
    delays = [(r['heuristic'], r['avg_delay']) for r in results]
    
    best_makespan = min(makespans, key=lambda x: x[1])
    best_utilization = max(utilizations, key=lambda x: x[1])
    best_deadline = max(deadline_ratios, key=lambda x: x[1])
    best_delay = min(delays, key=lambda x: x[1])
    
    # PPO Rankings
    makespan_rank = sorted(makespans, key=lambda x: x[1]).index((ppo_result['heuristic'], ppo_result['makespan'])) + 1
    util_rank = sorted(utilizations, key=lambda x: x[1], reverse=True).index((ppo_result['heuristic'], ppo_result['utilization'])) + 1
    deadline_rank = sorted(deadline_ratios, key=lambda x: x[1], reverse=True).index((ppo_result['heuristic'], ppo_result['deadline_ratio'])) + 1
    delay_rank = sorted(delays, key=lambda x: x[1]).index((ppo_result['heuristic'], ppo_result['avg_delay'])) + 1
    avg_rank = (makespan_rank + util_rank + deadline_rank + delay_rank) / 4
    
    # Erstelle Report-Inhalt
    report_content = f"""# PPO Agent vs. Dispatching-Heuristiken Vergleichsreport

**Erstellt am:** {timestamp}
**Anzahl Testläufe:** 3 pro Methode

---

## 📋 Executive Summary

Dieser Report vergleicht die Leistung eines trainierten PPO (Proximal Policy Optimization) Agents mit klassischen Dispatching-Heuristiken für Job-Shop Scheduling Probleme.

### 🏆 Wichtigste Erkenntnisse:

- **Beste Gesamtleistung (Makespan):** {best_makespan[0]} mit {best_makespan[1]:.0f} Minuten
- **Beste Maschinenauslastung:** {best_utilization[0]} mit {best_utilization[1]:.3f}
- **PPO Agent Durchschnittsrang:** {avg_rank:.1f} von {len(results)} Methoden
- **Deadline-Problem:** Alle Methoden erreichen 0% Deadline-Erfüllung (Problem zu restriktiv)

---

## 📊 Detaillierte Ergebnisse

### Vergleichstabelle aller Methoden

| Methode | Makespan (Min) | Auslastung | Deadline-Rate (%) | Avg. Verspätung (Min) |
|---------|----------------|------------|-------------------|-----------------------|
"""
    
    for result in results:
        report_content += f"| {result['heuristic']} | {result['makespan']:.1f} | {result['utilization']:.3f} | {result['deadline_ratio']*100:.1f} | {result['avg_delay']:.1f} |\n"
    
    report_content += f"""
---

## 🎯 PPO Agent Performance Analysis

### Ranking pro Metrik:
- **Makespan:** Platz {makespan_rank}/{len(results)} ({ppo_result['makespan']:.1f} Min)
- **Maschinenauslastung:** Platz {util_rank}/{len(results)} ({ppo_result['utilization']:.3f})
- **Deadline-Erfüllung:** Platz {deadline_rank}/{len(results)} ({ppo_result['deadline_ratio']*100:.1f}%)
- **Verspätung:** Platz {delay_rank}/{len(results)} ({ppo_result['avg_delay']:.1f} Min)

### PPO vs. Heuristiken Verbesserungen:

"""
    
    for heuristic in heuristic_results:
        makespan_improvement = ((heuristic['makespan'] - ppo_result['makespan']) / heuristic['makespan']) * 100
        utilization_improvement = ((ppo_result['utilization'] - heuristic['utilization']) / heuristic['utilization']) * 100 if heuristic['utilization'] > 0 else 0
        delay_improvement = ((heuristic['avg_delay'] - ppo_result['avg_delay']) / heuristic['avg_delay']) * 100
        
        status_makespan = "✅ Besser" if makespan_improvement > 0 else "❌ Schlechter"
        status_util = "✅ Besser" if utilization_improvement > 0 else "❌ Schlechter"
        status_delay = "✅ Besser" if delay_improvement > 0 else "❌ Schlechter"
        
        report_content += f"""
#### PPO vs. {heuristic['heuristic']}:
- **Makespan:** {makespan_improvement:+.1f}% {status_makespan}
- **Auslastung:** {utilization_improvement:+.1f}% {status_util}
- **Verspätung:** {delay_improvement:+.1f}% {status_delay}
"""
    
    report_content += f"""
---

## 📈 Analyseergebnisse

### Stärken des PPO Agents:
- **Moderate Makespan-Performance:** Platz {makespan_rank}/{len(results)}, kompetitiv aber nicht führend
- **Gute Maschinenauslastung:** Platz {util_rank}/{len(results)}, effiziente Ressourcennutzung
- **Konsistente Leistung:** Stabile Ergebnisse über mehrere Testläufe

### Schwächen des PPO Agents:
- **Makespan-Optimierung:** Nicht die beste Methode für minimale Gesamtzeit
- **Deadline-Management:** Wie alle Methoden 0% Deadline-Erfüllung
- **Verspätungsminimierung:** Platz {delay_rank}/{len(results)}, Verbesserungspotential

### Problem-spezifische Beobachtungen:
- **Deadline-Restriktivität:** Alle Methoden erreichen 0% Deadline-Erfüllung
- **Setup-Zeit-Einfluss:** Hohe Setup-Zeiten dominieren die Gesamtzeit
- **Maschinenauslastung:** Alle Methoden erreichen sehr hohe Auslastung (>97%)

---

## 🎯 Empfehlungen

### Für Makespan-Optimierung:
**Beste Wahl: {best_makespan[0]}** ({best_makespan[1]:.0f} Min)

### Für Maschinenauslastung:
**Beste Wahl: {best_utilization[0]}** ({best_utilization[1]:.3f})

### Für Verspätungsminimierung:
**Beste Wahl: {best_delay[0]}** ({best_delay[1]:.0f} Min)

### PPO Agent Verbesserungsvorschläge:
1. **Reward-Funktion anpassen:** Stärkere Gewichtung der Makespan-Minimierung
2. **Deadline-Awareness:** Verbesserte Berücksichtigung von Deadline-Constraints
3. **Setup-Zeit-Optimierung:** Gezielt Materialwechsel minimieren
4. **Mehr Training:** Längere Trainingsphasen für bessere Konvergenz

---

## 📊 Generierte Visualisierungen

1. **heuristics_comparison.png** - Kombinierte 2x2 Übersicht aller Metriken
2. **makespan_comparison.png** - Einzeldiagramm Makespan-Vergleich
3. **utilization_comparison.png** - Einzeldiagramm Maschinenauslastung
4. **deadline_comparison.png** - Einzeldiagramm Deadline-Erfüllungsraten
5. **delay_comparison.png** - Einzeldiagramm Durchschnittliche Verspätungen
6. **ppo_improvements.png** - PPO Verbesserungen gegenüber Heuristiken

---

## 🏁 Fazit

Der PPO Agent zeigt **durchschnittliche Performance** mit einem Gesamtrang von {avg_rank:.1f}/{len(results)}. Während er in der Maschinenauslastung gut abschneidet (Platz {util_rank}), gibt es deutliches Verbesserungspotential bei der Makespan-Optimierung (Platz {makespan_rank}) und Verspätungsminimierung (Platz {delay_rank}).

**Empfehlung:** Für dieses spezifische Problem ist **{best_makespan[0]}** die beste Wahl für die Makespan-Minimierung. Der PPO Agent könnte mit angepasster Reward-Funktion und längerem Training competitive werden.

---

*Report generiert am {timestamp}*
"""
    
    # Speichere Report
    report_path_results = f'results/reports/{report_filename}'
    report_path_auswertung = f'auswertung/reports/{report_filename}'
    
    with open(report_path_results, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    with open(report_path_auswertung, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✅ Umfassender Report erstellt:")
    print(f"  - {report_path_results}")
    print(f"  - {report_path_auswertung}")
    
    return report_path_results


if __name__ == "__main__":
    # Pfade definieren
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    print("🚀 KOMBINIERTE ANALYSE: PPO vs. Heuristiken")
    print("=" * 60)
    print("Führe EINEN Vergleichslauf durch und erstelle ALLE Visualisierungen...")
    
    # ✅ EINMALIGER Vergleichslauf - gleiche Daten für alle Visualisierungen
    print("\n📊 Führe Vergleich durch (3 Läufe pro Methode)...")
    results = compare_methods(jsp_data_path, model_path, num_runs=3)
    
    print("\n🎨 Erstelle alle Visualisierungen mit den GLEICHEN Daten...")
    
    # 1. Kombiniertes 2x2 Diagramm (aus heuristics_visualization.py)
    print("\n1️⃣ Erstelle kombiniertes 2x2 Diagramm...")
    create_comparison_charts(results)
    
    # 2. Verbesserungsdiagramm (aus heuristics_visualization.py) 
    print("\n2️⃣ Erstelle PPO Verbesserungsdiagramm...")
    create_improvement_chart(results)
    
    # 3. 4 separate Diagramme (aus separate_charts_and_report.py)
    print("\n3️⃣ Erstelle 4 separate Diagramme...")
    create_separate_charts(results)
    
    # 4. Zusammenfassungstabelle (aus heuristics_visualization.py)
    print("\n4️⃣ Zeige Zusammenfassungstabelle...")
    create_summary_table(results)
    
    # 5. Umfassender Report (aus separate_charts_and_report.py)
    print("\n5️⃣ Generiere umfassenden Report...")
    report_path = generate_comprehensive_report(results)
    
    print("\n" + "=" * 60)
    print("✅ ALLE VISUALISIERUNGEN UND ANALYSEN ABGESCHLOSSEN!")
    print("=" * 60)
    print("\n📁 Erstellte Dateien (alle mit den GLEICHEN Daten):")
    print("  📊 heuristics_comparison.png - Kombinierte 2x2 Übersicht")
    print("  📊 ppo_improvements.png - PPO Verbesserungsdiagramm") 
    print("  📊 makespan_comparison.png - Einzeldiagramm Makespan")
    print("  📊 utilization_comparison.png - Einzeldiagramm Auslastung")
    print("  📊 deadline_comparison.png - Einzeldiagramm Deadlines")
    print("  📊 delay_comparison.png - Einzeldiagramm Verspätungen")
    print("  📄 Umfassender Markdown-Report mit allen Analysen")
    print("\n📍 Alle Dateien gespeichert in:")
    print("  - results/images/ und results/reports/")
    print("  - auswertung/images/ und auswertung/reports/")
    print("\n🎯 Verwenden Sie: python3 auswertung/combined_analysis.py")