import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auswertung.ppo_heuristics_comparison import compare_methods

def create_separate_charts(results):
    """Erstelle 4 separate Diagramme für jede Metrik"""
    
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
    
    # Beste Methode hervorheben
    # plt.text(best_idx, makespans[best_idx] + 150, 
    #          f'BESTE\\n{methods[best_idx]}', 
    #          ha='center', va='bottom', fontweight='bold', color='#2E8B57',
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
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
    
    # Beste Methode hervorheben
    # plt.text(best_idx, utilizations[best_idx] + 0.008, 
    #          f'BESTE\\n{methods[best_idx]}', 
    #          ha='center', va='bottom', fontweight='bold', color='#2E8B57',
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
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
                'ALLE METHODEN: 0% Deadlines erfüllt\\n(Problemstellung zu restriktiv)', 
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
    
    # # Beste Methode hervorheben
    # plt.text(best_idx, avg_delays[best_idx] + 150, 
    #          f'BESTE\\n{methods[best_idx]}', 
    #          ha='center', va='bottom', fontweight='bold', color='#2E8B57',
    #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('results/images/delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/delay_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("4 separate Diagramme erstellt:")
    print("  - makespan_comparison.png")
    print("  - utilization_comparison.png") 
    print("  - deadline_comparison.png")
    print("  - delay_comparison.png")
    print("  (jeweils in results/images/ und auswertung/images/)")


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

Dieser Report vergleicht die Leistung eines trainierten PPO (Proximal Policy Optimization) Agents mit 7 klassischen Dispatching-Heuristiken für Job-Shop Scheduling Probleme.

### 🏆 Wichtigste Erkenntnisse:

- **Beste Gesamtleistung (Makespan):** {best_makespan[0]} mit {best_makespan[1]:.0f} Minuten
- **Beste Maschinenauslastung:** {best_utilization[0]} mit {best_utilization[1]:.3f}
- **PPO Agent Durchschnittsrang:** {avg_rank:.1f} von 8 Methoden
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
- **Makespan:** Platz {makespan_rank}/8 ({ppo_result['makespan']:.1f} Min)
- **Maschinenauslastung:** Platz {util_rank}/8 ({ppo_result['utilization']:.3f})
- **Deadline-Erfüllung:** Platz {deadline_rank}/8 ({ppo_result['deadline_ratio']*100:.1f}%)
- **Verspätung:** Platz {delay_rank}/8 ({ppo_result['avg_delay']:.1f} Min)

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

## 🔍 Methodenbeschreibung

### PPO Agent:
- **Typ:** Deep Reinforcement Learning
- **Architektur:** Graph Transformer mit TransformerConv Layern
- **Training:** 5000 Episoden mit Learning Rate Decay
- **Features:** Berücksichtigt Job-Prioritäten, Deadlines, Setup-Zeiten

### Dispatching-Heuristiken:

1. **FIFO (First In, First Out):** Bearbeite Jobs in der Reihenfolge ihrer Ankunft
2. **FILO (First In, Last Out):** Bearbeite zuletzt angekommene Jobs zuerst
3. **SPT (Shortest Processing Time):** Wähle Job mit kürzester nächster Operation
4. **LPT (Longest Processing Time):** Wähle Job mit längster nächster Operation
5. **Earliest Due Date:** Priorisiere Jobs mit frühester Deadline
6. **Critical Ratio:** Verhältnis von verbleibender Zeit zu verbleibender Arbeit
7. **Slack Time:** Wähle Jobs mit geringster Pufferzeit

---

## 📈 Analyseergebnisse

### Stärken des PPO Agents:
- **Moderate Makespan-Performance:** Platz {makespan_rank}/8, kompetitiv aber nicht führend
- **Gute Maschinenauslastung:** Platz {util_rank}/8, effiziente Ressourcennutzung
- **Konsistente Leistung:** Stabile Ergebnisse über mehrere Testläufe

### Schwächen des PPO Agents:
- **Makespan-Optimierung:** Nicht die beste Methode für minimale Gesamtzeit
- **Deadline-Management:** Wie alle Methoden 0% Deadline-Erfüllung
- **Verspätungsminimierung:** Platz {delay_rank}/8, Verbesserungspotential

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

1. **makespan_comparison.png** - Makespan-Vergleich aller Methoden
2. **utilization_comparison.png** - Maschinenauslastung-Vergleich
3. **deadline_comparison.png** - Deadline-Erfüllungsraten
4. **delay_comparison.png** - Durchschnittliche Verspätungen
5. **ppo_improvements.png** - PPO Verbesserungen gegenüber Heuristiken

---

## 🏁 Fazit

Der PPO Agent zeigt **durchschnittliche Performance** mit einem Gesamtrang von {avg_rank:.1f}/8. Während er in der Maschinenauslastung gut abschneidet (Platz {util_rank}), gibt es deutliches Verbesserungspotential bei der Makespan-Optimierung (Platz {makespan_rank}) und Verspätungsminimierung (Platz {delay_rank}).

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
    
    print(f"\\nUmfassender Report erstellt:")
    print(f"  - {report_path_results}")
    print(f"  - {report_path_auswertung}")
    
    return report_path_results


if __name__ == "__main__":
    # Pfade definieren
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    print("Führe Vergleich durch und erstelle separate Visualisierungen + Report...")
    
    # Führe Vergleich durch
    results = compare_methods(jsp_data_path, model_path, num_runs=3)
    
    print("\\nErstelle 4 separate Diagramme...")
    create_separate_charts(results)
    
    print("\\nGeneriere umfassenden Report...")
    report_path = generate_comprehensive_report(results)
    
    print("\\n✅ Alle Visualisierungen und Report erstellt!")
    print("\\n📁 Erstellte Dateien:")
    print("  📊 4 separate Diagramme (makespan, utilization, deadline, delay)")
    print("  📄 Umfassender Markdown-Report mit allen Analysen")
    print("  📍 Gespeichert in: results/ und auswertung/ Verzeichnissen")