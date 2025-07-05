import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auswertung.ppo_heuristics_comparison import compare_methods

def create_comparison_charts(results):
    """Erstelle einfache Vergleichsdiagramme mit einheitlichen Farben und Sortierung"""
    
    # Sortiere Ergebnisse nach Makespan (beste Performance zuerst)
    sorted_results = sorted(results, key=lambda x: x['makespan'])
    
    # Extrahiere Daten für Visualisierung
    methods = [r['heuristic'] for r in sorted_results]
    makespans = [r['makespan'] for r in sorted_results]
    utilizations = [r['utilization'] for r in sorted_results]
    deadline_ratios = [r['deadline_ratio'] * 100 for r in sorted_results]
    avg_delays = [r['avg_delay'] for r in sorted_results]
    
    # Einheitliche Farbe für alle Methoden
    uniform_color = '#4ECDC4'  # Türkis für alle Balken
    
    # Erstelle 2x2 Subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PPO Agent vs. Dispatching-Heuristiken Vergleich\n(Sortiert nach Makespan-Performance)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Makespan Vergleich (bereits sortiert)
    bars1 = ax1.bar(range(len(methods)), makespans, color=uniform_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Makespan (niedrigere Werte = besser)\nSortiert: Beste → Schlechteste', fontweight='bold')
    ax1.set_ylabel('Makespan (Minuten)')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Beste Makespan markieren (ist bereits der erste)
    bars1[0].set_edgecolor('#2E8B57')
    bars1[0].set_linewidth(3)
    ax1.text(0, makespans[0] + 50, 
             f'BESTE\n{makespans[0]:.0f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # Werte auf allen Balken anzeigen
    for i, (bar, value) in enumerate(zip(bars1, makespans)):
        if i > 0:  # Nicht für den ersten (bereits beschriftet)
            ax1.text(bar.get_x() + bar.get_width()/2., value + 20, 
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Maschinenauslastung (sortiert nach Auslastung)
    util_sorted_indices = sorted(range(len(sorted_results)), 
                                key=lambda i: sorted_results[i]['utilization'], reverse=True)
    util_methods = [methods[i] for i in util_sorted_indices]
    util_values = [utilizations[i] for i in util_sorted_indices]
    util_colors = [uniform_color] * len(methods)
    
    bars2 = ax2.bar(range(len(util_methods)), util_values, color=uniform_color, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax2.set_title('Maschinenauslastung (höhere Werte = besser)\nSortiert: Beste → Schlechteste', fontweight='bold')
    ax2.set_ylabel('Auslastung (0-1)')
    ax2.set_xticks(range(len(util_methods)))
    ax2.set_xticklabels(util_methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.95, 1.0)
    
    # Beste Auslastung markieren
    bars2[0].set_edgecolor('#2E8B57')
    bars2[0].set_linewidth(3)
    ax2.text(0, util_values[0] + 0.002, 
             f'BESTE\n{util_values[0]:.3f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # Werte anzeigen
    for i, (bar, value) in enumerate(zip(bars2, util_values)):
        if i > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., value + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Deadline-Erfüllungsrate (alle sind 0%, also verwende ursprüngliche Reihenfolge)
    bars3 = ax3.bar(range(len(methods)), deadline_ratios, color=uniform_color, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax3.set_title('Deadline-Erfüllungsrate (höhere Werte = besser)', fontweight='bold')
    ax3.set_ylabel('Erfüllungsrate (%)')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 5)
    
    # Spezielle Behandlung da alle 0% sind
    if max(deadline_ratios) == 0:
        ax3.text(len(methods)/2, 2.5, 
                'ALLE METHODEN: 0% Deadlines erfüllt\n(Problemstellung zu restriktiv)', 
                ha='center', va='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Durchschnittliche Verspätung (sortiert nach Verspätung)
    delay_sorted_indices = sorted(range(len(sorted_results)), 
                                 key=lambda i: sorted_results[i]['avg_delay'])
    delay_methods = [methods[i] for i in delay_sorted_indices]
    delay_values = [avg_delays[i] for i in delay_sorted_indices]
    delay_colors = [uniform_color] * len(methods)
    
    bars4 = ax4.bar(range(len(delay_methods)), delay_values, color=uniform_color, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax4.set_title('Durchschnittliche Verspätung (niedrigere Werte = besser)\nSortiert: Beste → Schlechteste', 
                  fontweight='bold')
    ax4.set_ylabel('Verspätung (Minuten)')
    ax4.set_xticks(range(len(delay_methods)))
    ax4.set_xticklabels(delay_methods, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Beste Verspätung markieren
    bars4[0].set_edgecolor('#2E8B57')
    bars4[0].set_linewidth(3)
    ax4.text(0, delay_values[0] + 50, 
             f'BESTE\n{delay_values[0]:.0f}', 
             ha='center', va='bottom', fontweight='bold', color='#2E8B57')
    
    # Werte anzeigen
    for i, (bar, value) in enumerate(zip(bars4, delay_values)):
        if i > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., value + 20, 
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Verbesserte Legende
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=uniform_color, alpha=0.8, label='Alle Methoden'),
        plt.Line2D([0], [0], color='#2E8B57', linewidth=3, label='Bester Wert pro Metrik')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    
    # Speichere Diagramm
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('auswertung/images', exist_ok=True)
    
    plt.savefig('results/images/heuristics_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/heuristics_comparison_sorted.png', dpi=300, bbox_inches='tight')
    print("✅ Sortiertes Vergleichsdiagramm gespeichert:")
    print("  - results/images/heuristics_comparison_sorted.png")
    print("  - auswertung/images/heuristics_comparison_sorted.png")
    plt.close()

def create_separate_charts_sorted(results):
    """Erstelle 4 separate, nach Performance sortierte Diagramme"""
    
    # Erstelle Verzeichnisse
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('auswertung/images', exist_ok=True)
    
    # 1. MAKESPAN DIAGRAMM (sortiert nach Makespan)
    makespan_sorted = sorted(results, key=lambda x: x['makespan'])
    methods = [r['heuristic'] for r in makespan_sorted]
    makespans = [r['makespan'] for r in makespan_sorted]
    
    plt.figure(figsize=(14, 8))
    
    # Farbverlauf von grün (beste) zu rot (schlechteste)
    uniform_color = '#4ECDC4'  # Einheitliche Farbe für alle Balken
    bars = plt.bar(range(len(methods)), makespans, color=uniform_color, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Beste Methode hervorheben
    bars[0].set_edgecolor('#2E8B57')
    bars[0].set_linewidth(3)
    
    plt.title('Makespan Vergleich - Sortiert nach Performance (Beste → Schlechteste)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Makespan (Minuten)', fontsize=12)
    plt.xlabel('Methoden (sortiert nach Makespan)', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Werte anzeigen (ohne Ranking-Nummern)
    for i, (bar, value, method) in enumerate(zip(bars, makespans, methods)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 20, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # PPO Position hervorheben
        if 'PPO' in method:
            plt.text(bar.get_x() + bar.get_width()/2., value - 100, 
                    'PPO', ha='center', va='top', fontweight='bold', 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="darkblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/images/makespan_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/makespan_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MASCHINENAUSLASTUNG DIAGRAMM (sortiert nach Auslastung)
    util_sorted = sorted(results, key=lambda x: x['utilization'], reverse=True)
    methods = [r['heuristic'] for r in util_sorted]
    utilizations = [r['utilization'] for r in util_sorted]
    
    plt.figure(figsize=(14, 8))
    
    colors = uniform_color
    bars = plt.bar(range(len(methods)), utilizations, color=uniform_color, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Beste Methode hervorheben
    bars[0].set_edgecolor('#2E8B57')
    bars[0].set_linewidth(3)
    
    plt.title('Maschinenauslastung Vergleich - Sortiert nach Performance (Beste → Schlechteste)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Auslastung (0-1)', fontsize=12)
    plt.xlabel('Methoden (sortiert nach Auslastung)', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0.95, 1.0)
    
    # Werte anzeigen (ohne Ranking-Nummern)
    for i, (bar, value, method) in enumerate(zip(bars, utilizations, methods)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 0.001, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # PPO Position hervorheben
        if 'PPO' in method:
            plt.text(bar.get_x() + bar.get_width()/2., value - 0.005, 
                    'PPO', ha='center', va='top', fontweight='bold', 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="darkblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/images/utilization_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/utilization_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. VERSPÄTUNG DIAGRAMM (sortiert nach Verspätung)
    delay_sorted = sorted(results, key=lambda x: x['avg_delay'])
    methods = [r['heuristic'] for r in delay_sorted]
    avg_delays = [r['avg_delay'] for r in delay_sorted]
    
    plt.figure(figsize=(14, 8))
    
    colors = uniform_color
    bars = plt.bar(range(len(methods)), avg_delays, color=uniform_color, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Beste Methode hervorheben
    bars[0].set_edgecolor('#2E8B57')
    bars[0].set_linewidth(3)
    
    plt.title('Durchschnittliche Verspätung Vergleich - Sortiert nach Performance (Beste → Schlechteste)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Durchschnittliche Verspätung (Minuten)', fontsize=12)
    plt.xlabel('Methoden (sortiert nach Verspätung)', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Werte anzeigen (ohne Ranking-Nummern)
    for i, (bar, value, method) in enumerate(zip(bars, avg_delays, methods)):
        plt.text(bar.get_x() + bar.get_width()/2., value + 20, 
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # PPO Position hervorheben
        if 'PPO' in method:
            plt.text(bar.get_x() + bar.get_width()/2., value - 80, 
                    'PPO', ha='center', va='top', fontweight='bold', 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="darkblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/images/delay_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/delay_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. DEADLINE-ERFÜLLUNGSRATE (alle 0%, zeige Info)
    plt.figure(figsize=(14, 8))
    methods = [r['heuristic'] for r in results]
    deadline_ratios = [r['deadline_ratio'] * 100 for r in results]
    
    # Verwende neutrale Farbe da alle gleich sind
    bars = plt.bar(range(len(methods)), deadline_ratios, color='lightgray', alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    plt.title('Deadline-Erfüllungsrate Vergleich - Alle Methoden: 0%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Deadline-Erfüllungsrate (%)', fontsize=12)
    plt.xlabel('Methoden', fontsize=12)
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 5)
    
    # Zentrale Nachricht
    plt.text(len(methods)/2, 2.5, 
            'ALLE METHODEN: 0% Deadlines erfüllt\n(Problemstellung zu restriktiv)\n\nKeine Sortierung möglich - alle Werte identisch', 
            ha='center', va='center', fontweight='bold', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/images/deadline_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.savefig('auswertung/images/deadline_comparison_sorted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 4 separate, sortierte Diagramme erstellt:")
    print("  - makespan_comparison_sorted.png")
    print("  - utilization_comparison_sorted.png") 
    print("  - delay_comparison_sorted.png")
    print("  - deadline_comparison_sorted.png")

def create_performance_ranking_table(results):
    """Erstelle eine Ranking-Tabelle für alle Metriken"""
    
    print("\n" + "="*120)
    print("PERFORMANCE RANKING - ALLE METHODEN NACH METRIKEN SORTIERT")
    print("="*120)
    
    # Makespan Ranking
    makespan_ranking = sorted(results, key=lambda x: x['makespan'])
    print("\n🏆 MAKESPAN RANKING (niedrigere Werte = besser):")
    print("-" * 60)
    for i, result in enumerate(makespan_ranking):
        ppo_marker = " ← PPO AGENT" if "PPO" in result['heuristic'] else ""
        print(f"#{i+1:2d}. {result['heuristic']:<20} {result['makespan']:>8.1f} Min{ppo_marker}")
    
    # Utilization Ranking
    util_ranking = sorted(results, key=lambda x: x['utilization'], reverse=True)
    print("\n🏆 MASCHINENAUSLASTUNG RANKING (höhere Werte = besser):")
    print("-" * 60)
    for i, result in enumerate(util_ranking):
        ppo_marker = " ← PPO AGENT" if "PPO" in result['heuristic'] else ""
        print(f"#{i+1:2d}. {result['heuristic']:<20} {result['utilization']:>8.3f}{ppo_marker}")
    
    # Delay Ranking
    delay_ranking = sorted(results, key=lambda x: x['avg_delay'])
    print("\n🏆 VERSPÄTUNG RANKING (niedrigere Werte = besser):")
    print("-" * 60)
    for i, result in enumerate(delay_ranking):
        ppo_marker = " ← PPO AGENT" if "PPO" in result['heuristic'] else ""
        print(f"#{i+1:2d}. {result['heuristic']:<20} {result['avg_delay']:>8.1f} Min{ppo_marker}")
    
    # PPO Summary
    ppo_result = next(r for r in results if "PPO" in r['heuristic'])
    makespan_rank = makespan_ranking.index(ppo_result) + 1
    util_rank = util_ranking.index(ppo_result) + 1
    delay_rank = delay_ranking.index(ppo_result) + 1
    avg_rank = (makespan_rank + util_rank + delay_rank) / 3
    
    print("\n" + "="*60)
    print("PPO AGENT ZUSAMMENFASSUNG:")
    print("="*60)
    print(f"Makespan:        Platz {makespan_rank:2d}/{len(results)}")
    print(f"Auslastung:      Platz {util_rank:2d}/{len(results)}")
    print(f"Verspätung:      Platz {delay_rank:2d}/{len(results)}")
    print(f"Deadline-Rate:   Platz  -/{len(results)} (alle 0%)")
    print(f"\n📊 DURCHSCHNITTSRANG: {avg_rank:.1f}/{len(results)}")
    
    # Performance-Kategorien
    if avg_rank <= len(results) * 0.33:
        category = "🥇 TOP-PERFORMER"
    elif avg_rank <= len(results) * 0.66:
        category = "🥈 MITTELFELD"
    else:
        category = "🥉 VERBESSERUNGSBEDARF"
    
    print(f"Kategorie: {category}")

if __name__ == "__main__":
    # Pfade definieren
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    
    jsp_data_path = os.path.join(main_dir, "data.json")
    model_path = os.path.join(main_dir, "results/models/gym_ppo_model_20250626_153446.pt")
    
    print("🚀 VERBESSERTE ANALYSE: PPO vs. Heuristiken mit Sortierung")
    print("=" * 60)
    print("- Einheitliche Farbe für alle Methoden")
    print("- Sortierung nach Performance-Ranking")
    print("- PPO-Position wird hervorgehoben")
    print("- Ohne Ranking-Nummern auf Balken")
    print()
    
    # Einmaliger Vergleichslauf
    print("📊 Führe Vergleich durch (3 Läufe pro Methode)...")
    results = compare_methods(jsp_data_path, model_path, num_runs=3)
    
    print("\n🎨 Erstelle sortierte Visualisierungen...")
    
    # 1. Kombiniertes sortiertes Diagramm
    print("\n1️⃣ Erstelle kombiniertes sortiertes 2x2 Diagramm...")
    create_comparison_charts(results)
    
    # 2. Separate sortierte Diagramme
    print("\n2️⃣ Erstelle 4 separate sortierte Diagramme...")
    create_separate_charts_sorted(results)
    
    # 3. Ranking-Tabelle
    print("\n3️⃣ Zeige Performance-Ranking...")
    create_performance_ranking_table(results)
    
    print("\n" + "=" * 60)
    print("✅ ALLE SORTIERTEN VISUALISIERUNGEN ABGESCHLOSSEN!")
    print("=" * 60)
    print("\n📁 Neue sortierte Dateien:")
    print("  📊 heuristics_comparison_sorted.png - Kombinierte 2x2 Übersicht")
    print("  📊 makespan_comparison_sorted.png - Makespan (beste → schlechteste)")
    print("  📊 utilization_comparison_sorted.png - Auslastung (beste → schlechteste)")
    print("  📊 delay_comparison_sorted.png - Verspätung (beste → schlechteste)")
    print("  📊 deadline_comparison_sorted.png - Deadline-Info")
    print("\n🎯 Verbesserungen:")
    print("  ✅ Einheitliche Farbe für alle Balken")
    print("  ✅ Sortierung nach Performance-Ranking")
    print("  ✅ PPO-Position wird klar markiert")
    print("  ✅ Saubere Werte ohne Ranking-Nummern")
    print("  ✅ Detaillierte Ranking-Tabelle")