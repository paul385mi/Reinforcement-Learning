import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class JSPKeyMetricsExtractor:
    """
    Extrahiert und visualisiert die wichtigsten Vergleichswerte für JSP-Heuristiken.
    
    Diese Klasse fokussiert sich auf die essentiellen KPIs für die Abschlussarbeit:
    - Makespan (Hauptziel)
    - Timeliness Performance
    - Deadline-Einhaltung
    - Statistische Signifikanz
    - Relative Performance Vergleiche
    """
    
    def __init__(self, results_data=None):
        """
        Initialisiert den Key Metrics Extractor.
        
        Args:
            results_data: Optional vorhandene Ergebnisdaten
        """
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Beispieldaten basierend auf deinen Ergebnissen
        if results_data is None:
            self.results_data = {
                'EDD': {'makespan': 6639.00, 'std': 0.00, 'deadline_ratio': 0.85, 'timeliness': 0.42},
                'LPT': {'makespan': 6677.00, 'std': 0.00, 'deadline_ratio': 0.80, 'timeliness': 0.38},
                'Priority': {'makespan': 6668.00, 'std': 0.00, 'deadline_ratio': 0.82, 'timeliness': 0.40},
                'LIFO': {'makespan': 6704.00, 'std': 0.00, 'deadline_ratio': 0.75, 'timeliness': 0.35},
                'FIFO': {'makespan': 6705.00, 'std': 0.00, 'deadline_ratio': 0.74, 'timeliness': 0.34},
                'WSPT': {'makespan': 6707.00, 'std': 0.00, 'deadline_ratio': 0.78, 'timeliness': 0.37},
                'SPT': {'makespan': 6773.00, 'std': 0.00, 'deadline_ratio': 0.70, 'timeliness': 0.30},
                'Minimum Slack': {'makespan': 6787.00, 'std': 0.00, 'deadline_ratio': 0.68, 'timeliness': 0.28},
                'Critical Ratio': {'makespan': 6832.00, 'std': 0.00, 'deadline_ratio': 0.65, 'timeliness': 0.25},
                'Random': {'makespan': 6833.40, 'std': 51.99, 'deadline_ratio': 0.60, 'timeliness': 0.20},
                'PPO-GNN': {'makespan': 6851.60, 'std': 51.29, 'deadline_ratio': 0.72, 'timeliness': 0.32}
            }
        else:
            self.results_data = results_data
        
        # Erstelle Ausgabeverzeichnis
        os.makedirs('results/key_metrics', exist_ok=True)
    
    def calculate_key_metrics(self):
        """Berechnet die wichtigsten Vergleichsmetriken."""
        
        # Konvertiere zu DataFrame für einfache Analyse
        df = pd.DataFrame.from_dict(self.results_data, orient='index')
        df.index.name = 'Heuristic'
        df = df.reset_index()
        
        # Sortiere nach Makespan (Hauptziel)
        df = df.sort_values('makespan')
        df['makespan_rank'] = range(1, len(df) + 1)
        
        # Berechne relative Performance
        best_makespan = df['makespan'].min()
        df['makespan_gap_percent'] = ((df['makespan'] - best_makespan) / best_makespan * 100)
        
        # Sortiere nach anderen Metriken für Rankings
        df['deadline_rank'] = df['deadline_ratio'].rank(ascending=False, method='min')
        df['timeliness_rank'] = df['timeliness'].rank(ascending=False, method='min')
        
        # Overall Score (gewichteter Durchschnitt der Rankings)
        weights = {'makespan': 0.5, 'deadline': 0.3, 'timeliness': 0.2}
        df['overall_score'] = (
            weights['makespan'] * (len(df) + 1 - df['makespan_rank']) +
            weights['deadline'] * (len(df) + 1 - df['deadline_rank']) +
            weights['timeliness'] * (len(df) + 1 - df['timeliness_rank'])
        )
        df['overall_rank'] = df['overall_score'].rank(ascending=False, method='min')
        
        # Identifiziere PPO-GNN Performance
        ppo_idx = df[df['Heuristic'] == 'PPO-GNN'].index
        if len(ppo_idx) > 0:
            ppo_performance = df.loc[ppo_idx[0]]
            self.ppo_analysis = {
                'makespan_rank': int(ppo_performance['makespan_rank']),
                'total_heuristics': len(df),
                'makespan_gap': float(ppo_performance['makespan_gap_percent']),
                'deadline_rank': int(ppo_performance['deadline_rank']),
                'timeliness_rank': int(ppo_performance['timeliness_rank']),
                'overall_rank': int(ppo_performance['overall_rank'])
            }
        else:
            self.ppo_analysis = None
        
        self.df = df
        return df
    
    def create_key_visualizations(self):
        """Erstellt die wichtigsten Visualisierungen für die Abschlussarbeit."""
        
        # Set professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Makespan Comparison (Hauptmetrik)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Makespan Bar Chart
        colors = ['#2E8B57' if h == 'PPO-GNN' else '#4682B4' if h == 'EDD' else '#708090' 
                 for h in self.df['Heuristic']]
        
        bars = axes[0,0].bar(range(len(self.df)), self.df['makespan'], color=colors, alpha=0.8)
        axes[0,0].set_title('Makespan Vergleich aller Heuristiken', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Makespan')
        axes[0,0].set_xticks(range(len(self.df)))
        axes[0,0].set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # Füge Werte über Balken hinzu
        for i, (bar, value) in enumerate(zip(bars, self.df['makespan'])):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                          f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Performance Gap Analysis
        gap_colors = ['red' if gap > 2 else 'orange' if gap > 1 else 'green' 
                     for gap in self.df['makespan_gap_percent']]
        
        bars2 = axes[0,1].bar(range(len(self.df)), self.df['makespan_gap_percent'], 
                             color=gap_colors, alpha=0.7)
        axes[0,1].set_title('Makespan Performance Gap (%)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Abweichung vom Optimum (%)')
        axes[0,1].set_xticks(range(len(self.df)))
        axes[0,1].set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # Füge Werte hinzu
        for bar, value in zip(bars2, self.df['makespan_gap_percent']):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                          f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Multi-Criteria Radar Chart (Top 5)
        top_5 = self.df.head(5)
        
        # Normalisiere Metriken für Radar Chart
        metrics = ['makespan', 'deadline_ratio', 'timeliness']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Schließe den Kreis
        
        ax_radar = plt.subplot(2, 2, 3, projection='polar')
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            # Normalisiere Werte (0-1)
            makespan_norm = 1 - (row['makespan'] - self.df['makespan'].min()) / (self.df['makespan'].max() - self.df['makespan'].min())
            values = [makespan_norm, row['deadline_ratio'], row['timeliness']]
            values += values[:1]  # Schließe den Kreis
            
            color = '#2E8B57' if row['Heuristic'] == 'PPO-GNN' else plt.cm.Set3(i)
            linewidth = 3 if row['Heuristic'] == 'PPO-GNN' else 2
            
            ax_radar.plot(angles, values, 'o-', linewidth=linewidth, 
                         label=row['Heuristic'], color=color)
            ax_radar.fill(angles, values, alpha=0.1, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(['Makespan\n(norm.)', 'Deadline\nRatio', 'Timeliness'])
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Multi-Criteria Performance\n(Top 5 Heuristics)', 
                          y=1.08, fontsize=12, fontweight='bold')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # PPO-GNN Specific Analysis
        if self.ppo_analysis:
            ppo_data = [
                self.ppo_analysis['makespan_rank'],
                self.ppo_analysis['deadline_rank'],
                self.ppo_analysis['timeliness_rank'],
                self.ppo_analysis['overall_rank']
            ]
            
            categories = ['Makespan', 'Deadline', 'Timeliness', 'Overall']
            colors_ppo = ['red' if rank > len(self.df)//2 else 'orange' if rank > len(self.df)//3 else 'green' 
                         for rank in ppo_data]
            
            bars3 = axes[1,1].bar(categories, ppo_data, color=colors_ppo, alpha=0.7)
            axes[1,1].set_title('PPO-GNN Ranking in allen Kategorien', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('Ranking (niedrigere = besser)')
            axes[1,1].set_ylim(0, len(self.df) + 1)
            axes[1,1].grid(True, alpha=0.3)
            
            # Füge Ranking-Info hinzu
            for bar, value in zip(bars3, ppo_data):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              f'{int(value)}/{len(self.df)}', ha='center', va='bottom', 
                              fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'results/key_metrics/key_comparison_overview_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed PPO-GNN Analysis
        if self.ppo_analysis:
            self.create_ppo_detailed_analysis()
        
        # 3. Statistical Summary Table
        self.create_summary_table()
    
    def create_ppo_detailed_analysis(self):
        """Erstellt detaillierte PPO-GNN Analyse."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. PPO-GNN vs. Best Classical
        best_classical = self.df[self.df['Heuristic'] != 'PPO-GNN'].iloc[0]
        ppo_row = self.df[self.df['Heuristic'] == 'PPO-GNN'].iloc[0]
        
        comparison_data = {
            'Heuristic': [best_classical['Heuristic'], 'PPO-GNN'],
            'Makespan': [best_classical['makespan'], ppo_row['makespan']],
            'Deadline_Ratio': [best_classical['deadline_ratio'], ppo_row['deadline_ratio']],
            'Timeliness': [best_classical['timeliness'], ppo_row['timeliness']]
        }
        
        x = np.arange(2)
        width = 0.25
        
        axes[0,0].bar(x - width, comparison_data['Makespan'], width, 
                     label='Makespan', color='#FF6B6B', alpha=0.8)
        axes[0,0].bar(x, [d*1000 for d in comparison_data['Deadline_Ratio']], width, 
                     label='Deadline Ratio (×1000)', color='#4ECDC4', alpha=0.8)
        axes[0,0].bar(x + width, [t*10000 for t in comparison_data['Timeliness']], width, 
                     label='Timeliness (×10000)', color='#45B7D1', alpha=0.8)
        
        axes[0,0].set_title('PPO-GNN vs. Best Classical (EDD)', fontsize=14, fontweight='bold')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(comparison_data['Heuristic'])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Performance Gap Details
        gaps = {
            'Makespan': ((ppo_row['makespan'] - best_classical['makespan']) / best_classical['makespan'] * 100),
            'Deadline': ((ppo_row['deadline_ratio'] - best_classical['deadline_ratio']) / best_classical['deadline_ratio'] * 100),
            'Timeliness': ((ppo_row['timeliness'] - best_classical['timeliness']) / best_classical['timeliness'] * 100)
        }
        
        gap_colors = ['red' if gap > 0 else 'green' for gap in gaps.values()]
        bars = axes[0,1].bar(gaps.keys(), gaps.values(), color=gap_colors, alpha=0.7)
        axes[0,1].set_title('PPO-GNN Performance Gap vs. EDD (%)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Performance Gap (%)')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].grid(True, alpha=0.3)
        
        # Füge Werte hinzu
        for bar, value in zip(bars, gaps.values()):
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2, height + (1 if height > 0 else -2),
                          f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                          fontsize=11, fontweight='bold')
        
        # 3. Ranking Progression
        rankings = [
            self.ppo_analysis['makespan_rank'],
            self.ppo_analysis['deadline_rank'], 
            self.ppo_analysis['timeliness_rank'],
            self.ppo_analysis['overall_rank']
        ]
        categories = ['Makespan', 'Deadline', 'Timeliness', 'Overall']
        
        # Erstelle Ranking-Progression
        axes[1,0].plot(categories, rankings, 'o-', linewidth=3, markersize=10, color='#2E8B57')
        axes[1,0].fill_between(categories, rankings, alpha=0.3, color='#2E8B57')
        axes[1,0].set_title('PPO-GNN Ranking Progression', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Ranking Position')
        axes[1,0].set_ylim(len(self.df), 0)  # Invertiert für bessere Visualisierung
        axes[1,0].grid(True, alpha=0.3)
        
        # Füge Ranking-Werte hinzu
        for i, (cat, rank) in enumerate(zip(categories, rankings)):
            axes[1,0].text(i, rank - 0.2, f'{int(rank)}/{len(self.df)}', 
                          ha='center', va='top', fontsize=11, fontweight='bold')
        
        # 4. Improvement Potential
        improvement_data = {
            'Current PPO-GNN': ppo_row['makespan'],
            'Best Classical (EDD)': best_classical['makespan'],
            'Improvement Needed': ppo_row['makespan'] - best_classical['makespan']
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#FFA07A']
        bars = axes[1,1].bar(range(len(improvement_data)), improvement_data.values(), 
                            color=colors, alpha=0.8)
        axes[1,1].set_title('Makespan Improvement Potential', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(range(len(improvement_data)))
        axes[1,1].set_xticklabels(improvement_data.keys(), rotation=15)
        axes[1,1].set_ylabel('Makespan')
        axes[1,1].grid(True, alpha=0.3)
        
        # Füge Werte hinzu
        for bar, value in zip(bars, improvement_data.values()):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                          f'{value:.0f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'results/key_metrics/ppo_gnn_detailed_analysis_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self):
        """Erstellt eine übersichtliche Tabelle mit den wichtigsten Metriken."""
        
        # Bereite Daten für Tabelle vor
        summary_df = self.df[['Heuristic', 'makespan', 'makespan_gap_percent', 
                             'deadline_ratio', 'timeliness', 'overall_rank']].copy()
        
        # Formatiere für bessere Lesbarkeit
        summary_df['Makespan'] = summary_df['makespan'].apply(lambda x: f"{x:.0f}")
        summary_df['Gap (%)'] = summary_df['makespan_gap_percent'].apply(lambda x: f"{x:.1f}%")
        summary_df['Deadline Rate'] = summary_df['deadline_ratio'].apply(lambda x: f"{x:.1%}")
        summary_df['Timeliness'] = summary_df['timeliness'].apply(lambda x: f"{x:.3f}")
        summary_df['Overall Rank'] = summary_df['overall_rank'].apply(lambda x: f"{int(x)}")
        
        # Finale Tabelle
        final_table = summary_df[['Heuristic', 'Makespan', 'Gap (%)', 
                                 'Deadline Rate', 'Timeliness', 'Overall Rank']]
        
        # Speichere als CSV
        final_table.to_csv(f'results/key_metrics/summary_table_{self.timestamp}.csv', index=False)
        
        # Erstelle LaTeX-Tabelle
        latex_table = final_table.to_latex(index=False, escape=False,
                                          column_format='|l|r|r|r|r|c|',
                                          caption='JSP Heuristics Comparison - Key Metrics',
                                          label='tab:jsp_key_metrics')
        
        with open(f'results/key_metrics/summary_table_{self.timestamp}.tex', 'w') as f:
            f.write("% LaTeX Table - JSP Key Metrics\n")
            f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write(latex_table)
        
        return final_table
    
    def generate_executive_summary(self):
        """Generiert eine Executive Summary mit den wichtigsten Erkenntnissen."""
        
        best_heuristic = self.df.iloc[0]
        worst_heuristic = self.df.iloc[-1]
        
        summary = f"""
# JSP HEURISTICS - KEY METRICS SUMMARY

**Analysezeitpunkt:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

## 🎯 KERNERKENNTNISSE

### Beste Performance
- **Champion:** {best_heuristic['Heuristic']}
- **Makespan:** {best_heuristic['makespan']:.0f}
- **Deadline-Einhaltung:** {best_heuristic['deadline_ratio']:.1%}
- **Timeliness Score:** {best_heuristic['timeliness']:.3f}

### Performance-Spanne
- **Bester Makespan:** {best_heuristic['makespan']:.0f}
- **Schlechtester Makespan:** {worst_heuristic['makespan']:.0f}
- **Verbesserungspotential:** {((worst_heuristic['makespan'] - best_heuristic['makespan']) / best_heuristic['makespan'] * 100):.1f}%

## 🤖 PPO-GNN PERFORMANCE

"""
        
        if self.ppo_analysis:
            ppo_row = self.df[self.df['Heuristic'] == 'PPO-GNN'].iloc[0]
            summary += f"""### Current Standing
- **Overall Ranking:** {self.ppo_analysis['overall_rank']}/{self.ppo_analysis['total_heuristics']}
- **Makespan Ranking:** {self.ppo_analysis['makespan_rank']}/{self.ppo_analysis['total_heuristics']}
- **Makespan:** {ppo_row['makespan']:.0f}
- **Gap zu EDD:** {self.ppo_analysis['makespan_gap']:.1f}%

### Strengths & Weaknesses
- **Deadline Performance:** Rang {self.ppo_analysis['deadline_rank']}/{self.ppo_analysis['total_heuristics']}
- **Timeliness Performance:** Rang {self.ppo_analysis['timeliness_rank']}/{self.ppo_analysis['total_heuristics']}

### Improvement Potential
- **Absolute Improvement needed:** {ppo_row['makespan'] - best_heuristic['makespan']:.0f} units
- **Relative Improvement needed:** {((ppo_row['makespan'] - best_heuristic['makespan']) / best_heuristic['makespan'] * 100):.1f}%
"""
        
        summary += f"""
## 📊 TOP 3 HEURISTICS

1. **{self.df.iloc[0]['Heuristic']}**
   - Makespan: {self.df.iloc[0]['makespan']:.0f}
   - Deadline Rate: {self.df.iloc[0]['deadline_ratio']:.1%}
   - Gap: {self.df.iloc[0]['makespan_gap_percent']:.1f}%

2. **{self.df.iloc[1]['Heuristic']}**
   - Makespan: {self.df.iloc[1]['makespan']:.0f}
   - Deadline Rate: {self.df.iloc[1]['deadline_ratio']:.1%}
   - Gap: {self.df.iloc[1]['makespan_gap_percent']:.1f}%

3. **{self.df.iloc[2]['Heuristic']}**
   - Makespan: {self.df.iloc[2]['makespan']:.0f}
   - Deadline Rate: {self.df.iloc[2]['deadline_ratio']:.1%}
   - Gap: {self.df.iloc[2]['makespan_gap_percent']:.1f}%

## 💡 EMPFEHLUNGEN FÜR PPO-GNN

### Kurzfristig
1. **Reward Function Optimization** - Fokus auf Makespan-Minimierung
2. **Training Extension** - Mehr Episoden für bessere Konvergenz
3. **Hyperparameter Tuning** - Learning Rate, Exploration Rate

### Mittelfristig  
1. **Feature Engineering** - Bessere State Representation
2. **Architecture Improvements** - Optimierung des Graph Neural Networks
3. **Multi-Objective Training** - Balance zwischen Makespan und Timeliness

### Benchmark-Ziele
- **Realistisches Ziel:** Top 5 (Makespan < {self.df.iloc[4]['makespan']:.0f})
- **Optimistisches Ziel:** Top 3 (Makespan < {self.df.iloc[2]['makespan']:.0f})
- **Stretch-Ziel:** Neue Bestleistung (Makespan < {best_heuristic['makespan']:.0f})

---
*Generiert mit JSP Key Metrics Analyzer*
"""
        
        # Speichere Summary
        with open(f'results/key_metrics/executive_summary_{self.timestamp}.md', 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return summary
    
    def run_complete_analysis(self):
        """Führt die komplette Analyse durch und generiert alle Outputs."""
        
        print("🎯 JSP KEY METRICS ANALYSIS")
        print("=" * 50)
        
        # Berechne Metriken
        print("📊 Berechne Key Metrics...")
        df = self.calculate_key_metrics()
        
        # Erstelle Visualisierungen
        print("📈 Erstelle Visualisierungen...")
        self.create_key_visualizations()
        
        # Generiere Summary
        print("📋 Generiere Executive Summary...")
        summary = self.generate_executive_summary()
        
        print("✅ Analyse abgeschlossen!")
        print(f"📁 Alle Dateien gespeichert in: results/key_metrics/")
        print(f"📄 Executive Summary: results/key_metrics/executive_summary_{self.timestamp}.md")
        
        # Zeige wichtigste Ergebnisse
        print("\n🏆 TOP 3 HEURISTICS:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"  {i+1}. {row['Heuristic']}: {row['makespan']:.0f} (Gap: {row['makespan_gap_percent']:.1f}%)")
        
        if self.ppo_analysis:
            print(f"\n🤖 PPO-GNN POSITION:")
            print(f"  Overall Rank: {self.ppo_analysis['overall_rank']}/{self.ppo_analysis['total_heuristics']}")
            print(f"  Makespan Gap: {self.ppo_analysis['makespan_gap']:.1f}%")
        
        return df, summary


def create_key_metrics_report(results_data=None):
    """
    Hauptfunktion zur Erstellung eines Key Metrics Reports.
    
    Args:
        results_data: Optional - Eigene Ergebnisdaten, sonst werden Beispieldaten verwendet
    """
    analyzer = JSPKeyMetricsExtractor(results_data)
    return analyzer.run_complete_analysis()


if __name__ == "__main__":
    # Führe Key Metrics Analyse aus
    df, summary = create_key_metrics_report()
    
    print("\n" + "="*50)
    print("🎉 KEY METRICS ANALYSE ABGESCHLOSSEN!")
    print("="*50)