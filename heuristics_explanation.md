# Job Shop Problem (JSP) Heuristiken

In diesem Dokument werden die verschiedenen Heuristiken erklärt, die zur Lösung des Job Shop Problems implementiert wurden. Jede Heuristik verwendet unterschiedliche Strategien, um zu entscheiden, welcher Job als nächstes bearbeitet werden soll.

## Überblick

Das Job Shop Problem (JSP) ist ein klassisches Optimierungsproblem in der Produktionsplanung. Es geht darum, eine Reihe von Jobs auf verschiedenen Maschinen so zu planen, dass die Gesamtfertigungszeit (Makespan) minimiert wird. Jeder Job besteht aus einer Reihe von Operationen, die in einer bestimmten Reihenfolge auf bestimmten Maschinen ausgeführt werden müssen.

## Leistungsmetriken

Bei der Bewertung der Heuristiken werden zwei Hauptmetriken verwendet:

1. **Makespan**: Die Gesamtzeit bis zur Fertigstellung aller Jobs (niedriger ist besser)
2. **Maschinenauslastung**: Der Prozentsatz der Zeit, in der die Maschinen tatsächlich arbeiten (höher ist besser)

## Implementierte Heuristiken

### 1. FIFO (First In First Out)

**Prinzip**: Wählt den Job mit dem niedrigsten Index aus, der bearbeitet werden kann.

**Implementierung**: Die Heuristik durchläuft die Liste der gültigen Aktionen und wählt den ersten verfügbaren Job aus.

**Anwendungsfall**: Einfache Implementierung, die eine faire Behandlung aller Jobs gewährleistet und Verhungern verhindert.

```python
def fifo_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            return job_idx
    return 0  # Fallback
```

### 2. LIFO (Last In First Out)

**Prinzip**: Wählt den Job mit dem höchsten Index aus, der bearbeitet werden kann.

**Implementierung**: Die Heuristik durchläuft die Liste der gültigen Aktionen rückwärts und wählt den ersten verfügbaren Job aus.

**Anwendungsfall**: Kann in bestimmten Szenarien nützlich sein, wenn neuere Jobs Priorität haben sollen.

```python
def lifo_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    for job_idx in range(len(valid_actions_mask) - 1, -1, -1):
        if valid_actions_mask[job_idx] == 1:
            return job_idx
    return 0  # Fallback
```

### 3. SPT (Shortest Processing Time)

**Prinzip**: Wählt den Job mit der kürzesten nächsten Operation aus.

**Implementierung**: Die Heuristik berechnet die Verarbeitungszeit der nächsten Operation für jeden verfügbaren Job und wählt den Job mit der kürzesten Zeit aus.

**Anwendungsfall**: Reduziert die durchschnittliche Durchlaufzeit und kann die Anzahl der gleichzeitig im System befindlichen Jobs minimieren.

```python
def spt_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    
    min_time = float('inf')
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            if op_idx < len(jsp_data["jobs"][job_idx]["operations"]):
                proc_time = jsp_data["jobs"][job_idx]["operations"][op_idx]["processingTime"]
                if proc_time < min_time:
                    min_time = proc_time
                    selected_job = job_idx
    
    return selected_job
```

### 4. LPT (Longest Processing Time)

**Prinzip**: Wählt den Job mit der längsten nächsten Operation aus.

**Implementierung**: Die Heuristik berechnet die Verarbeitungszeit der nächsten Operation für jeden verfügbaren Job und wählt den Job mit der längsten Zeit aus.

**Anwendungsfall**: Kann bei der Lastverteilung helfen und ist oft nützlich, wenn längere Jobs zuerst bearbeitet werden sollen, um Verzögerungen zu vermeiden.

```python
def lpt_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    
    max_time = -1
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            if op_idx < len(jsp_data["jobs"][job_idx]["operations"]):
                proc_time = jsp_data["jobs"][job_idx]["operations"][op_idx]["processingTime"]
                if proc_time > max_time:
                    max_time = proc_time
                    selected_job = job_idx
    
    return selected_job
```

### 5. MWKR (Most Work Remaining)

**Prinzip**: Wählt den Job mit der meisten verbleibenden Gesamtarbeitszeit aus.

**Implementierung**: Die Heuristik summiert die Verarbeitungszeiten aller verbleibenden Operationen für jeden verfügbaren Job und wählt den Job mit der höchsten Gesamtzeit aus.

**Anwendungsfall**: Priorisiert Jobs mit viel verbleibender Arbeit, was dazu beitragen kann, den Makespan zu reduzieren, indem die anspruchsvollsten Jobs zuerst bearbeitet werden.

```python
def mwkr_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    
    max_remaining_time = -1
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            remaining_time = 0
            
            # Sum up the processing times of all remaining operations
            for i in range(op_idx, len(jsp_data["jobs"][job_idx]["operations"])):
                remaining_time += jsp_data["jobs"][job_idx]["operations"][i]["processingTime"]
            
            if remaining_time > max_remaining_time:
                max_remaining_time = remaining_time
                selected_job = job_idx
    
    return selected_job
```

### 6. CR (Critical Ratio)

**Prinzip**: Balanciert die Verarbeitungszeit mit den Fälligkeitsterminen.
- CR = (Fälligkeitstermin - Aktuelle Zeit) / Verbleibende Verarbeitungszeit
- Ein niedrigerer CR-Wert bedeutet einen kritischeren Job.

**Implementierung**: Die Heuristik berechnet das Verhältnis zwischen der verbleibenden Zeit bis zum Fälligkeitstermin und der verbleibenden Verarbeitungszeit für jeden verfügbaren Job und wählt den Job mit dem niedrigsten Verhältnis aus.

**Anwendungsfall**: Berücksichtigt sowohl die Dringlichkeit (Fälligkeitstermin) als auch den Arbeitsaufwand, was zu einer ausgewogeneren Planung führt.

```python
def cr_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    current_time = max(state.get('machine_times', [0]))  # Current time is max of machine times
    
    # Assume due dates as 2x the sum of all processing times for each job if not provided
    due_dates = []
    for job in jsp_data["jobs"]:
        total_time = sum(op["processingTime"] for op in job["operations"])
        due_dates.append(total_time * 2)  # Simple heuristic for due date
    
    min_cr = float('inf')
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            remaining_time = 0
            
            # Calculate remaining processing time
            for i in range(op_idx, len(jsp_data["jobs"][job_idx]["operations"])):
                remaining_time += jsp_data["jobs"][job_idx]["operations"][i]["processingTime"]
            
            # Calculate critical ratio
            time_left = due_dates[job_idx] - current_time
            cr = time_left / remaining_time if remaining_time > 0 else float('inf')
            
            # Lower CR is more critical
            if cr < min_cr:
                min_cr = cr
                selected_job = job_idx
    
    return selected_job
```

### 7. RANDOM

**Prinzip**: Wählt zufällig einen der verfügbaren Jobs aus.

**Implementierung**: Die Heuristik erstellt eine Liste aller verfügbaren Jobs und wählt einen zufällig aus.

**Anwendungsfall**: Dient als Baseline für den Vergleich mit anderen Heuristiken und kann in bestimmten stochastischen Optimierungsszenarien nützlich sein.

```python
def random_heuristic(state, jsp_data):
    valid_actions_mask = state['valid_actions_mask']
    valid_jobs = [job_idx for job_idx, is_valid in enumerate(valid_actions_mask) if is_valid == 1]
    
    if valid_jobs:
        return random.choice(valid_jobs)
    return 0  # Fallback
```

### 8. PPO (Proximal Policy Optimization)

**Prinzip**: Ein Reinforcement Learning-Ansatz, der eine Policy-Netzwerk verwendet, um die optimale Aktion basierend auf dem aktuellen Zustand zu wählen.

**Implementierung**: Verwendet ein trainiertes neuronales Netzwerk, um die Wahrscheinlichkeiten für jede mögliche Aktion zu berechnen und wählt die Aktion mit der höchsten Wahrscheinlichkeit aus.

**Anwendungsfall**: Kann komplexe Muster und Abhängigkeiten lernen, die einfache Heuristiken möglicherweise nicht erfassen können, und sich an verschiedene JSP-Instanzen anpassen.

## Vergleich der Heuristiken

Die Ergebnisse des Vergleichs zeigen, dass jede Heuristik ihre Stärken und Schwächen hat:

1. **CR (Critical Ratio)** scheint in Bezug auf den Makespan am besten abzuschneiden, was darauf hindeutet, dass die Berücksichtigung sowohl der Fälligkeitstermine als auch der verbleibenden Arbeit zu einer effizienteren Planung führt.

2. **LIFO (Last In First Out)** zeigt die höchste Maschinenauslastung, was überraschend sein kann, aber in bestimmten Szenarien sinnvoll ist.

3. **RANDOM** hat erwartungsgemäß den höchsten Makespan, was die Bedeutung intelligenter Planungsstrategien unterstreicht.

4. **PPO (Reinforcement Learning)** zeigt eine solide Leistung, die mit den besseren Heuristiken vergleichbar ist, was das Potenzial von Lernansätzen für dieses Problem demonstriert.

## Fazit

Die Wahl der richtigen Heuristik hängt von den spezifischen Anforderungen und Eigenschaften des Job Shop Problems ab. In der Praxis kann eine Kombination verschiedener Heuristiken oder ein adaptiver Ansatz, der die Heuristik basierend auf dem aktuellen Systemzustand auswählt, zu den besten Ergebnissen führen.

Die Implementierung zusätzlicher Heuristiken wie LPT, MWKR und CR bietet mehr Optionen für die Optimierung und ermöglicht einen umfassenderen Vergleich verschiedener Planungsstrategien.
