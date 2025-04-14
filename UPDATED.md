# Erklärung der Belohnungsfunktion

## Überblick über die `_calculate_reward` Funktion

Die Belohnungsfunktion bewertet, wie gut jede Planungsentscheidung ist, indem sie mehrere Faktoren berücksichtigt:

- **Makespan-Belohnung**: Bestraft längere Bearbeitungszeiten
- **Setup-Belohnung**: Bestraft Maschinenrüstzeiten
- **Leerlauf-Strafe**: Bestraft Zeiten, in denen Maschinen ungenutzt bleiben
- **Termin-Belohnung**: Belohnt das Einhalten von Auftragsterminen, bestraft Terminüberschreitungen
- **Prioritäts-Belohnung**: Höhere Belohnungen für das Abschließen von Aufträgen mit hoher Priorität
- **Kritische-Aufträge-Belohnung**: Zusätzlicher Bonus für das Abschließen von Aufträgen mit sehr hoher Priorität
- **Globale-Fortschritts-Belohnung**: Belohnt den allgemeinen Fertigstellungsfortschritt
- **Ziel-Belohnung**: Nutzt Modellvorhersagen zur Bewertung langfristiger Auswirkungen
- **Platzierungs-Belohnung**: Belohnt gute Entscheidungen bei der Operationsplatzierung
- **Vorausschau-Belohnung**: Simuliert zukünftige Schritte, um aktuelle Entscheidungen zu bewerten

## Pünktlichkeits-Komponente

**Was es ist**: Bewertet, wie gut der Zeitplan die zeitlichen Erwartungen erfüllt.

**Wie es funktioniert**:
- Vergleicht die aktuelle Zeit mit der durchschnittlichen Deadline aller Aufträge
- Belohnt das Vorankommen im Zeitplan, bestraft Verzögerungen
- Bietet zusätzliche Belohnungen für das frühere Abschließen von Operationen als erwartet

**Beispiel**:
- Wenn die durchschnittliche Deadline 100 Zeiteinheiten beträgt und die aktuelle Zeit 50 ist, ist der Pünktlichkeitsfaktor 0,5 (gut)
- Wenn die aktuelle Zeit die durchschnittliche Deadline überschreitet, wird eine Strafe angewendet
- Für einzelne Operationen: Wenn eine Operation bei 30% ihrer zugewiesenen Zeit fertiggestellt wird, erhält sie einen Bonus

## Maschinenstillstand-Komponente

**Was es ist**: Bestraft speziell Maschinen, die stillstehen, während andere Maschinen arbeiten.

**Wie es funktioniert**:
- Berechnet die gesamte Leerlaufzeit aller Maschinen (außer der aktuell genutzten)
- Berechnet die durchschnittliche Leerlaufzeit pro Maschine und wendet eine Strafe proportional zu diesem Wert an
- Fügt zusätzliche Strafen für Maschinen hinzu, die über längere Zeiträume (>20% der aktuellen Zeit) stillstehen

**Beispiel**:
- Wenn Maschine 1 arbeitet, aber die Maschinen 2-5 jeweils 10 Zeiteinheiten lang stillstehen, beträgt die durchschnittliche Leerlaufzeit 10
- Eine Strafe proportional zu diesem Durchschnitt wird angewendet
- Wenn eine Maschine länger als 20% der Gesamtplanungszeit stillgestanden hat, wird eine zusätzliche Strafe angewendet

## Credit-Assignment-Problem-Komponente

**Was es ist**: Verteilt Strafen auf vergangene Aktionen, die zu aktuellen Problemen beigetragen haben, nicht nur auf die jüngste Aktion.

**Wie es funktioniert**:
- Führt einen Verlauf vergangener Aktionen
- Wenn Probleme auftreten (verpasste Deadlines, übermäßige Leerlaufzeit), identifiziert es, welche vergangenen Aktionen dazu beigetragen haben
- Wendet Strafen auf diese vergangenen Aktionen mit exponentiellem Abfall an (ältere Aktionen erhalten kleinere Strafen)

**Beispiel**:
- Wenn Auftrag A seine Deadline in Schritt 20 verpasst:
  - Die Aktion in Schritt 20 erhält die volle Strafe
  - Die Aktion in Schritt 19 erhält möglicherweise 70% der Strafe
  - Die Aktion in Schritt 18 erhält möglicherweise 49% (0,7²) der Strafe
  - Und so weiter, bis zu 10 Schritte zurück

Dieser Ansatz hilft dem Lernalgorithmus zu verstehen, welche früheren Entscheidungen später zu Problemen geführt haben, und verbessert so die langfristige Planung.