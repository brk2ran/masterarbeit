# MASTERARBEIT -- MLPerf-Tiny EDA Pipeline

Dieses Repository enthält die reproduzierbare Analysepipeline zur
Masterarbeit von **Burak Turan**:

**Explorative Datenanalyse der MLPerf-Tiny-Ergebnisse: Trends und
Treiber der Latenz- und Energieeffizienz sowie deren Implikationen für
Edge-KI**

Autor: Burak Turan\
Studiengang: Wirtschaftsinformatik (M.Sc.)\
Hochschule: HTW Berlin\
Jahr: 2026

Das Projekt analysiert die **MLPerf-Tiny-Ergebnisse der Closed
Division** über die Benchmark-Runden **v0.5, v0.7, v1.0, v1.1, v1.2 und
v1.3**. Der Fokus liegt auf einer reproduzierbaren explorativen
Datenanalyse (EDA) entlang der Zeitachse, der Untersuchung hardware- und
softwarebezogener Merkmale, der Identifikation von Systemklassen sowie
der Analyse von Pareto-Trade-offs zwischen **Latenz** und
**Energieverbrauch**.

------------------------------------------------------------------------

# Ziel des Repositories

Das Repository dient dazu,

-   die in der Arbeit beschriebenen Datenaufbereitungs- und
    Analyseschritte nachvollziehbar abzubilden,
-   zentrale Tabellen- und Abbildungsartefakte reproduzierbar zu
    erzeugen,
-   die Zuordnung zwischen Text, Code und Ergebnisartefakten transparent
    zu machen,
-   Änderungen an Tabellen und Grafiken versioniert zu dokumentieren,
-   Gutachterinnen und Gutachtern eine nachvollziehbare Rekonstruktion
    der Analysepipeline zu ermöglichen.

------------------------------------------------------------------------

# Datenquelle

Die Analyse basiert auf **öffentlich verfügbaren Benchmark-Ergebnissen
von MLCommons MLPerf Tiny**.

Offizielle Quelle:\
https://mlcommons.org/benchmarks/inference-tiny/

Zusätzliches Dashboard:\
https://public.tableau.com/app/profile/data.visualization6666/viz/MLCommons-Tiny_16993773264820/Dashboard1

Verwendete Benchmark-Runden:

-   v0.5
-   v0.7
-   v1.0
-   v1.1
-   v1.2
-   v1.3

Rohdaten befinden sich in:

    data/raw/

Beispieldateien:

    raw_v0.5.csv
    raw_v0.7.csv
    raw_v1.0.csv
    raw_v1.1.csv
    raw_v1.2.csv
    raw_v1.3.csv

Gesamtgröße der Rohdaten: ca. **82 KB**.

Es werden keine personenbezogenen oder sensiblen Daten verarbeitet.

------------------------------------------------------------------------

# Zeitraum

Die Analyse wurde im Zeitraum **November 2025 -- März 2026**
durchgeführt.

------------------------------------------------------------------------

# Repository-Struktur

    MASTERARBEIT/

    README.md
    LICENSE
    requirements.txt
    requirements-lock.txt

    data/
        raw/
        interim/

    src/
        checks.py
        clustering.py
        clustering_kmodes_robustness.py
        data_preprocessing.py
        eda.py
        features.py
        plots.py

    reports/
        tables/
        figures/

    docs/
        baseline/
        baseline_tables/

------------------------------------------------------------------------

# Software-Umgebung

Entwicklungsumgebung

-   Windows
-   Visual Studio Code

Python-Version

-   Python 3.13.2

Wichtige Bibliotheken

-   pandas
-   numpy
-   scikit-learn
-   matplotlib

Abhängigkeiten:

    requirements.txt
    requirements-lock.txt

------------------------------------------------------------------------

# Installation

Virtuelle Umgebung erstellen

    python -m venv .venv

Aktivieren (Windows)

    .venv\Scripts\activate

Abhängigkeiten installieren

    pip install -r requirements.txt

------------------------------------------------------------------------

# Ausführung der Pipeline

## Abgabe-Quickstart

Dieses Repository stellt den finalen Abgabestand der Analysepipeline dar.
Ausführungsreihenfolge: `python -m src.data_preprocessing` → `python -m src.eda` → `python -m src.clustering` → `python -m src.clustering_kmodes_robustness` → `python -m src.plots` → `python -m src.checks`.
Die finalen, in der Arbeit referenzierten Artefakte befinden sich in `reports/tables/` und `reports/figures/`.
Der exakte Abgabestand ist über den finalen Git-Tag bzw. Release eindeutig identifizierbar.


1.  Datenaufbereitung

```{=html}
    python -m src.data_preprocessing
```

2.  Explorative Analyse

```{=html}
    python -m src.eda
```

3.  Abbildungen erzeugen

```{=html}
    python -m src.plots
```

4.  Clusteranalyse

```{=html}
    python -m src.clustering
```

5.  Robustheitsanalyse

```{=html}
    python -m src.clustering_kmodes_robustness
```

6.  Validierungschecks

```{=html}
    python -m src.checks
```

Generierte Artefakte befinden sich in:

    reports/tables
    reports/figures

------------------------------------------------------------------------

# Qualitätssicherung

Die Pipeline enthält mehrere Schritte zur Sicherstellung der
Datenintegrität:

-   Harmonisierung der Benchmarkdaten über mehrere Runden
-   Prüfung der round × task-Abdeckung
-   Plausibilitätsprüfung von Latenz- und Energie-Metriken
-   Umgang mit fehlenden Werten
-   versionierte Ablage von Ergebnisartefakten

Baseline-Dateien in

    docs/baseline/

ermöglichen den Vergleich aktueller Ergebnisse mit Referenzständen.

------------------------------------------------------------------------

# Datenschutz

Das Repository verwendet ausschließlich **öffentlich verfügbare
Benchmarkdaten**.

-   keine personenbezogenen Daten
-   keine sensiblen Daten
-   keine Einwilligungserklärungen erforderlich

------------------------------------------------------------------------

# Zitation

Bitte zitieren Sie bei Verwendung dieses Repositories:

Burak Turan (2026)\
Explorative Datenanalyse der MLPerf-Tiny-Ergebnisse: Trends und Treiber
der Latenz- und Energieeffizienz sowie deren Implikationen für Edge-KI\
HTW Berlin

------------------------------------------------------------------------

# Lizenz

Dieses Projekt steht unter der **MIT License**.

Siehe Datei:

    LICENSE

------------------------------------------------------------------------

# Autor

Burak Turan\
Masterarbeit -- Wirtschaftsinformatik\
HTW Berlin
