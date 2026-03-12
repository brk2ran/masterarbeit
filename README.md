# MASTERARBEIT – MLPerf-Tiny EDA Pipeline

Dieses Repository enthält die reproduzierbare Analysepipeline zur Masterarbeit von **Burak Turan**. Untersucht werden die **MLPerf-Tiny-Ergebnisse der Closed Division** über die Benchmark-Runden **v0.5, v0.7, v1.0, v1.1, v1.2 und v1.3**. Der Fokus liegt auf einer explorativen Datenanalyse (EDA) entlang der Zeitachse, der Untersuchung hardware- und softwarebezogener Merkmale, der Identifikation von Systemklassen sowie der Analyse von Pareto-Trade-offs zwischen **Latenz** und **Energieverbrauch**.

## Ziel des Repositories

Das Repository dient dazu,

- die in der Arbeit beschriebenen Datenaufbereitungs- und Analyseschritte nachvollziehbar abzubilden,
- zentrale Tabellen- und Abbildungsartefakte reproduzierbar zu erzeugen,
- die Zuordnung zwischen Text, Code und Ergebnisartefakten transparent zu machen,
- Änderungen an Tabellen und Grafiken versioniert zu dokumentieren.

## Untersuchte Datenbasis

Verwendet werden die offiziellen MLPerf-Tiny-Ergebnisexporte der **Closed Division** aus den Benchmark-Runden:

- v0.5
- v0.7
- v1.0
- v1.1
- v1.2
- v1.3

Die Rohdaten liegen als CSV-Dateien in `data/raw/`.  
Die harmonisierte, zusammengeführte Arbeitsbasis wird als Parquet-Datei in `data/interim/` abgelegt.

## Repository-Struktur

```text
MASTERARBEIT/
├── data/
│   ├── raw/
│   └── interim/
├── docs/
│   ├── baseline/
│   │   ├── figures_manifest.json
│   │   ├── tables_manifest.json
│   │   └── meta.json
│   └── baseline_tables/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── checks.py
│   ├── clustering.py
│   ├── clustering_kmodes_robustness.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── features.py
│   └── plots.py
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

## Methodische Einordnung

Die Pipeline bildet den in der Arbeit beschriebenen Analyseprozess ab:

1. **Datenaufbereitung und Harmonisierung**
   - Einlesen der CSV-Exporte aus mehreren Benchmark-Runden
   - Normalisierung und Mapping heterogener Spalten
   - Umformung in ein einheitliches Wide-Format
   - Persistenz als Parquet-Datei

2. **Feature Engineering**
   - Kanonisierung von Aufgaben und Metadaten
   - Ableitung zentraler Merkmale wie:
     - `task_canon`
     - `scope_status`
     - `accelerator_present`
     - `processor_family`
     - `cpu_freq_bucket`
     - `software_stack`

3. **EDA und FF1-Analysen**
   - Coverage-Analyse
   - longitudinale Trendtabellen für Latenz und Energie
   - hardware- und softwarebezogene Effekt-/Span-Analysen

4. **FF2-Analysen**
   - Clusterbildung zur Identifikation typischer Systemklassen
   - Profilierung von Clustern
   - Pareto-Analyse für Latenz und Energie
   - Robustheitsprüfung mit K-Modes

5. **Validierung und Reproduzierbarkeit**
   - Coverage-Checks
   - Baseline-/Diff-Prüfungen
   - versionierte Ablage finaler Artefakte

## Voraussetzungen

Getestet mit Python 3.x in einer virtuellen Umgebung.

Benötigte Pakete sind in `requirements.txt` definiert.

## Installation

### 1. Virtuelle Umgebung anlegen
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Abhängigkeiten installieren
```powershell
pip install -r requirements.txt
```

## Ausführungsreihenfolge

Die Pipeline kann schrittweise ausgeführt werden.

### 1. Datenaufbereitung
```powershell
python -m src.data_preprocessing
```

### 2. EDA-Tabellen / FF1-Auswertungen
```powershell
python -m src.eda
```

### 3. Abbildungen erzeugen
```powershell
python -m src.plots
```

### 4. FF2-Cluster- und Pareto-Analyse
```powershell
python -m src.clustering
```

### 5. Robustheitsprüfung K-Means vs. K-Modes
```powershell
python -m src.clustering_kmodes_robustness
```

### 6. Validierung / Checks
```powershell
python -m src.checks
```

## Wichtige Ausgabeordner

### `reports/tables/`
Enthält die zentralen Tabellenartefakte, die in der Arbeit verwendet oder im Anhang referenziert werden, z. B.:

- Trendtabellen zu Latenz und Energie
- Coverage-Tabellen
- FF1b-/FF1c-Auswertungen
- FF2-Clusterprofile
- Pareto-Zusammenfassungen
- Robustheitsvergleiche K-Means/K-Modes

### `reports/figures/`
Enthält die finalen Abbildungen für:

- longitudinale Trends
- Coverage
- Clusterprofile
- Pareto-Darstellungen

## Reproduzierbarkeit und Baselines

Zur Qualitätssicherung und Nachvollziehbarkeit werden Baseline-Informationen in `docs/baseline/` gepflegt. Dazu gehören insbesondere:

- `figures_manifest.json`
- `tables_manifest.json`
- `meta.json`

Diese Dateien dienen dem Vergleich aktueller Ausgaben mit einem definierten Referenzstand.

## Bezug zur Arbeit

Dieses Repository bildet die in der Masterarbeit beschriebenen Analyseschritte technisch ab. Die wichtigsten textlichen Ankerpunkte sind:

- **Kapitel 4.2–4.6**: Datenbasis, Aufbereitung, Feature Engineering, Analyseverfahren, Reproduzierbarkeit
- **Kapitel 5**: FF1-Auswertungen
- **Kapitel 6**: FF2, Cluster, Pareto, Robustheit
- **Anhang A/B**: ergänzende Tabellen und methodische Artefakte

Die in der Arbeit referenzierten Tabellen und Abbildungen sind – soweit ausgelagert – in `reports/tables/` und `reports/figures/` versioniert abgelegt.

## Hinweise zum Scope

- Die Analysen beziehen sich auf die **Closed Division**.
- Die Arbeit folgt einem **explorativen** Ansatz; die Ergebnisse sind daher als datengetriebene Musteranalyse und nicht als kausale Modellierung zu verstehen.
- Nicht alle Exportfelder sind über alle Runden und Aufgaben hinweg vollständig verfügbar. Coverage-Gates und Scope-Regeln sind deshalb Teil der methodischen Operationalisierung.

## Autor

**Burak Turan**  
Masterarbeit im Studiengang Wirtschaftsinformatik  
HTW Berlin

