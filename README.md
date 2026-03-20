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

# Datensatz / Softwarepaket

Dieses Repository dokumentiert ein im Rahmen der Masterarbeit erstelltes
**Softwarepaket zur reproduzierbaren Sekundäranalyse öffentlich
verfügbarer Benchmarkdaten**.

Es wurden Python-Skripte zur Datenaufbereitung, explorativen
Datenanalyse, Visualisierung, Clusteranalyse, Robustheitsanalyse und
Qualitätssicherung entwickelt. Grundlage der Auswertung sind öffentlich
verfügbare Benchmark-Ergebnisse von **MLPerf Tiny** in der
**Closed Division** über mehrere Benchmark-Runden.

## Arten von Daten

Verarbeitet werden ausschließlich **strukturierte tabellarische
Benchmarkdaten** in CSV-Form. Im Verlauf der Pipeline entstehen daraus
bereinigte und harmonisierte Zwischendatensätze, Ergebnistabellen und
Abbildungen.

## Sprache

-   Dokumentation und README: **Deutsch**
-   Quellcode, Dateinamen und technische Bezeichner: überwiegend **Englisch**

## Genutzte wissenschaftliche Methode

Die Arbeit basiert auf einer **Sekundärdatenanalyse** bereits
veröffentlichter Benchmarkdaten. Methodisch handelt es sich um eine
**explorative Datenanalyse (EDA)** mit deskriptiver Auswertung entlang
der Zeitachse sowie ergänzenden Analysen zu hardware- und
softwarebezogenen Einflussfaktoren, Clustering von Systemklassen und
Pareto-Trade-off-Analysen zwischen Latenz und Energieverbrauch.

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

## Datenursprung

Die verwendeten Daten wurden **nicht selbst erhoben**, sondern stammen
aus den öffentlich bereitgestellten Benchmark-Ergebnissen von
**MLCommons / MLPerf Tiny**.

### Autorenschaft / Verantwortlichkeit

Verantwortlich für die externen Quelldaten ist die jeweilige
Originalquelle bzw. die veröffentlichende Organisation **MLCommons**.

### Identifier

Ein separater **DOI** für die konkret verwendeten Rohdaten ist in der
hier genutzten Quelle nicht ausgewiesen. Die Referenzierung erfolgt daher
über die offizielle Benchmark-Webseite und das öffentliche Dashboard.

### Lizenz der Quelldaten

Für die auf der MLCommons-Webseite bereitgestellten MLPerf-Tiny-Ergebnisse ist auf der genutzten Benchmark-Seite keine separat ausgewiesene Datensatzlizenz angegeben. Die Seite verweist insbesondere auf Regeln, Referenzimplementierungen sowie Results Usage Guidelines von MLCommons. Für die Nutzung der externen Inhalte gelten daher die Bedingungen und Richtlinien der Originalquelle. Zusätzlich ist zu beachten, dass die MLPerf Results Messaging Guidelines ausdrücklich keine Lizenz zur Nutzung des MLPerf-Namens oder -Logos begründen.

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

## Datenformate und -größe

-   Rohdaten: **CSV**
-   aufbereitete Zwischendatensätze: **CSV**
-   Ergebnistabellen: **CSV**
-   Abbildungen / Visualisierungen: **PNG**

Es werden keine personenbezogenen oder sensiblen Daten verarbeitet.

------------------------------------------------------------------------

# Zeitraum

Die Analyse wurde im Zeitraum **November 2025 -- März 2026**
durchgeführt.

In diesem Zeitraum erfolgten insbesondere:

-   Sichtung und Zusammenstellung der öffentlich verfügbaren Benchmarkdaten
-   Entwicklung und Überarbeitung der Python-Pipeline
-   Datenbereinigung und Harmonisierung
-   explorative Analyse und Visualisierung
-   Cluster- und Robustheitsanalysen
-   Erstellung der finalen Tabellen- und Abbildungsartefakte

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

## Ablageort und Dateibenennung

Die Dateien sind nach ihrem Zweck und Inhalt benannt. Beispiele:

-   Rohdaten nach Benchmark-Runde, z. B. `raw_v1.2.csv`
-   Abbildungen nach Analysegegenstand, z. B. `trend_latency_us.png`
-   Ergebnisdateien nach inhaltlicher Funktion, z. B. Tabellen in `reports/tables/`

Die maßgebliche Projektstruktur liegt im versionierten Repository. Die
externen Originaldaten verbleiben bei den jeweiligen Quellen.

Für das Repository selbst ist derzeit kein konkretes Löschdatum
festgelegt.

------------------------------------------------------------------------

# Software-Umgebung

Entwicklungsumgebung

-   Windows
-   Visual Studio Code  
    https://code.visualstudio.com/

Python-Version

-   Python 3.13.2  
    https://www.python.org/

Wichtige Bibliotheken

-   pandas  2.3.3
    https://pandas.pydata.org/
-   numpy  2.4.0
    https://numpy.org/
-   scikit-learn  1.8.0
    https://scikit-learn.org/
-   matplotlib  3.10.8
    https://matplotlib.org/

Abhängigkeiten:

    requirements.txt
    requirements-lock.txt

Die konkret verwendeten Paketversionen sind in `requirements.txt` und
`requirements-lock.txt` dokumentiert.

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

```text
    python -m src.data_preprocessing
```

2.  Explorative Analyse

```text
    python -m src.eda
```

3.  Abbildungen erzeugen

```text
    python -m src.plots
```

4.  Clusteranalyse

```text
    python -m src.clustering
```

5.  Robustheitsanalyse

```text
    python -m src.clustering_kmodes_robustness
```

6.  Validierungschecks

```text
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

## Urheber- und Schutzrechte

Die Urheber- und Nutzungsrechte an den externen Benchmarkdaten und den zugehörigen Markenbezeichnungen verbleiben bei den jeweiligen Originalquellen bzw. bei MLCommons. Für die Nachnutzung gelten die Bedingungen der Originalanbieter.

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
