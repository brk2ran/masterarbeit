# Baseline (EDA-Pipeline)

Diese Baseline definiert den Referenzzustand der Datenpipeline und der daraus erzeugten EDA-Artefakte.
Ziel: Änderungen an Preprocessing/Features/EDA/Plots sollen nachvollziehbar und prüfbar sein (Reproduzierbarkeit).

## Pipeline-Schritte (Baseline-Run)
1) Preprocessing:
   - python -m src.data_preprocessing
   - Output: data/interim/mlperf_tiny_raw.parquet

2) EDA-Tabellen:
   - python -m src.eda
   - Output: reports/tables/*.csv

3) Plots:
   - python -m src.plots --all --logy --svg
   - Output: reports/figures/*.(png|svg)

4) Baseline-Freeze:
   - python -m src.baseline --freeze
   - Output: docs/baseline/*

## Scope-Entscheidung (für Vergleichbarkeit)
- IN_SCOPE: AD, IC, KWS, VWW
- OUT_OF_SCOPE: 1D_DS_CNN (nur in v1.3, Streaming Wakeword) wird ausgeschlossen
- UNKNOWN: Zeilen, bei denen task/model nicht eindeutig ableitbar sind

Begründung: Für die Trendanalyse über v0.5–v1.3 soll eine konsistente Aufgaben-/Modellbasis verwendet werden.
1D_DS_CNN ist ein zusätzliches Benchmark-Element (Streaming Wakeword) und nicht über alle Runden vergleichbar.

## Was gilt als "Baseline-stabil"
- Rohdaten (data/raw/*.csv) müssen unverändert sein (raw_hashes.csv)
- Parquet-Metadaten (Zeilen/Spalten/Spaltennamen) müssen plausibel stabil sein (parquet_meta.json)
- Tabellen/Plots müssen vorhanden sein (expected_tables.txt, expected_figures.txt)
- Kennzahlen (Scope-Anteile, Missingness, Quantile, UNKNOWN-Anteil) dürfen sich nur ändern,
  wenn es einen bewussten Code-/Daten-Change gab.
