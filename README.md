# Masterarbeit – MLPerf-Tiny EDA Pipeline

Reproduzierbare Python-Pipeline (VS Code, Git, venv) zur explorativen Datenanalyse (EDA) der MLPerf-Tiny Ergebnisse über mehrere Benchmark-Runden (v0.5–v1.3). Fokus: konsolidierter Datensatz, Validierung, Trendtabellen, Visualisierungen sowie anschließende Analysen (Pareto/Clustering).

## Ziele (fachlich)
- Konsolidierung heterogener CSV-Exports in ein einheitliches Schema
- Systematische Validierung (Coverage Round×Task, Units, Plausibilität, Missingness, Duplikate)
- EDA-Basisartefakte:
  - Tabellen: Coverage + Trendtabellen (Median/Quantile)
  - Plots: Trends, Datenqualitätsindikatoren
- Erweiterbar um: Einflussfaktoren (HW/SW), Clustering, Pareto-Analyse

## Repo-Struktur
```text
data/
  raw/                  # Rohdaten (CSV-Exports: raw_v0.5.csv ... raw_v1.3.csv)
  interim/               # Zwischenartefakte (Parquet)
docs/                    # Dokumentation/Validierungsreports (CSV/MD)
notebooks/               # Explorative Notebooks
reports/
  tables/                # EDA-Tabellen (CSV) – versioniert
  figures/               # Plots (PNG/SVG) – versioniert
src/
  data_preprocessing.py  # Ingestion + Normalisierung + Validierungsreport + Parquet-Export
  features.py            # Canonical Features + Analyse-Subsets + Coverage Utilities
  eda.py                 # Generiert Tabellen nach reports/tables/
  plots.py               # Generiert Plots nach reports/figures/
  clustering.py          # (optional) Clustering-Analysen
