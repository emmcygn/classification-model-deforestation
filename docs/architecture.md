# Architecture

## System Overview

DeforestAI is a two-tier application: a Python/FastAPI backend handling ML, data processing, and AI integration, and a React/TypeScript frontend for interactive map visualization and policy brief display.

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│  Explorer Page          │  Pipeline Page                │
│  ├─ Region Selector     │  ├─ Dataset Tab (stats)       │
│  ├─ RiskMap (Leaflet)   │  ├─ Training Tab (config)     │
│  ├─ CellDetailPanel     │  └─ Evaluation Tab (metrics)  │
│  ├─ TemporalPanel       │                               │
│  └─ PolicyBrief + PDF   │                               │
└───────────┬─────────────┴───────────────────────────────┘
            │ HTTP (Vite proxy /api → :8000)
┌───────────▼─────────────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│  api/routes/                                            │
│  ├─ explorer.py  → /regions, /grid, /cell, /temporal,   │
│  │                  /report, /geocode, /spatial-clusters │
│  └─ pipeline.py  → /dataset, /train, /runs, /retrain    │
│                                                         │
│  core/                                                  │
│  ├─ ml/        → dataset, training, evaluation,         │
│  │               explainability, registry,               │
│  │               feature_engineering, spatial            │
│  ├─ cv/        → change_detection (ExG from Sentinel-2) │
│  ├─ ai/        → policy brief (GPT-4o-mini), cache      │
│  ├─ data/      → fetch pipeline, source modules         │
│  └─ geo/       → geocoding (Nominatim)                  │
│                                                         │
│  data/                                                  │
│  ├─ raw/       → palawan_grid.csv,                      │
│  │               sierra_madre_grid.csv,                  │
│  │               mindanao_bukidnon_grid.csv              │
│  ├─ cache/     → cached policy briefs (JSON)            │
│  ├─ models/    → trained .joblib files                  │
│  └─ registry.db → SQLite run history                    │
└─────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────┐
│              External Data Sources                      │
│  ├─ GFW Tile Server (Hansen tree cover + loss tiles)    │
│  ├─ SRTM 30m (USGS/NASA elevation data)                │
│  ├─ OSM Overpass API (roads, protected areas)           │
│  ├─ WorldPop API (population density)                   │
│  ├─ Sentinel-2 cloudless / EOX (ExG vegetation index)   │
│  ├─ NASA FIRMS (fire hotspot data, via spatial proxy)   │
│  └─ OpenAI API (GPT-4o-mini, optional)                  │
└─────────────────────────────────────────────────────────┘
```

## Design Philosophy

The main contribution of this system is the **workflow**, not the model accuracy. The architecture is designed so that every AI component is modular, auditable, and replaceable. A stronger model (more data, better target formulation, time-series features) slots into the same pipeline without changing the data ingestion, explainability, reporting, or UI layers.

## Key Design Decisions

### Separation of Data Pipeline from Application

The data pipeline (`core/data/`) runs independently of the application server. It produces static CSV files consumed by the ML and API layers. This means:

- **Demos never break** — the application reads pre-baked CSVs, not live APIs
- **Pipeline failures are isolated** — if Overpass rate-limits us, the app still works
- **Reproducibility** — same CSVs produce same model every time

The `live_pipeline.py` module provides the same interface as an async function, ready to wire to an API endpoint when real-time fetching is needed.

### Two-Phase Feature Engineering

Features are split into two phases:
1. **Pipeline-time (stored in CSV):** 9 features from external APIs — Hansen GFW, SRTM, OSM, WorldPop, Sentinel-2/EOX
2. **Load-time (computed by `feature_engineering.py`):** 5 derived spatial features — neighbor loss statistics, loss acceleration, fire hotspot density

This allows adding new spatial features without re-running the expensive data fetch pipeline.

### ML Module is Location-Agnostic

The `core/ml/` modules (dataset, training, evaluation, explainability, registry) know nothing about Philippines or deforestation specifically. They operate on a DataFrame with up to 14 feature columns and a `high_risk` target. This means:

- Adding a new region (e.g., Visayas) requires only a new CSV
- Changing the model (e.g., XGBoost — already available) requires editing only `training.py`
- The same pipeline works for any binary classification on tabular geospatial data

### Temporal Holdout by Design

The dataset loader (`dataset.py`) enforces temporal separation at load time via `_apply_temporal_split()`, and the `FEATURE_MANIFEST` validates every feature's time window at train time. Features with post-2019 data windows are explicitly banned. The target is 2020-2022 loss. This prevents temporal leakage from being re-introduced as features are added.

The current model is an **exploratory baseline** (spatial CV F1=0.38). The temporal and evaluation infrastructure is designed to support a stronger model when better features and data become available.

### AI is Narrative Only

GPT-4o-mini is used exclusively for synthesizing structured data into prose (executive summaries, policy recommendations). All numerical outputs — risk scores, SHAP values, feature importances, hotspot coordinates — are computed deterministically by the ML pipeline.

This is deliberate: officials need to trust the numbers. AI-generated prose adds value by making structured data readable, but the numbers themselves must be auditable and reproducible.

### Cached Briefs for Keyless Demo

Pre-generated policy briefs for all three regions are shipped as JSON files. When `OPENAI_API_KEY` is not set, the API serves deterministic data sections with cached narrative prose, flagged with `_cached: true`. The frontend displays a "cached example" badge.

This ensures someone cloning the repo and running it without configuring keys gets the full experience.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/explorer/regions` | List available regions with metadata |
| GET | `/api/explorer/grid?run_id=X&region=Y` | Grid cells with predictions |
| GET | `/api/explorer/cell?lat=X&lon=Y&run_id=Z&region=R` | Cell detail with SHAP + provenance |
| GET | `/api/explorer/temporal?region=X` | Year-over-year loss data |
| POST | `/api/explorer/report?run_id=X&region=Y` | Generate policy brief |
| GET | `/api/explorer/clusters?run_id=X&region=Y` | DBSCAN deforestation front clusters |
| GET | `/api/explorer/geocode?q=X` | Geocode a location |
| GET | `/api/pipeline/dataset?region=X` | Dataset summary stats |
| GET | `/api/pipeline/dataset/sample?n=100` | Sample dataset rows |
| POST | `/api/pipeline/train` | Train model with hyperparameters |
| POST | `/api/pipeline/retrain` | Retrain with annotations/corrections |
| GET | `/api/pipeline/runs` | List all training runs |
| GET | `/api/pipeline/runs/{run_id}` | Get specific run details |

## Frontend Component Tree

```
App
├── Explorer (/)
│   ├── Sidebar
│   │   ├── Region Selector (Palawan / Sierra Madre / Mindanao Bukidnon)
│   │   ├── Model Selector (trained runs)
│   │   ├── Search Location (geocoding)
│   │   ├── Region Stats (high/low risk counts)
│   │   ├── TemporalPanel (forest loss area chart)
│   │   ├── Generate Report button
│   │   └── CellDetailPanel (SHAP + provenance)
│   ├── RiskMap (Leaflet + LayersControl)
│   │   ├── Street / Satellite basemap toggle
│   │   ├── Hansen GFW deforestation overlays
│   │   └── CircleMarkers (color-coded by risk)
│   └── PolicyBrief slide-over panel
│       ├── Executive Summary
│       ├── Risk Assessment
│       ├── Hotspots
│       ├── Top Drivers
│       ├── Recommendations
│       ├── Data Provenance
│       └── ExportButton (PDF)
└── Pipeline (/pipeline)
    ├── DatasetTab (stats, charts, feature table)
    ├── TrainingTab (hyperparameters, model type selection, run history)
    └── EvaluationTab (metrics, confusion matrix, feature importance, threshold curve)
```
