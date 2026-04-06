# DeforestAI — Design Spec

## Product Concept

A deforestation risk intelligence platform with two core surfaces:

1. **Explorer** — Interactive map + data dashboard. Navigate to a region, see deforestation stats, model-predicted risk scores with per-cell explainability, and generate structured analysis reports against a fixed schema. Not a chat interface — deterministic, templated outputs with an AI narrative synthesis layer.

2. **Pipeline** — A real ML training pipeline UI. Dataset exploration, feature engineering config, model training, evaluation dashboard, run history, and model comparison. Trains a real scikit-learn model on tabular deforestation risk features.

**Target users**: Conservation NGOs, researchers, and investigative journalists monitoring deforestation.

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend | FastAPI (Python) | Async API, native ML ecosystem |
| Frontend | React + Vite + TypeScript | Fast dev, type safety |
| Maps | Leaflet (OpenStreetMap tiles) | Free, no API key |
| ML | scikit-learn (Random Forest) | No GPU, fast iterations, interpretable |
| Explainability | SHAP | Per-prediction feature contributions, deterministic |
| AI Synthesis | Claude API (Anthropic SDK) | Narrative layer on structured report |
| Database | SQLite | Run history, model registry, lightweight |
| Data | Curated sample dataset (Hansen/GFW-derived) | Public, well-documented |

---

## Architecture

```
backend/
  api/
    routes/
      explorer.py         # Region lookup, stats, report generation
      pipeline.py         # Dataset, training, evaluation endpoints
  core/
    ml/
      dataset.py          # Data loading, feature engineering
      training.py         # Training loop, model serialization
      evaluation.py       # Metrics, SHAP, confusion matrix
      registry.py         # Model versioning, metadata (SQLite)
    ai/
      analysis.py         # Claude API — narrative synthesis against report schema
    geo/
      lookup.py           # Geocoding, region resolution
  data/
    raw/                  # Source CSVs / GeoJSONs
    processed/            # Feature-engineered datasets
    models/               # Saved model artifacts (.joblib)
    runs/                 # Training run logs (JSON)

frontend/
  src/
    pages/
      Explorer.tsx        # Map + stats + report panel
      Pipeline.tsx        # Dataset, training, evaluation tabs
    components/
      map/                # Leaflet map, risk overlay, cell detail panel
      pipeline/           # Training config, metrics charts, run history
      reports/            # Report card, explainability summary
```

---

## ML Model

### Task
Binary classification — predict whether a grid cell is high-risk or low-risk for deforestation.

### Features (tabular, from public datasets)
- Historical tree cover loss rate (last 5-10 years)
- Proximity to roads/infrastructure
- Elevation and slope
- Current tree cover percentage
- Distance to existing deforestation frontier
- Protected area status (binary)
- Population density nearby

### Target
Derived from held-out recent years — whether significant tree cover loss occurred.

### Model
Random Forest classifier (scikit-learn). Produces native feature importance. Model-agnostic pipeline — swappable for XGBoost, neural nets, etc.

### Explainability
SHAP values computed per prediction. Deterministic, templated output:
> "High risk driven by: road proximity (0.3km, contributes +0.28), recent loss rate (4.2%/yr, contributes +0.22), low elevation (120m, contributes +0.15)"

### What's Real vs. Stubbed
- **Real**: Full pipeline from raw data → features → train → evaluate → save. Training loop, SHAP explainability, model serialization, run comparison, evaluation metrics.
- **Stubbed with clear interfaces**: Raw data is a curated sample dataset covering a region of the Brazilian Amazon (Rondônia — one of the most deforested areas). Synthetic but realistic tabular data generated from known distributions of Hansen/GFW statistics. Documentation specifies exactly how to replace with real GFW API data and extend to CV-based satellite image classification.

---

## Pipeline UI

### Dataset Tab
- View loaded dataset: row count, feature distributions (histograms), geographic coverage on mini-map
- Feature selection toggles — pick which features to include
- Train/test split slider

### Training Tab
- Hyperparameter config form (n_estimators, max_depth, min_samples_split, etc.)
- "Train" button → real training run with progress indicator
- Run history table: timestamp, params, accuracy, F1, model ID

### Evaluation Tab
- Select any run from history
- Confusion matrix visualization
- Feature importance bar chart
- Precision/recall/F1 summary
- Compare two runs side-by-side
- "Export model" action (saves artifact + metadata)

---

## Explorer UI

### Map (main area)
- Leaflet map with deforestation risk overlay layer
- Grid cells color-coded by risk score (from selected trained model)
- Click a cell to open detail panel

### Grid Cell Detail Panel
- Feature values table (elevation, road proximity, tree cover, etc.)
- Risk score + confidence
- **Explainability summary**: SHAP-based, deterministic, templated text showing which features drove this specific prediction and by how much

### Region Panel (sidebar)
- Search bar (geocoding — address/coordinates, not NL)
- Region stats: total area, tree cover %, loss over time (line chart), active risk zones count

### Report Generation
"Generate Report" button produces a structured report against a fixed schema:

1. **Region summary** — area, coverage statistics
2. **Risk distribution** — % of cells at high/medium/low risk
3. **Top risk factors** — aggregated feature importances across the region
4. **Notable data points** — auto-flagged anomalies (e.g., "3 cells transitioned from low to high risk", "cluster of high-risk cells near road BR-364")
5. **Trend analysis** — historical loss trajectory
6. **Narrative synthesis** — Claude stitches the structured data into readable prose against the report template. This is the only non-deterministic layer.

**Data flow**: model predictions → SHAP feature contributions → templated explainability → structured report schema → Claude fills narrative layer only.

### Model Selector
Dropdown to pick which trained model's predictions to display on the map. Ties explorer directly to pipeline output.

---

## Documentation & Roadmap Requirements

The submission must include extensive documentation on:
- What is stubbed and why
- Clear extension points (interfaces where real data sources, CV models, etc. plug in)
- Roadmap and next steps for production-ifying each component
- Architecture decision records for key choices

---

## Out of Scope (for this prototype)
- Real-time satellite data ingestion
- CV/image-based classification (documented as extension)
- User authentication
- Multi-tenant / per-client model management (documented as extension)
- CI/CD training pipeline automation (documented as extension)
- Deployment/infrastructure
