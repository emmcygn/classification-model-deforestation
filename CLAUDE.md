# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Mozaic Earth Product Engineer Tech Test** — a "Nature Intelligence" micro-tool prototype addressing a problem in Nature Tech, Climate Adaptation, or Geospatial Data. Timeline: ~1 week (extended from the spec's 4-6 hours).

### What Evaluators Care About (in priority order)
1. **Product Intuition** — creative problem identification and solution
2. **AI Leverage** — meaningful AI/CV/ML integration (not cosmetic)
3. **Architectural Steering** — cohesive system, not AI slop
4. **AI Experience** — does the user-AI interaction genuinely solve a need?

### Constraints from Spec
- Frontend: clean, functional UI
- Backend: Next.js API Routes or FastAPI
- AI Integration: must add genuine value (data synthesis, image classification, automated reporting, spatial-logic assistant)
- Use local models or free-tier services — avoid incurring costs
- Don't over-engineer edge cases or perfect the UI

## Tech Stack

- **Backend**: Python 3.11, FastAPI, uvicorn
- **ML**: scikit-learn (Random Forest), SHAP (TreeExplainer), pandas, numpy
- **AI**: OpenAI GPT-4o-mini — policy brief narrative synthesis
- **Data Sources**: Hansen/UMD GFW tiles, SRTM 30m, OpenStreetMap Overpass API
- **Database**: SQLite via aiosqlite (model registry / run history)
- **Frontend**: React 18, Vite, TypeScript
- **Styling**: Tailwind CSS v4
- **Maps**: Leaflet + react-leaflet
- **Charts**: Recharts
- **Geocoding**: geopy (Nominatim)

## Commands

```bash
# Backend
cd backend
venv/Scripts/activate          # Windows (or source venv/bin/activate on Unix)
python -m pytest tests/ -v     # Run tests
python -m core.data.fetch_philippines  # Fetch real Philippine data (~15 min)
uvicorn main:app --port 8000 --reload  # Start API server

# Frontend
cd frontend
npm install
npm run dev                    # Dev server on :5173
npx vite build                 # Production build
```

## Architecture

```
backend/
  main.py                      # FastAPI app entry, CORS, router registration
  api/routes/
    pipeline.py                # /api/pipeline/* — dataset info, train, runs (multi-region)
    explorer.py                # /api/explorer/* — regions, grid, cell, temporal, report
  core/
    data/
      fetch_philippines.py     # One-shot pipeline: Hansen + SRTM + OSM + WorldPop + EOX → CSV
      live_pipeline.py         # On-demand fetch for arbitrary bounding boxes (stub)
      sources/
        hansen.py              # GFW tile pixel sampling (tree cover, loss, loss_year)
        srtm_source.py         # SRTM elevation + slope computation
        osm.py                 # Overpass API for roads + protected areas
        worldpop.py            # WorldPop population density API
        firms.py               # NASA FIRMS fire hotspot data
    ml/
      dataset.py               # Multi-region dataset loader (Palawan, Sierra Madre, Mindanao Bukidnon)
      training.py              # Random Forest / XGBoost training with class_weight=balanced
      evaluation.py            # Accuracy, F1, confusion matrix, feature importance
      explainability.py        # SHAP TreeExplainer per-prediction explanations
      feature_engineering.py   # Derived spatial features (neighbor stats, loss acceleration, fire density)
      spatial.py               # DBSCAN spatial clustering for deforestation fronts
      registry.py              # SQLite-backed run history (RunRegistry class)
    cv/
      change_detection.py      # Sentinel-2 cloudless ExG vegetation index (EOX)
      validation.py            # CV result validation
    ai/
      analysis.py              # GPT-4o-mini structured policy brief generator
      brief_cache.py           # Cached briefs for keyless demo
    geo/lookup.py              # Nominatim geocoding
  data/
    raw/
      palawan_grid.csv         # Real Philippine data (2,300 land cells)
      sierra_madre_grid.csv    # Real Philippine data (3,378 land cells)
      mindanao_bukidnon_grid.csv # Real Philippine data (1,991 land cells)
    cache/
      palawan_brief.json       # Cached policy brief for keyless demo
      sierra_madre_brief.json  # Cached policy brief for keyless demo
      mindanao_bukidnon_brief.json # Cached policy brief for keyless demo
    models/                    # Saved .joblib model files
    registry.db                # SQLite run history (created at runtime)
  tests/                       # pytest tests for dataset, training, explainability, registry

frontend/
  src/
    App.tsx                    # Router shell — Explorer + Pipeline nav
    lib/api.ts                 # Typed API client with region, temporal, policy brief types
    pages/
      Explorer.tsx             # Region selector, map, temporal chart, policy brief, PDF export
      Pipeline.tsx             # Tabbed: Dataset / Training / Evaluation
    components/
      map/
        RiskMap.tsx             # Leaflet map with satellite toggle + Hansen GFW overlays
        CellDetailPanel.tsx    # SHAP explanation with data provenance per feature
        TemporalPanel.tsx      # Year-over-year forest loss area chart
      report/
        PolicyBrief.tsx        # Structured policy brief display with cached badge
        ExportButton.tsx       # PDF export via html2canvas + jsPDF
      pipeline/
        DatasetTab.tsx         # Dataset stats, class distribution chart, feature table
        TrainingTab.tsx        # Hyperparameter form, train button, run history table
        EvaluationTab.tsx      # Metrics comparison, confusion matrix, feature importance chart
```

## Submission Deliverables

The final repo must include a **README.md** with:
- **The Idea**: What problem are you solving?
- **The Thinking**: How did you solve it? Which AI models?
- **Reflections**: What would you do differently? Limitations?
- **Setup**: How to run locally

Submit to: neil@mozaic.earth — Subject: "Product Engineer Tech Test - [Your Name]"
