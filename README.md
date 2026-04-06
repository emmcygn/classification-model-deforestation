# DeforestAI Philippines

> A Nature Intelligence micro-tool that turns satellite-derived deforestation data into explainable, auditable risk screening for Philippine provincial officials — from raw geospatial features to SHAP-explained predictions to AI-synthesized policy briefs, in one workflow.

**Focus regions:** Palawan (UNESCO Biosphere Reserve), Sierra Madre (largest remaining Philippine rainforest), and Mindanao Bukidnon (highland plateau forest).

**Architecture docs:** See [`docs/`](docs/) for detailed architecture, tradeoffs, methodology, and model performance notes.

## The Idea

The Philippines loses ~47,000 hectares of forest annually. Provincial environment officers (DENR/PENRO/CENRO) lack accessible, data-driven tools to identify where deforestation risk is highest, what's driving it, and where to focus monitoring resources.

The core insight: **the value isn't in the model — it's in the pipeline.** A provincial officer doesn't need a perfect classifier. They need a system that takes satellite data they can't interpret themselves, produces prioritized risk maps they can act on, explains *why* each cell is flagged, and generates structured reports they can forward to a governor. The ML model is one replaceable component inside that system.

DeforestAI demonstrates this end-to-end:

1. **Real geospatial data pipeline** — 14 features from 7 sources (Hansen GFW, SRTM, OSM, WorldPop, Sentinel-2, spatial statistics), each validated against a temporal manifest
2. **Explainable risk screening** — SHAP TreeExplainer shows exactly which features drove each prediction. Officials see *why* a cell is flagged, not just a color on a map
3. **AI-synthesized policy briefs** — GPT-4o-mini generates executive summaries and regulatory recommendations (NIPAS Act, EO 23, DENR AOs) from deterministic model outputs. All numbers are computed, not hallucinated
4. **Full data provenance** — every feature traces back to its source with URLs. Every run persists its dataset hash, test indices, and split strategy for reproducibility
5. **Honest evaluation** — spatial CV as the default metric, temporal holdout enforced by a feature manifest that bans post-2019 data at train time, per-region performance breakdowns

## The Thinking

### Why this architecture

The system is designed around a principle: **AI components should be modular, auditable, and gracefully degradable.**

- **FastAPI + React** — Python backend for natural ML ecosystem access (scikit-learn, SHAP, pandas). React frontend for interactive map/dashboard UX.
- **Tabular features over computer vision** — A Random Forest on engineered geospatial features is interpretable, runs without GPU, and produces SHAP explanations that officials can audit. The pipeline is model-agnostic — swapping for XGBoost (already available in the UI) or a neural net requires changing one module.
- **SHAP for explainability** — Deterministic output from real model internals. Not LLM-hallucinated explanations. Each prediction decomposes into per-feature contributions that sum to the prediction.
- **GPT-4o-mini for narrative only** — AI synthesizes structured data into policy-ready prose. All numerical sections are computed deterministically. The tool degrades gracefully without an API key (cached example briefs shipped in the repo).
- **Feature manifest with temporal guardrails** — Every feature declares its source, time window, and training eligibility. The training pipeline validates the manifest and rejects post-target features. This prevents temporal leakage from being re-introduced as features are added.
- **Spatial CV as default** — The honest metric (not the flattering random-split one) is the primary evaluation path in both backend and UI.

### AI integration points

| Component | AI/ML technique | Value added |
|-----------|----------------|-------------|
| Risk classification | Random Forest (scikit-learn) | Screens 7,669 cells across 3 regions — replaces manual satellite image review |
| Explainability | SHAP TreeExplainer | Per-cell feature attribution — officials understand *why*, enabling domain expertise to override the model |
| Spatial clustering | DBSCAN | Identifies contiguous deforestation fronts from scattered cell predictions |
| Change detection | ExG index differencing (Sentinel-2) | Continuous vegetation health signal from satellite imagery |
| Policy briefs | GPT-4o-mini | Turns structured data into prose that can be forwarded to decision-makers |
| Active learning | Uncertainty sampling | Suggests cells where human review most improves model quality |

### Data sources (all free, no auth required)

| Feature | Source | Access |
|---------|--------|--------|
| Tree cover 2000 / forest loss | Hansen/UMD GFW v1.7 | GFW tile server |
| Elevation + slope | SRTM 30m (USGS/NASA) | `srtm` Python package |
| Road distances | OpenStreetMap | Overpass API |
| Protected areas | OpenStreetMap | Overpass API |
| Population density (2019) | WorldPop | WorldPop REST API |
| Vegetation change (ExG 2018-2019) | Sentinel-2 cloudless (EOX) | EOX tile server |
| Spatial/neighbor features | Derived from Hansen pre-2020 loss | Computed at load time |
| Fire hotspot proxy | Derived from pre-2020 loss rate | Spatial proxy (not live FIRMS) |

## Reflections

### What this demonstrates

The main contribution is the **workflow**, not the forecast quality:

- **Product-grade pipeline** — satellite data → feature engineering → temporal holdout → model training → SHAP explanation → policy brief → PDF export, all connected through a usable UI
- **Multiple AI modalities working together** — ML classification, explainability, spatial clustering, computer vision, LLM narrative synthesis, and active learning — each adding genuine value at a different layer
- **Trust engineering** — data provenance per feature, SHAP per prediction, cached badge for offline briefs, held-out calibration view, honest metrics front-and-center
- **Evaluation honesty** — temporal holdout, feature manifest with banned features, spatial CV as default, per-region breakdown, documented target limitations

### What this does NOT demonstrate

I want to be direct about the model's limitations:

- **The model is an exploratory baseline, not operational.** Spatial CV F1=0.38, precision 28%. That means ~7 of 10 flagged cells are false positives. Useful for broad regional prioritization ("these municipalities deserve attention"), not for site-level operational decisions ("send a team to this grid cell").
- **The target formulation is coarse.** Binary "any loss in 2020-2022" collapses active deforestation frontiers and one-off clearing events into the same label. A cell that lost forest in 2019 is treated as low-risk. See [methodology docs](docs/methodology.md#target-formulation-limitations) for why this matters and what alternatives exist.
- **The model is a replaceable component.** The pipeline — data ingestion, feature validation, explainability, reporting — is the durable contribution. A stronger model (more data, time-series features, better target formulation) slots directly into this infrastructure.

### What I'd build with more time

- **Better model formulation** — yearly panel data instead of one collapsed binary label, time-series NDVI trajectory features, survival/hazard model for time-to-event risk
- **Mobile field data capture** — local communities report encroachment via mobile app, validated by PENRO ecologists remotely (maps to Mozaic Earth's citizen science model)
- **Multi-region on-demand** — activate the live pipeline for any Philippine province, not just pre-baked regions
- **TNFD-aligned reporting** — structure outputs to support Task Force on Nature-related Financial Disclosures governance/strategy/risk/metrics pillars

### Limitations

- Post-fix honest metrics (spatial CV): F1=0.38, Precision=28%, Recall=63%. Class distribution after temporal recompute: 164 high-risk cells out of 7,669 (2.1%)
- Sierra Madre has very low positive rate (0.7%, 23 high-risk cells) — insufficient training signal for this region
- Tree cover percentage is estimated from GFW forest presence tiles (exact % not available via tile API)
- Single static dataset — no real-time data ingestion (live pipeline stub is ready)
- SHAP computation is per-request — would need pre-computation at scale

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- (Optional) OpenAI API key for live policy brief generation

### Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix:
source venv/bin/activate

pip install -r requirements.txt

# Start server
uvicorn main:app --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### Environment Variables
```bash
# Optional — policy briefs work without it (shows cached examples)
export OPENAI_API_KEY=your-key-here
```

### Quick Start
1. Start backend and frontend (see above)
2. Go to Pipeline > Training > click "Train Model" (spatial CV is the default)
3. Go to Explorer > select a region
4. Select the trained model > see risk map
5. Click any cell for SHAP explanation with data provenance
6. Click "Generate Report" for structured policy brief
7. Export as PDF
