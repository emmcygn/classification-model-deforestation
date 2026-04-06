# Design Tradeoffs

## Data Pipeline

### Pre-baked CSVs vs. Live API Fetching

**Chose:** Pre-baked CSVs as primary, with a live pipeline stub.

**Why:** A demo that depends on external APIs (Hansen tiles, Overpass, SRTM) is fragile. During the Palawan data fetch, Overpass returned 429 (rate limit) and 504 (timeout) errors for several chunks. If this happened during a live demo or LinkedIn screen recording, the tool would appear broken.

**Tradeoff:** Data is static. Users can't analyze new regions without re-running the pipeline. The `live_pipeline.py` stub addresses this architecturally — activating it is a one-line change.

### 0.02° Grid Resolution

**Chose:** 0.02° (~2.2km per cell).

**Why:** Balances spatial resolution against API call volume. At 0.02°, Palawan produces ~29,700 total grid cells but only ~2,300 land cells after ocean filtering. Fine enough for site-level decisions at the provincial scale.

**Tradeoff:** Still coarse for barangay-level work (would want 100-500m). The architecture supports finer resolution — just change the `step` parameter — but each halving 4x the cell count and API calls. At 0.01° the fetch pipeline becomes impractically slow.

### Two-Phase Feature Engineering

**Chose:** 9 base features stored in CSV + 5 derived spatial features computed at load time.

**Why:** External API calls (Hansen, SRTM, OSM, WorldPop, EOX) are slow and rate-limited. Spatial features (neighbor stats, loss acceleration) depend on the full grid being available and can be recomputed cheaply. Separating these phases means adding new derived features doesn't require re-running the expensive fetch pipeline.

**Tradeoff:** Load-time feature computation adds ~1-2 seconds when loading a dataset. Acceptable for a prototype; would pre-compute and store for production.

### WorldPop API vs. Road Proximity Proxy

**Chose:** WorldPop API for population density (upgraded from the original road proximity proxy).

**Why:** The original approach used exponential decay from nearest road as a crude population estimate. WorldPop provides actual modeled population density data, which is significantly more accurate, especially for remote indigenous communities that may not be near roads.

**Tradeoff:** WorldPop API adds an external dependency during data fetch. The fallback to road proximity proxy is still available if the API is unavailable.

### Sentinel-2 ExG vs. NDVI/EVI

**Chose:** Excess Green (ExG) index from Sentinel-2 cloudless composites via EOX tile server.

**Why:** EOX provides free, pre-processed cloudless composites — no need to handle cloud masking or download raw Sentinel-2 scenes. ExG (2*G - R - B) is simpler than NDVI (which requires near-infrared) but still captures vegetation greenness change between time periods.

**Tradeoff:** ExG is less sensitive than NDVI for distinguishing vegetation types. The EOX composites are annual, limiting temporal precision. For production, raw Sentinel-2 NDVI time series would be better.

### Fire Hotspot Density as Spatial Proxy vs. Real FIRMS Data

**Chose:** Spatial proxy (count of high-loss neighbors) rather than querying NASA FIRMS API directly.

**Why:** FIRMS API has usage limits and requires additional pipeline complexity. The spatial proxy captures a similar signal — areas where many neighboring cells experienced loss likely experienced fire-related clearing.

**Tradeoff:** This is a proxy, not actual fire data. Real FIRMS hotspot counts would be more accurate and could distinguish fire-driven loss from other deforestation types. The feature name is somewhat misleading — noted in data provenance.

## ML Model

### Random Forest vs. Gradient Boosting vs. Neural Network

**Chose:** Random Forest with `class_weight="balanced"` as default, XGBoost available as alternative.

**Why:**
- Interpretable: SHAP TreeExplainer produces exact feature contributions, not approximations
- No GPU required: runs on any machine in seconds
- Handles class imbalance well with `class_weight="balanced"`
- Feature importances are a direct model output, not a post-hoc analysis

**Tradeoff:** Random Forest typically underperforms XGBoost on tabular data by 1-3%. XGBoost is available in the training UI for comparison. For a prototype where the workflow and explainability matter more than marginal model accuracy, Random Forest is the right default. The model is a replaceable baseline — the pipeline infrastructure supports any scikit-learn-compatible classifier.

### Temporal Holdout vs. Random Split

**Chose:** Temporal holdout (features from 2001-2019, target from 2020-2022) enforced at load time.

**Why:** The original random split allowed target leakage — features derived from loss data could encode information about the target period. Temporal holdout is the correct evaluation protocol for a "predict future deforestation" task. See [review-findings.md](review-findings.md) for the full leakage analysis.

**Tradeoff:** Reduces effective training data (can't use 2020-2022 observations as features). Metrics drop significantly (spatial CV F1 from 0.48 leaked to 0.38 honest), but are now trustworthy.

### SHAP TreeExplainer vs. LIME vs. Permutation Importance

**Chose:** SHAP TreeExplainer.

**Why:** TreeExplainer is exact for tree-based models (not approximate). It computes Shapley values in polynomial time via the tree structure, producing per-feature contributions that sum to the prediction. LIME is model-agnostic but approximate and slower. Permutation importance is global, not per-prediction.

**Tradeoff:** SHAP computation is per-request (~50ms per cell). At scale (thousands of cells), this would need pre-computation and caching. For a prototype serving one user at a time, it's fine.

## AI Integration

### GPT-4o-mini vs. Claude vs. Local Model

**Chose:** GPT-4o-mini via OpenAI API.

**Why:** ~$0.15/1M input tokens, structured JSON output mode, adequate quality for policy brief generation. Claude Sonnet is higher quality but more expensive. Local models (Ollama) would be free but require setup and produce inconsistent structured output.

**Tradeoff:** Requires an API key for live generation. Mitigated by cached briefs for the keyless experience.

### Structured JSON Output vs. Free-form Prose

**Chose:** Structured policy brief with JSON schema. AI generates only `executive_summary` and `recommendations`; all other sections are deterministic.

**Why:** Government officials need predictable, auditable outputs. A free-form AI response could hallucinate statistics or make up regulatory references. By constraining AI to prose synthesis of pre-computed data, we get readable output without sacrificing accuracy.

**Tradeoff:** Less "impressive" than a full AI-generated report. But more trustworthy — which is what matters for the target audience.

## Frontend

### Leaflet vs. Mapbox GL JS vs. Deck.gl

**Chose:** Leaflet with react-leaflet.

**Why:** Free (no API key), well-documented, handles our use case (circle markers on a tile map) without complexity. Mapbox GL JS is more performant for large datasets but requires an API key. Deck.gl is overkill for <5000 markers.

**Tradeoff:** Leaflet struggles with >10,000 markers. Our filtered datasets (2,300-3,378 cells per region) are within limits. If expanding to full Philippines (~60k cells), would need to switch to Deck.gl or server-side rendering.

### PDF Export via html2canvas + jsPDF vs. Server-side PDF

**Chose:** Client-side PDF generation.

**Why:** No server-side dependencies (wkhtmltopdf, Puppeteer, etc.). The user's browser renders the brief, captures it as an image, and produces a PDF. Works offline after initial page load.

**Tradeoff:** PDF quality is limited to screen rendering. Doesn't support multi-page briefs well. A production system would use server-side rendering with proper pagination and branding.
