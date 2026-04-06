# Methodology

## Data Pipeline

### Grid Generation

Each region is divided into a regular lat/lon grid at 0.02° spacing (~2.2km per cell at the equator, slightly less at Philippine latitudes ~8-18°N). This produces:

- **Palawan:** ~29,700 total grid cells, of which 2,300 are forested land
- **Sierra Madre:** ~9,500 total grid cells, of which 3,378 are forested land
- **Mindanao Bukidnon:** ~6,000 total grid cells, of which 1,991 are forested land

Ocean and non-forest cells are filtered using Hansen GFW tile data — cells where neither tree cover nor forest loss is detected are excluded.

### Feature Engineering

Each land cell is enriched with up to 14 features from 7 data sources. The first 9 are stored in the CSV; the remaining 5 are derived at load time by `feature_engineering.py`.

### Feature Temporal Audit

Every feature is validated against the `FEATURE_MANIFEST` in `dataset.py` at train time. Features whose time window overlaps the target period (2020-2022) are rejected.

| Feature | Source Window | Available at Prediction Time | Status |
|---------|--------------|------------------------------|--------|
| `tree_cover_2000_pct` | 2000 (static) | Yes | **Keep** |
| `elevation_m` | Static | Yes | **Keep** |
| `slope_deg` | Static | Yes | **Keep** |
| `dist_to_road_km` | Snapshot | Yes | **Keep** |
| `dist_to_deforestation_frontier_km` | 2001-2019 | Yes | **Keep** |
| `protected_area` | Snapshot | Yes | **Keep** |
| `population_density_per_km2` | 2019 estimate | Yes | **Keep** |
| `annual_loss_rate_pct` | 2001-2019 | Yes | **Keep** |
| `exg_change_2018_2019` | 2018-2019 | Yes | **Keep** |
| `neighbor_loss_count` | 2001-2019 | Yes | **Keep** |
| `neighbor_loss_density` | 2001-2019 | Yes | **Keep** |
| `neighbor_mean_tree_cover` | 2000 (static) | Yes | **Keep** |
| `loss_acceleration` | 2001-2019 | Yes | **Keep** |
| `fire_hotspot_density` | 2001-2019 | Yes | **Keep** |
| `exg_change_2018_2022` | 2018-2022 | **No** (observes target) | **BANNED** |
| `loss_year` | 2001-2022 | **No** (is the target) | **BANNED** |
| `high_risk` | 2020-2022 | **No** (is the target) | **BANNED** |

**Base features (stored in CSV):**

| Feature | Source | Method | Unit |
|---------|--------|--------|------|
| `tree_cover_2000_pct` | GFW tile server | Sample `umd_tree_cover_density_2000` tile. Alpha > 0 = forest. Percentage estimated from deterministic distribution (45-95%). | % |
| `elevation_m` | SRTM 30m | Query `srtm` Python package (auto-downloads HGT files). Direct lookup by lat/lon. | meters |
| `slope_deg` | SRTM 30m | Compute gradient from 4 neighboring elevation points (N/S/E/W at 0.001° offset). `slope = atan(sqrt(dz/dx² + dz/dy²))`. | degrees |
| `dist_to_road_km` | OSM Overpass API | Fetch all `highway` ways (motorway through track) in region. Compute haversine distance from each cell to nearest road node. | km |
| `dist_to_deforestation_frontier_km` | Derived from Hansen | Haversine distance from each cell to nearest cell with pre-2020 loss (`loss_year` 1-19). Uses only historical loss to avoid target leakage. | km |
| `protected_area` | OSM Overpass API | Fetch `boundary=protected_area` and `leisure=nature_reserve` ways. Point-in-polygon test using Shapely. | binary (0/1) |
| `population_density_per_km2` | WorldPop API (2019) | Population density estimate for cell location (pre-target year). | per km² |
| `annual_loss_rate_pct` | Derived from Hansen | Regional loss ratio computed from pre-2020 loss only (cells with loss in 2001-2019 / total cells), annualized. Doubled for cells with observed pre-2020 loss. | %/year |
| `exg_change_2018_2019` | Sentinel-2 cloudless (EOX) | Excess Green index (ExG = 2*G - R - B) computed on satellite tiles for 2018 vs 2019. Negative values indicate vegetation loss. | dimensionless |

**Derived spatial features (computed at load time by `feature_engineering.py`):**

| Feature | Method | Unit |
|---------|--------|------|
| `neighbor_loss_count` | Count of cells within 15km radius that had pre-2020 loss (`pre_period_loss`). Uses BallTree for efficient spatial lookup. | count |
| `neighbor_loss_density` | Proportion of neighbors (within 15km) with pre-2020 loss. | ratio (0-1) |
| `neighbor_mean_tree_cover` | Mean `tree_cover_2000_pct` of neighbors within 15km. Captures whether the cell is in contiguous forest or fragmented landscape. | % |
| `loss_acceleration` | Cell's `annual_loss_rate_pct` relative to regional mean: `(cell_rate / regional_mean) - 1`. Positive values = loss accelerating above average. | dimensionless |
| `fire_hotspot_density` | Count of high-loss-rate neighbors within 15km (pre-2020 loss rate > 75th percentile). Spatial proxy for fire-driven clearing — uses only pre-period data, not the live FIRMS feed. | count |

### Temporal Holdout Design

To avoid target leakage, the data pipeline enforces strict temporal separation:

- **Features** use only 2001-2019 data (`loss_year` values 1-19)
- **Target** (`high_risk`) is loss in 2020-2022 only (`loss_year` values 20-22)
- `dist_to_deforestation_frontier_km` uses only pre-2020 loss cells
- `annual_loss_rate_pct` computed from pre-2020 loss only
- Spatial features (`neighbor_loss_count`, `neighbor_loss_density`) use `pre_period_loss` (2001-2019)
- `_apply_temporal_split()` in `dataset.py` recomputes labels from `loss_year` at load time

### Target Variable

`high_risk` is a binary label derived from Hansen GFW `loss_year` tiles:

- **1 (high risk):** The cell experienced tree cover loss in 2020-2022 (loss_year value 20-22 in Hansen v1.7)
- **0 (low risk):** No loss detected in that period

This is **ground truth from real satellite observations**, not a synthetic label. The temporal holdout ensures the model learns to predict future loss from past feature profiles.

### Target Formulation Limitations

The binary `high_risk` label has known weaknesses that constrain model quality:

**Regime collapse:** A cell on an active deforestation frontier (loss every year 2020-2022) gets the same label as a cell with a single small clearing event. The binary target cannot distinguish sustained pressure from one-off disturbance.

**Arbitrary temporal boundary:** A cell that lost forest in 2019 (`loss_year=19`) is labeled low-risk, while a cell that lost forest in 2020 (`loss_year=20`) is high-risk. The model treats the 2019/2020 boundary as meaningful when deforestation processes are continuous.

**No time-to-event modeling:** The target doesn't encode *when* within 2020-2022 loss occurred or the rate of progression. A survival/hazard model would be more appropriate for genuine risk prediction.

**Alternative formulations (not implemented, noted for future work):**
- **Continuous risk score:** probability of loss per year, calibrated against observed rates
- **Time-to-event:** survival analysis predicting years until first loss
- **Multi-class:** separate "active frontier," "one-off clearing," "stable" classes
- **Rolling window:** predict next-year loss from trailing 3-year features, evaluated year-by-year

These limitations are fundamental to the target definition, not to the model architecture. The current binary formulation is a reasonable starting point for a prototype but would need refinement for operational deployment.

### Data Provenance

Every feature carries metadata tracing it to its source:

```json
{
  "tree_cover_2000_pct": {
    "source": "Hansen/UMD GFW v1.7",
    "url": "https://glad.umd.edu/dataset/gfw"
  },
  "elevation_m": {
    "source": "SRTM 30m (USGS/NASA)",
    "url": "https://www.usgs.gov/centers/eros"
  },
  "exg_change_2018_2019": {
    "source": "Sentinel-2 cloudless (EOX)",
    "url": "https://s2maps.eu/"
  }
}
```

This metadata is surfaced in the cell detail panel (per feature) and in policy briefs (data sources section).

## CV Change Detection

### ExG Vegetation Index

For each cell, an Excess Green (ExG) index is computed from Sentinel-2 cloudless composites (via EOX tile server):

- **ExG = 2*G - R - B** (normalized by total intensity)
- Computed on satellite tiles for two time windows (2018 vs 2019)
- `exg_change_2018_2019` = difference between the two periods
- Negative values indicate vegetation loss, complementing Hansen's binary loss with a continuous measure

### DBSCAN Spatial Clustering

High-risk cells are spatially clustered using DBSCAN to identify contiguous deforestation hotspot zones:

- **eps:** calibrated to grid resolution (0.03° for 0.02° grid — captures immediate neighbors)
- **min_samples:** 3 (minimum cells to form a cluster)
- Clusters identify coherent deforestation fronts rather than isolated high-risk cells
- Noise points (isolated high-risk cells) are flagged as potential early-warning signals

## ML Model

### Algorithm

**Random Forest Classifier** (scikit-learn `RandomForestClassifier`).

Default hyperparameters:
- `n_estimators`: 100
- `max_depth`: 10
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `class_weight`: "balanced" (critical for handling the 0.9-8.2% high-risk class)
- `random_state`: 42 (reproducibility)

**XGBoost** is also available as an alternative via the training UI for comparison.

### Training Procedure

1. Load all available region data (Palawan, Sierra Madre, Mindanao Bukidnon — or a specific region)
2. Apply temporal split: recompute `high_risk` from `loss_year` (20-22 = positive)
3. Compute derived spatial features if not present (neighbor stats, loss acceleration, fire density)
4. Extract up to 14 feature columns and `high_risk` target
5. Stratified train/test split (80/20, preserving class ratio) or spatial split
6. Fit model on training set
7. Evaluate on test set (accuracy, precision, recall, F1, confusion matrix)
8. Compute feature importances from the trained model
9. Save model to `.joblib` file and record run in SQLite registry

All hyperparameters are user-configurable via the Training tab UI.

### Explainability

Each prediction is accompanied by SHAP values computed via `shap.TreeExplainer`:

- **Per-feature contributions:** How much each feature pushed the prediction toward high or low risk
- **Base value:** The model's average prediction (prior to feature influence)
- **Summary text:** "High Risk — driven by: annual_loss_rate_pct (1.52, +0.11), dist_to_road_km (30.92, +0.04), ..."

SHAP TreeExplainer is exact for tree-based models — it computes true Shapley values in polynomial time using the tree structure, not approximations.

## AI Policy Brief Generation

### Architecture

The policy brief is a hybrid of deterministic computation and AI synthesis:

**Deterministic sections** (computed by the ML pipeline):
- Site overview (cell count, area, bounds)
- Risk assessment (high/low counts, percentages, hectares)
- Hotspots (top 5 cells by probability, with coordinates)
- Top risk drivers (feature importances from trained model)
- Notable findings (auto-flagged anomalies)
- Data provenance (source per feature)

**AI-generated sections** (GPT-4o-mini):
- Executive summary (2-3 sentences citing specific numbers from above)
- Policy recommendations (3-5 bullets citing Philippine regulations)

### Prompt Design

The prompt explicitly instructs the model to:
- Cite specific numbers from the provided data
- Not add information not present in the data
- Frame recommendations in Philippine regulatory context (NIPAS Act, EO 23, DENR AOs)
- Respond in structured JSON format

### Fallback

When `OPENAI_API_KEY` is not set:
1. Check for cached brief (`data/cache/{region}_brief.json`)
2. If found, serve deterministic data sections with cached narrative prose, flagged with `_cached: true`
3. If not found, serve deterministic sections only with "AI unavailable" message
