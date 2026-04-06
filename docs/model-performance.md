# Model Performance

The model is an **exploratory baseline** included to demonstrate the end-to-end workflow and evaluation discipline, not as evidence of production-grade forecasting. The pipeline — data ingestion, feature validation, explainability, reporting — is the durable contribution. A stronger model slots directly into this infrastructure.

## Current Results

Trained on combined Palawan + Sierra Madre + Mindanao Bukidnon data (7,669 cells total, 164 high-risk after temporal recompute). Features use temporal holdout: trained on pre-2020 observations, predicting loss in 2020-2022. Default evaluation uses spatial cross-validation (the honest metric).

### Random Split (80/20 stratified)

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| F1 Score | ~0.55 |
| Precision | ~40% |
| Recall | ~85% |

### Spatial Cross-Validation (honest estimate)

| Metric | Value |
|--------|-------|
| F1 Score | 0.38 |
| Precision | 28% |
| Recall | 63% |

Spatial CV is the honest generalization estimate — it tests how the model performs on geographically unseen areas.

### Per-Region Performance

*Note: these numbers will change after retraining on the clean dataset (ExG leak fixed, forest-only filter, spatial CV default). Retrain via the Pipeline UI to get current numbers.*

| Region | Notes |
|--------|-------|
| Palawan | Highest positive rate (5.1%), most training signal |
| Mindanao Bukidnon | Highland plateau, moderate positive rate (1.2%) |
| Sierra Madre | Very sparse positives (0.7%, 23 cells) — model has minimal signal |

### Class Distribution (after temporal recompute)

These are the correct numbers after `_apply_temporal_split()` recomputes `high_risk` from `loss_year >= 20`. The raw CSVs may show higher counts from an older generation; runtime recompute is authoritative.

| Region | Total Cells | High Risk | % High Risk |
|--------|-------------|-----------|-------------|
| Palawan | 2,300 | 118 | 5.1% |
| Sierra Madre | 3,378 | 23 | 0.7% |
| Mindanao Bukidnon | 1,991 | 23 | 1.2% |
| Combined | 7,669 | 164 | 2.1% |

The class imbalance is severe (2.1% positive rate). This is addressed by `class_weight="balanced"` which inversely weights classes by frequency.

### Feature Importances (14 features)

| Feature | Source | Interpretation |
|---------|--------|----------------|
| `dist_to_deforestation_frontier_km` | Derived from Hansen (pre-2020 loss only) | Proximity to existing deforestation — strongest predictor, consistent with frontier expansion literature |
| `annual_loss_rate_pct` | Derived from Hansen (pre-2020 loss only) | Historical loss rate strongly predicts future loss |
| `tree_cover_2000_pct` | GFW tile server | Higher baseline forest cover = more to lose |
| `exg_change_2018_2019` | Sentinel-2 cloudless (EOX) | Vegetation greenness change (pre-target period only) |
| `neighbor_loss_density` | Derived spatial | Proportion of neighboring cells with pre-2020 loss — captures spatial contagion |
| `neighbor_loss_count` | Derived spatial | Count of neighboring cells with loss within 15km radius |
| `loss_acceleration` | Derived spatial | Local loss rate relative to regional average — flags accelerating fronts |
| `fire_hotspot_density` | Derived spatial | Count of high-loss neighbors as fire proxy |
| `neighbor_mean_tree_cover` | Derived spatial | Mean tree cover of neighbors — context for isolated vs. contiguous forest |
| `dist_to_road_km` | OSM Overpass API | Road proximity enables logging and land conversion access |
| `elevation_m` | SRTM 30m | Lower elevations are more accessible and more at risk |
| `population_density_per_km2` | WorldPop API | Human activity correlates with forest loss |
| `slope_deg` | SRTM 30m | Steeper terrain is harder to log |
| `protected_area` | OSM Overpass API | Protection status — weak predictor reflecting enforcement gaps |

### Interpretation Notes

**Spatial CV F1=0.38 is the honest metric.** Geographic autocorrelation inflates random-split scores because nearby cells share similar features and outcomes. Spatial CV reveals how the model performs on genuinely new regions.

**Temporal holdout eliminates target leakage.** Features are computed from 2001-2019 data only; the target is loss in 2020-2022. The model genuinely predicts future risk from past observations. See [review-findings.md](review-findings.md) for details on the leakage fix.

**Per-region variation is expected.** Palawan's higher positive rate (6.8%) gives the model more training signal than Sierra Madre (0.9%). Mindanao Bukidnon (8.2%) provides additional positive examples, improving overall calibration.

**Frontier distance dominates:** Proximity to existing deforestation is the strongest predictor, consistent with deforestation literature. Deforestation expands outward from existing cleared areas, following roads and rivers.

**Spatial/neighbor features add signal.** The derived spatial features (neighbor_loss_density, loss_acceleration, fire_hotspot_density) capture the spatial contagion pattern — deforestation spreads to neighboring cells. These are computed at load time from pre-2020 loss data to avoid leakage.

**Protected area importance is low:** This reflects reality in the Philippines — protected area designation doesn't always translate to effective enforcement. The model correctly learns that protection status alone is a weak predictor.

## Statistical Judgment: Why This Model Is an Exploratory Baseline

**This model is not decision-ready.** It is an exploratory baseline that demonstrates the pipeline, not a production classifier. Here's the honest assessment:

**Precision of 28% means 7 in 10 alerts are false positives.** A provincial officer acting on every high-risk cell would waste most of their field time investigating false alarms. This is useful for broad prioritization ("these 3 municipalities deserve attention") but not for site-level operational decisions ("send a team to this exact grid cell").

**Spatial CV F1=0.38 is the honest generalization estimate.** This is substantially below the random-split F1 (~0.55) because geographic autocorrelation inflates the random-split score. The spatial CV number is what the model would deliver on a genuinely new region — and it's weak.

**Per-region variation reveals the real story.** Palawan F1=0.67 is decent because it has 6.8% positive rate and recognizable frontier patterns. Sierra Madre F1=0.41 with 0.9% positives is near-random — the model has almost no signal for intact-forest regions with sparse deforestation. This asymmetry is expected and reveals a fundamental limit of tabular features for this task.

**Why this is still a valid prototype:** The value isn't the model accuracy — it's the pipeline. Real satellite data, explainable predictions (SHAP), temporal holdout evaluation, policy brief synthesis, and an honest calibration view. A stronger model (more data, better features, time-series approaches) would slot directly into this infrastructure.

**What would make it decision-ready:**
- 10x more training data (all Philippine regions, multi-year temporal labels)
- Time-series features (NDVI trajectory, not a single snapshot)
- Real fire data from FIRMS (not spatial proxies)
- Calibrated probability outputs validated against held-out temporal windows
- F1 > 0.6 on spatial CV, precision > 50%

## Improving Performance

### More Data

- **More regions** (Visayas, additional Mindanao provinces) to increase geographic diversity
- **Higher temporal resolution** — year-by-year loss labels instead of period aggregation
- **Real WorldPop raster data** instead of API estimates for population density

### Better Features

- **Soil type, rainfall, and temperature** from climate datasets
- **Land tenure and concession data** from DENR databases
- **Higher-resolution satellite indices** (NDVI, EVI) from Sentinel-2 or Landsat time series
- **Real NASA FIRMS fire data** instead of the current spatial proxy

### Alternative Models

- **XGBoost/LightGBM** — comparison is available in the training UI; typically 1-3% improvement on tabular data
- **Two-stage model:** first classify land cover type, then predict deforestation risk within forest cells only
- **Spatial models:** explicitly model spatial autocorrelation (neighboring cells influence each other)

### Evaluation Improvements

- **Calibration curves** to assess whether predicted probabilities match actual risk rates
- **Temporal cross-validation** — rolling window evaluation across years
- **Threshold optimization** — the UI includes a threshold curve for tuning precision/recall tradeoff
