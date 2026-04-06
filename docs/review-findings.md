# Code Review Findings & Fixes

This document records critical findings from adversarial code review and the fixes applied. Transparency about what was wrong and how it was fixed is part of the engineering story.

## Finding 1 (CRITICAL): Target Leakage — Model Was Re-encoding Known Loss

**The problem:** The model claimed to predict "future deforestation risk" but features were derived from the same loss data used as the target label. The model was looking up the answer, not predicting it.

**Specific leakage paths:**
- `high_risk` target included loss from the same period as feature computation (2018-2022)
- `dist_to_deforestation_frontier_km` was computed from ALL loss cells, including the cell being predicted
- `annual_loss_rate_pct` was derived from `has_loss`, which is a near-direct encoding of the label
- `tree_cover_2000_pct` was adjusted downward when `has_loss=True`, creating circular dependency
- Spatial features (`neighbor_loss_count/density`) used `high_risk` from the full dataset before train/test split, leaking test labels into train features

**The fix — temporal holdout:**
- **Features** now use only 2001-2019 data (`loss_year` 1-19)
- **Target** is loss in 2020-2022 only (`loss_year` 20-22)
- `dist_to_deforestation_frontier_km` uses only pre-2020 loss cells
- `annual_loss_rate_pct` computed from pre-2020 loss only
- Tree cover estimation no longer depends on `has_loss`
- Spatial features use `pre_period_loss` (2001-2019) instead of `high_risk`
- `_apply_temporal_split()` in dataset.py recomputes labels from `loss_year` at load time

**Impact on metrics:**
- Before fix (leaked): Spatial CV F1=0.48, Precision=42%
- After fix (honest): Spatial CV F1=0.38, Precision=28%, Recall=63%
- Per-region (post-fix): Palawan F1=0.67, Sierra Madre F1=0.41, Mindanao F1=0.52
- The model now genuinely predicts future risk from past observations
- Trained on 3 regions (7,669 cells: Palawan 2,300 + Sierra Madre 3,378 + Mindanao Bukidnon 1,991)

**Files changed:** `fetch_philippines.py`, `hansen.py`, `feature_engineering.py`, `dataset.py`

---

## Finding 2 (HIGH): Cached Briefs Ignored Model Context

**The problem:** When `OPENAI_API_KEY` was absent, the `/report` endpoint returned an entirely cached brief with hardcoded numbers (cell counts, hotspot coordinates, risk percentages). If the user trained a different model or selected a bounding box, the output was stale.

**The fix:** Deterministic data sections (risk_assessment, hotspots, top_drivers, notable_findings) are ALWAYS computed from the actual model. Only narrative prose (executive_summary, recommendations) uses cached text, flagged with `_cached: true`.

**File changed:** `explorer.py` `/report` endpoint

---

## Finding 3 (HIGH): Cell Detail Could Return Wrong Region's Cell

**The problem:** `/cell` endpoint loaded the combined multi-region dataset and found the nearest cell by lat/lon distance. In a multi-region setup, this could silently return a cell from the wrong region with wrong provenance and wrong SHAP explanation.

**The fix:** Added `region` parameter to `/cell` endpoint and frontend `getCellDetail()`. The endpoint now loads only the requested region's data.

**Files changed:** `explorer.py`, `api.ts`, `Explorer.tsx`

---

## Finding 4 (HIGH): "Real Data" Claim Was Materially False for Tree Cover

**The problem:** README claimed "Real data, not synthetic" but `tree_cover_2000_pct` was generated from a seeded random distribution (45-95%) and further adjusted based on `has_loss`.

**The fix:** 
1. Removed `has_loss` adjustment from tree cover estimation (eliminates circular dependency)
2. Updated README to honestly state: "Tree cover percentage is estimated from GFW forest presence data (exact % not available via tile API)"
3. GFW confirms forest presence (>30% canopy) — that IS real data. The percentage within that range is estimated.

**File changed:** `hansen.py`, `README.md`

---

## Finding 5 (MEDIUM): Retrain Baseline Comparison Was Invalid

**The problem:** 
1. Annotations were loaded globally, not filtered by run or region
2. Baseline model used a potentially different split method than the retrained model, making delta comparisons meaningless

**The fix:**
1. Annotations filtered to cells matching the selected region
2. Both baseline and retrained models use the same split method (spatial or random) and same random state

**File changed:** `pipeline.py` `/retrain` endpoint

---

## Finding 6 (MEDIUM): Per-Site "Primary Driver" Was Global, Not Site-Specific

**The problem:** Review summary used `model.feature_importances_` (global) and assigned the same top feature to every site. This presented model-wide importance as site-specific insight.

**The fix:** Per-site SHAP analysis — sample up to 10 cells per site, compute mean absolute SHAP values, use the site-specific top contributor as primary driver.

**File changed:** `explorer.py` `/review-summary` endpoint

---

## Finding 7 (HIGH): Calibration View Showed In-Sample Fit, Not Real Performance

**The problem:** The `/calibration` endpoint loaded the full dataset and ran `model.predict()` on it — the same data the model was trained on. The sidebar displayed accuracy from this as if it were evidence of real-world quality. This is materially misleading.

**The fix:** The endpoint now reproduces the exact train/test split from the training run (using the stored `test_size`, `spatial_split`, and deterministic `random_state=42`) and evaluates only on the held-out test set. The frontend labels it "Held-out Test Set Performance" with "Out-of-sample" qualifier.

**Files changed:** `explorer.py` `/calibration` endpoint, `Explorer.tsx`

---

## Finding 8 (HIGH): Annotation Stored Fabricated Probability From SHAP Arithmetic

**The problem:** `CellDetailPanel.tsx` computed `risk_probability` as `abs(base_value + sum(shap_values))` — this is not a calibrated probability, not bounded [0,1], and not the same value shown on the map. The annotation endpoint persisted this as if it were an actual model score, poisoning the audit trail.

**The fix:** The `/cell` endpoint now returns the actual `model.predict_proba()` value as `risk_probability`. The frontend passes this directly to the annotation endpoint instead of computing from SHAP.

**Files changed:** `explorer.py` `/cell` endpoint, `api.ts` CellDetail type, `CellDetailPanel.tsx`

---

## Finding 9 (MEDIUM): Map Grid Rectangles Were 2.5x Too Large

**The problem:** `RiskMap.tsx` hardcoded `step = 0.05` for grid rectangle rendering, but the actual data is at 0.02° spacing. This made hotspots look 2.5x larger and more contiguous than the data supports.

**The fix:** Changed `step` to `0.02` to match the actual data resolution.

**File changed:** `RiskMap.tsx`

---

## Finding 10 (MEDIUM): CV Change Detection Accepted Backwards Date Ranges

**The problem:** The frontend allowed selecting 2021→2020 (backwards). The backend accepted it without validation. This could produce "vegetation loss" from a temporally inverted comparison.

**The fix:** Frontend now constrains the "To" dropdown to only show years after "From". Backend validates `year_after > year_before` and returns 422 if violated.

**Files changed:** `ChangeDetection.tsx`, `explorer.py` `/change-detection` endpoint

---

## Finding 11 (MEDIUM): Hectare Estimate Had False Precision

**The problem:** `change_detection.py` converted pixels to hectares using a fixed 38m×38m assumption from zoom level 12 at the equator. Actual ground area varies with latitude, and the pipeline operates on visualization tiles, not analysis-ready bands.

**The fix:** Added explicit methodology caveat in both the API response `method` field and the frontend display: "Approximate — derived from visualization tiles, not analysis-ready data."

**Files changed:** `change_detection.py`, `ChangeDetection.tsx`

---

## Finding 12 (MEDIUM): Geocoding Failures Surfaced as 500s

**The problem:** `lookup.py` called Nominatim with no timeout, retry, or exception handling. If the service rate-limited, timed out, or rejected the user agent, the route raised an unhandled exception → HTTP 500.

**The fix:** Added 5-second timeout to Nominatim client. Geocode function now catches `GeocoderTimedOut`, `GeocoderServiceError`, `GeocoderUnavailable` and raises descriptive `RuntimeError`. The route catches this and returns HTTP 502 with a clear message.

**Files changed:** `lookup.py`, `explorer.py` `/geocode` endpoint

---

## Finding 13 (HIGH): "Change Verdict" Appended Duplicate Annotations

**The problem:** Clicking "Change verdict" in CellDetailPanel cleared local state, but the backend's `save_annotation` blindly inserted a new row. One reviewer flipping accept→reject would show up as both accepted AND rejected in stats. Retraining used the full annotation history, not the latest verdict per cell — poisoning the human-in-the-loop pipeline.

**The fix:** `annotations.py` now upserts: checks for an existing annotation at the same (lat, lon, run_id) and updates it instead of inserting a duplicate. Stats and retraining now see exactly one verdict per cell. Added `updated_at` column and unique index on `(round(lat,3), round(lon,3), run_id)`.

**File changed:** `annotations.py`

---

## Finding 14 (MEDIUM): Calibration Map Misplaced Test Points via Feature-Vector Matching

**The problem:** The calibration endpoint reconstructed held-out cell locations by nearest feature-vector distance. If two cells had near-identical features, multiple test rows could collapse onto the same map location.

**The fix:** For random splits, split the df index array directly with the same `random_state=42` to get exact row indices. For spatial splits, reproduce the block assignment logic to get a boolean mask over original df indices. No feature-vector matching needed.

**Files changed:** `explorer.py` `/calibration` endpoint

---

## Finding 15 (MEDIUM): Cached-Narrative Flag Mismatch Between API and UI

**The problem:** Backend set `_cached_narrative: true`, but the frontend type expected `_cached` and the PolicyBrief component checked `_cached`. Cached briefs never showed the "cached example" badge.

**The fix:** Unified on `_cached` throughout — backend, frontend type, docs.

**Files changed:** `explorer.py`, `api.ts`, all docs referencing the flag

---

## Finding 16 (MEDIUM): Architecture Doc Drift

**The problem:** `docs/architecture.md` documented `/spatial-clusters` but the actual route is `/clusters`. The cached brief flag was documented as `_cached_narrative` but is now `_cached`.

**The fix:** Corrected route path and flag name in architecture.md, methodology.md, and review-findings.md.

**Files changed:** `architecture.md`, `methodology.md`, `review-findings.md`

---

## Finding 17 (MEDIUM): Test Suite Required Network Access

**The problem:** `test_change_detection.py` called `detect_change()` against live Sentinel-2 tile servers. Tests failed without network access, making CI unreliable.

**The fix:** Marked the integration test with `@pytest.mark.network`. Added `pytest.ini` with `addopts = -m "not network"` to exclude by default. Test catches network errors and skips gracefully. Run explicitly with `pytest -m network`.

**Files changed:** `test_change_detection.py`, `pytest.ini` (new)

---

## Finding 18 (HIGH): ExG Feature Had Temporal Leakage

**The problem:** `exg_change_2018_2022` compared 2018 to 2022 Sentinel-2 imagery. Since the target is loss in 2020-2022, the 2022 image observes vegetation state *after* target-period loss occurred. Correlation with target: r=-0.184 (p<1e-18). The previous review rounds caught other leakage paths but missed this one.

**The fix:** Changed to `exg_change_2018_2019` — both years are within the feature period (pre-2020). Regenerated all 3 region CSVs with the corrected feature. Updated `fetch_philippines.py`, `dataset.py`, and all docs.

**Files changed:** `fix_exg_column.py` (new), `fetch_philippines.py`, `dataset.py`, all CSVs, `methodology.md`, `model-performance.md`

---

## Finding 19 (MEDIUM): Model Performance Overclaimed as Decision-Ready

**The problem:** README described the tool as "decision-ready intelligence" and "packaged for officials who need to make decisions." With spatial CV F1=0.38 and precision=28%, the model produces ~7 false positives for every 3 true alerts. This is an exploratory baseline, not operational intelligence.

**The fix:** Reframed README to "exploratory prototype" with honest metrics in the tagline. Added "Statistical Judgment" section to model-performance.md explaining exactly why the metrics are weak and what "decision-ready" would require. Added "Target Formulation Limitations" to methodology.md.

**Files changed:** `README.md`, `model-performance.md`, `methodology.md`

---

## Finding 20 (MEDIUM): Test Suite Had No Leakage Guards

**The problem:** Tests only checked output shapes and key presence. No tests verified temporal separation, feature-target independence, or cross-region generalization. For a model-heavy submission, this was a credibility gap.

**The fix:** Added `test_temporal_integrity.py` with tests for: temporal split correctness, feature-target correlation bounds, ExG period validation, neighbor feature provenance, cross-region generalization. Extended `test_training.py` with metric range and confusion matrix assertions.

**Files changed:** `test_temporal_integrity.py` (new), `test_training.py`

---

## Lessons Learned

1. **Target leakage is easy to introduce incrementally.** Each feature was added independently and looked reasonable in isolation. The leakage emerged from the interaction between features and the target derivation.

2. **Temporal holdout should be designed first, not retrofitted.** The correct framing (features from 2001-2019, target from 2020-2022) should have been established before any feature engineering.

3. **Cached outputs must be clearly scoped.** Caching AI-generated prose is fine; caching data-derived numbers is misleading.

4. **Honest documentation of limitations builds more trust than inflated metrics.** Spatial CV F1=0.38 is modest but defensible. The leaked F1=0.48 was neither.

5. **In-sample metrics presented as calibration are worse than no metrics.** A calibration view that evaluates on training data actively misleads the user into trusting the model more than warranted.

6. **Every number persisted in an audit trail must come from the model, not from post-hoc arithmetic.** SHAP values are explanatory, not probabilistic — using them as a probability substitute poisons downstream workflows.

7. **UI constants (grid resolution, pixel sizes) must match the data pipeline.** A 2.5x mismatch in grid rendering makes visual analysis unreliable and undermines the "decision-ready" claim.
