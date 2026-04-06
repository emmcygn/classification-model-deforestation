# DeforestAI Philippines — Design Spec

## Problem

The Philippines loses ~47,000 hectares of forest annually. Provincial environment officers lack accessible, data-driven tools to identify where deforestation risk is highest, what's driving it, and where to focus monitoring resources. Existing satellite data (Hansen GFW) is raw and uninterpreted — it requires GIS expertise to extract actionable intelligence.

DeforestAI Philippines bridges this gap: real geospatial data, ML-based risk prediction, SHAP-driven explainability, and AI-generated policy briefs — packaged for officials who need to make decisions, not run Python scripts.

## Target Users

- Philippine provincial/municipal environment officers (DENR, PENRO, CENRO)
- Conservation NGOs operating in Palawan and Sierra Madre
- Mozaic Earth evaluators (tech test context)

## Focus Regions

1. **Palawan** — UNESCO Man and Biosphere Reserve, under mining/palm oil pressure. Bounding box: ~8.3°N to 12.5°N, ~117.0°E to 120.5°E.
2. **Sierra Madre corridor (Aurora/Quirino/Isabela)** — Largest remaining Philippine rainforest, illegal logging pressure. Bounding box: ~15.5°N to 17.5°N, ~121.0°E to 122.5°E.

Grid resolution: 0.02° (~2.2km). Ocean cells are filtered out by the pipeline (Hansen tiles return no data over water). Expected ~3,000-4,000 land cells for Palawan and ~2,000-3,000 for Sierra Madre (~5,000-7,000 total). Higher fidelity than the Rondonia prototype (0.05°) while staying performant for frontend rendering.

## Architecture

### Backend (FastAPI + Python)

```
backend/
  core/
    data/
      fetch_philippines.py    # One-shot data pipeline — fetches real data, produces CSVs
      live_pipeline.py        # Same logic, callable on-demand for arbitrary bounds (stubbed but turnkey)
      sources/
        hansen.py             # Hansen GFW tree cover + loss extraction from tiles
        srtm.py               # SRTM elevation + slope from OpenTopography or USGS
        osm.py                # Road network distances via Overpass API
        worldpop.py           # Population density raster sampling
        wdpa.py               # Protected area intersection
    ml/                       # Unchanged — dataset, training, evaluation, explainability, registry
    ai/
      analysis.py             # Refactored: GPT-4o-mini, structured policy brief output
      brief_cache.py          # Load/serve cached briefs when no API key
    geo/
      lookup.py               # Unchanged — Nominatim geocoding
  api/routes/
    pipeline.py               # Unchanged
    explorer.py               # Updated: site-level endpoints, temporal data, provenance metadata
    export.py                 # NEW: PDF export of policy briefs
  data/
    raw/
      palawan_grid.csv        # Pre-baked real data
      sierra_madre_grid.csv   # Pre-baked real data
    cache/
      palawan_brief.json      # Cached policy brief for keyless demo
      sierra_madre_brief.json # Cached policy brief for keyless demo
```

### Frontend (React + Vite + TypeScript)

```
frontend/src/
  pages/
    Explorer.tsx              # Updated: region selector, site-level mode, temporal panel
    Pipeline.tsx              # Unchanged
  components/
    map/
      RiskMap.tsx             # Updated: site selection (draw rectangle), region switch
      CellDetailPanel.tsx     # Updated: data provenance per feature
      TemporalPanel.tsx       # NEW: year-over-year forest loss chart for selected site
    pipeline/                 # Unchanged
    report/
      PolicyBrief.tsx         # NEW: structured policy brief display (replaces simple report)
      ExportButton.tsx        # NEW: download PDF button
  lib/
    api.ts                    # Updated: new endpoints for temporal, export, regions
```

## Data Pipeline

### Pre-baked Pipeline (`fetch_philippines.py`)

Runs once locally. For each region:

1. Generate grid of lat/lon points at 0.02° resolution within bounding box
2. For each grid cell, fetch:
   - **Tree cover 2000 baseline** — Sample Hansen GFW `treecover2000` tiles. Extract green channel intensity as % cover.
   - **Annual loss rate** — Sample Hansen `loss_year` tiles across years 2001-2022. Count loss pixels per cell, compute annual rate.
   - **Elevation + slope** — Query SRTM data via OpenTopography API (free, no key for small requests) or fallback to pre-downloaded GeoTIFF. Compute slope from elevation gradient.
   - **Distance to nearest road** — Query OSM Overpass API for highway features within region bbox. For each cell, compute distance to nearest road segment.
   - **Protected area status** — Download WDPA shapefile for Philippines (free from protectedplanet.net). Point-in-polygon test for each cell.
   - **Population density** — Sample WorldPop Philippines raster (free download, ~1km resolution).
   - **Distance to deforestation frontier** — Derived: for each cell, distance to nearest cell with loss_year > 0 in recent 5 years.

Output: CSV with same schema as current synthetic data (8 features + lat/lon + high_risk target).

**high_risk target derivation:** Binary label — 1 if the cell experienced tree cover loss in the last 3 years of Hansen data (2020-2022), 0 otherwise. This is ground truth from real observations, not synthetic.

### Live Pipeline (`live_pipeline.py`)

Same logic as above, packaged as an async function:
```python
async def fetch_region(lat_min, lat_max, lon_min, lon_max, step=0.02) -> pd.DataFrame
```

Stubbed with clear interface. Each data source module (`hansen.py`, `srtm.py`, etc.) has both a `fetch()` and a `load_cached()` path. The live pipeline calls `fetch()`, the pre-baked script calls `fetch()` once and saves the result.

### Data Source Modules

Each module in `core/data/sources/` follows a common interface:

```python
def fetch(lat: float, lon: float) -> float | int:
    """Fetch value for a single grid cell from the live API."""

def fetch_bulk(grid: pd.DataFrame) -> pd.Series:
    """Fetch values for all grid cells. Uses bulk APIs where available."""
```

This makes each source independently testable and swappable.

### Fallback Strategy

Some APIs may be slow or rate-limited. For each source:
- **Hansen tiles**: Always available (Google Cloud Storage, no auth). Primary source.
- **SRTM/OpenTopography**: Free tier exists. Fallback: use `elevation` package (Python) which caches SRTM data locally.
- **OSM Overpass**: Free, but rate-limited. Fetch entire region road network once, compute all distances locally.
- **WorldPop**: Direct GeoTIFF download. No API needed — download once, sample locally.
- **WDPA**: Shapefile download. No API needed.

Net: only Hansen tiles and Overpass are network calls per cell. Everything else is bulk download + local processing.

## Model

Unchanged from current architecture:
- scikit-learn Random Forest with `class_weight="balanced"`
- 8 features, binary classification (high_risk)
- SHAP TreeExplainer for per-cell prediction explanations
- SQLite registry for run history

The model trains on real Philippine data. Since `high_risk` is derived from actual Hansen loss observations (not synthetic), the model learns genuine deforestation patterns.

## AI Integration

### GPT-4o-mini Policy Brief

Swap `anthropic` SDK for `openai` SDK. Model: `gpt-4o-mini`.

**Policy brief structure** (deterministic sections + AI narrative):

```json
{
  "executive_summary": "AI-generated 2-3 sentence overview",
  "site_overview": {
    "name": "Palawan Province",
    "total_cells": 5200,
    "area_km2": 6292,
    "bounds": {...}
  },
  "risk_assessment": {
    "high_risk_cells": 1240,
    "high_risk_pct": 23.8,
    "high_risk_hectares": 15004
  },
  "hotspots": [
    {
      "lat": 9.82, "lon": 118.73,
      "risk_probability": 0.94,
      "primary_driver": "road proximity",
      "recommended_action": "Increase patrol frequency"
    }
  ],
  "top_drivers": [
    {"feature": "dist_to_road_km", "importance": 0.23, "interpretation": "AI-generated explanation"}
  ],
  "temporal_trend": {
    "years": [2015, 2016, ..., 2022],
    "annual_loss_hectares": [320, 410, ..., 580],
    "trend_direction": "increasing",
    "ai_analysis": "AI-generated trend interpretation"
  },
  "recommendations": "AI-generated 3-5 bullet policy recommendations",
  "data_provenance": [
    {"feature": "tree_cover_2000_pct", "source": "Hansen/UMD GFW v1.7", "url": "https://glad.umd.edu/dataset/gfw"},
    {"feature": "elevation_m", "source": "SRTM 30m (USGS/NASA)", "url": "https://www.usgs.gov/centers/eros"},
    ...
  ],
  "generated_at": "2026-04-05T12:00:00Z",
  "model_run_id": "abc123"
}
```

All sections except `executive_summary`, `recommendations`, and `ai_analysis` fields are deterministic. AI adds synthesis where human interpretation is needed.

**Prompt design:** The prompt explicitly instructs the model to cite numbers from the data, avoid adding information not present, and frame recommendations in terms of Philippine regulatory context (DENR, NIPAS Act, EO 23 logging moratorium).

### Cached Briefs

Pre-generated briefs for Palawan and Sierra Madre saved as JSON in `backend/data/cache/`. Served when `OPENAI_API_KEY` is not set. Displayed with a "(cached example)" badge in the UI.

## Frontend Changes

### Region Selector

Top of Explorer sidebar: dropdown with "Palawan" and "Sierra Madre". Switching loads the corresponding grid CSV and recenters the map.

### Site-Level Mode

User can draw a rectangle on the map (Leaflet draw plugin or click-drag box select). When a site is selected:
- Grid cells within the selection are highlighted
- Stats in the sidebar scope to the selection
- Policy brief generates for the selected area only
- Temporal chart shows loss history for the selected area

### Temporal Panel

New component below the cell detail panel in the sidebar (or as a collapsible section). Shows:
- Line chart: annual forest loss (hectares) per year, 2001-2022
- Area chart: cumulative loss over time
- Data from Hansen loss_year tiles aggregated per grid cell

Scoped to either the full region or the user's site selection.

### Policy Brief Display

Replaces the current simple report overlay. Structured sections matching the JSON schema above. Clean, professional typography. Expandable sections for detail.

### Data Provenance

In the cell detail panel, each feature value shows a small source label:
```
Elevation: 342m  [SRTM/USGS]
Dist to road: 2.3km  [OpenStreetMap]
Tree cover 2000: 87%  [Hansen/GFW]
```

In the policy brief, a "Data Sources" section at the bottom lists all sources with links.

### PDF Export

Button in the policy brief panel: "Export as PDF". Uses browser print-to-PDF via a print-optimized CSS stylesheet on a dedicated route (`/brief/:runId/print`), or `html2canvas` + `jspdf` for client-side generation. The PDF should look like a professional document an official would forward.

## What Stays the Same

- All ML modules (training, evaluation, explainability, registry)
- Pipeline page (Dataset/Training/Evaluation tabs) — now shows real Philippine data
- Backend test suite (tests remain valid, data schema unchanged)
- Hansen GFW satellite overlays on the map
- Leaflet basemap toggle (street/satellite)

## README Framing

The README should be rewritten to frame the project in Mozaic Earth's language:

- "Site-level nature intelligence" not just "deforestation tool"
- Emphasize: real data, auditability, decision-ready outputs
- Connect to DENR regulatory context
- Reflections section should mention: TNFD alignment potential, mobile data capture extension (Mozaic's Guardian model), integration with biodiversity metrics beyond just tree cover

## Dependencies to Add

**Backend:**
- `openai` (replaces `anthropic`)
- `elevation` or `rasterio` (SRTM processing)
- `shapely` (WDPA point-in-polygon)
- `requests` (Overpass API)
- `Pillow` (Hansen tile pixel sampling)

**Frontend:**
- `leaflet-draw` or equivalent for rectangle selection
- `jspdf` + `html2canvas` (PDF export)

## Non-Goals

- Mobile app / field data capture (Mozaic's Guardian model — mention in Reflections only)
- BNG / DEFRA Biodiversity Metric alignment (UK-specific)
- User authentication
- Multi-tenant architecture
- Real-time data ingestion
