"""Explorer API routes — map data, cell details, temporal, reports."""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import numpy as np
from pydantic import BaseModel

from core.ml.dataset import load_dataset, prepare_features, FEATURE_COLUMNS, list_regions
from core.ml.training import load_model
from core.ml.explainability import explain_prediction, explain_summary_text
from core.ml.registry import RunRegistry
from core.ml.annotations import AnnotationStore
from core.ml.spatial import cluster_high_risk_cells
from core.geo.lookup import geocode
from core.ai.analysis import generate_policy_brief
from core.ai.brief_cache import load_cached_brief
from core.cv.change_detection import detect_change
from core.cv.validation import validate_exg_against_hansen

try:
    from core.data.fetch_philippines import PROVENANCE
except ImportError:
    PROVENANCE = {}

router = APIRouter(prefix="/api/explorer", tags=["explorer"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "registry.db"
registry = RunRegistry(DB_PATH)
annotation_store = AnnotationStore(DATA_DIR / "annotations.db")


@router.get("/regions")
async def get_regions():
    return list_regions()


@router.get("/geocode")
async def geocode_search(q: str = Query(...)):
    try:
        result = geocode(q)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not result:
        raise HTTPException(status_code=404, detail="Location not found")
    return result


@router.get("/clusters")
async def get_spatial_clusters(
    run_id: str = Query(...),
    region: str = Query(None),
    risk_threshold: float = Query(0.4),
    eps_km: float = Query(15.0),
    min_samples: int = Query(3),
):
    """Identify deforestation fronts using DBSCAN spatial clustering."""
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset(region=region)
    feature_names = run["feature_names"]
    X, _, _ = prepare_features(df, feature_names)

    probabilities = model.predict_proba(X)[:, 1]

    result = cluster_high_risk_cells(
        lats=df["lat"].values,
        lons=df["lon"].values,
        risk_probs=probabilities,
        risk_threshold=risk_threshold,
        eps_km=eps_km,
        min_samples=min_samples,
    )
    result["run_id"] = run_id
    result["region"] = region
    return result


@router.get("/grid")
async def get_grid(
    run_id: str = Query(...),
    region: str = Query(None),
    threshold: float = Query(0.5),
):
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset(region=region)
    feature_names = run["feature_names"]
    X, _, _ = prepare_features(df, feature_names)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    cells = []
    for i in range(len(df)):
        cells.append({
            "lat": float(df.iloc[i]["lat"]),
            "lon": float(df.iloc[i]["lon"]),
            "prediction": int(predictions[i]),
            "risk_probability": round(float(probabilities[i]), 4),
        })

    return {"cells": cells, "run_id": run_id, "region": region, "threshold": threshold}


@router.get("/calibration")
async def get_calibration(
    run_id: str = Query(...),
    region: str = Query(None),
):
    """Get held-out test cells with model prediction vs ground truth.

    Reproduces the exact train/test split from the training run so that
    calibration metrics reflect out-of-sample performance, not in-sample fit.
    """
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    run_region = run["params"].get("region") if run.get("params") else None
    df = load_dataset(region=region or run_region)
    feature_names = run["feature_names"]
    X, y_true, _ = prepare_features(df, feature_names)

    # Use persisted test indices if available (exact reproducibility),
    # otherwise fall back to reproducing the split
    persisted_indices = run.get("test_indices")
    if persisted_indices is not None:
        test_indices = np.array(persisted_indices)
        # Clamp to current dataset size (in case dataset changed)
        test_indices = test_indices[test_indices < len(df)]
    else:
        params = run.get("params", {}) or {}
        test_size = params.get("test_size", 0.2)
        use_spatial = params.get("spatial_split", False)

        if use_spatial:
            block_size = 0.1
            rng = np.random.default_rng(42)
            block_ids = (
                (df["lat"] // block_size).astype(int).astype(str) + "_" +
                (df["lon"] // block_size).astype(int).astype(str)
            )
            unique_blocks = block_ids.unique()
            n_test_blocks = max(1, int(len(unique_blocks) * test_size))
            test_blocks = set(rng.choice(unique_blocks, n_test_blocks, replace=False))
            test_mask = block_ids.isin(test_blocks).values
            test_indices = np.where(test_mask)[0]
        else:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(df))
            _, test_indices, _, _ = train_test_split(
                indices, y_true, test_size=test_size, random_state=42, stratify=y_true
            )

    X_test = X[test_indices]
    y_test = y_true[test_indices]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cells = []
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for i in range(len(test_indices)):
        pred = int(predictions[i])
        actual = int(y_test[i])
        if pred == 1 and actual == 1:
            label = "tp"
        elif pred == 0 and actual == 0:
            label = "tn"
        elif pred == 1 and actual == 0:
            label = "fp"
        else:
            label = "fn"
        counts[label] += 1
        df_idx = int(test_indices[i])
        cells.append({
            "lat": float(df.iloc[df_idx]["lat"]),
            "lon": float(df.iloc[df_idx]["lon"]),
            "prediction": pred,
            "actual": actual,
            "risk_probability": round(float(probabilities[i]), 4),
            "label": label,
        })

    return {
        "cells": cells,
        "counts": counts,
        "run_id": run_id,
        "region": region,
        "split": "spatial" if use_spatial else "random",
        "test_size": test_size,
        "test_cells": len(test_indices),
    }


@router.get("/cell")
async def get_cell_detail(lat: float, lon: float, run_id: str, region: str = None):
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset(region=region)
    feature_names = run["feature_names"]

    distances = ((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2).values
    idx = int(np.argmin(distances))

    X, _, _ = prepare_features(df, feature_names)
    x_single = X[idx:idx+1]

    explanation = explain_prediction(model, x_single, feature_names)
    summary_text = explain_summary_text(explanation, X[idx], feature_names)

    # Get actual model probability (not SHAP-derived)
    risk_probability = float(model.predict_proba(x_single)[0, 1])

    row = df.iloc[idx]
    features = {}
    for col in feature_names:
        prov = PROVENANCE.get(col, {})
        features[col] = {
            "value": round(float(row[col]), 4),
            "source": prov.get("source", ""),
        }

    # Include loss_year if available
    loss_year_val = None
    if "loss_year" in df.columns:
        raw = int(row.get("loss_year", 0))
        loss_year_val = 2000 + raw if raw > 0 else None

    return {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "features": features,
        "explanation": explanation,
        "summary_text": summary_text,
        "loss_year": loss_year_val,
        "risk_probability": round(risk_probability, 4),
    }


@router.get("/temporal")
async def get_temporal_data(
    region: str = Query(...),
    lat_min: float = Query(None),
    lat_max: float = Query(None),
    lon_min: float = Query(None),
    lon_max: float = Query(None),
):
    df = load_dataset(region=region)

    if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
        mask = (
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
        )
        df = df[mask]

    total = len(df)
    if total == 0:
        return {"years": [], "loss_counts": [], "total_cells": 0, "high_risk_cells": 0, "avg_loss_rate_pct": 0.0, "region": region}

    high_risk_count = int(df["high_risk"].sum())
    avg_loss_rate = float(df["annual_loss_rate_pct"].mean())

    # Use real loss_year data if available
    years = list(range(2001, 2023))
    if "loss_year" in df.columns:
        loss_per_year = []
        for y_offset in range(1, 23):  # 1=2001 through 22=2022
            count = int((df["loss_year"] == y_offset).sum())
            loss_per_year.append(count)
    else:
        # Fallback: estimate from aggregate rate
        loss_per_year = []
        for y in years:
            year_factor = 1 + (y - 2001) * 0.02
            estimated = int(avg_loss_rate / 100 * total * year_factor)
            loss_per_year.append(max(0, estimated))

    return {
        "years": years,
        "loss_counts": loss_per_year,
        "total_cells": total,
        "high_risk_cells": high_risk_count,
        "avg_loss_rate_pct": round(avg_loss_rate, 2),
        "region": region,
    }


@router.post("/report")
async def generate_region_report(
    run_id: str,
    region: str = Query(None),
    lat_min: float = Query(None),
    lat_max: float = Query(None),
    lon_min: float = Query(None),
    lon_max: float = Query(None),
):
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset(region=region)
    feature_names = run["feature_names"]

    if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
        mask = (
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
        )
        df = df[mask]

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No data in selected region")

    X, _, _ = prepare_features(df, feature_names)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    high_count = int((predictions == 1).sum())
    low_count = int((predictions == 0).sum())
    total = len(predictions)

    importances = model.feature_importances_
    top_features = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])
    ]

    hotspots = []
    high_risk_indices = np.where(predictions == 1)[0]
    if len(high_risk_indices) > 0:
        sorted_by_prob = high_risk_indices[np.argsort(-probabilities[high_risk_indices])]
        for idx in sorted_by_prob[:5]:
            row = df.iloc[idx]
            hotspots.append({
                "lat": round(float(row["lat"]), 4),
                "lon": round(float(row["lon"]), 4),
                "risk_probability": round(float(probabilities[idx]), 4),
            })

    notable = []
    if len(high_risk_indices) > 0:
        max_prob_idx = high_risk_indices[np.argmax(probabilities[high_risk_indices])]
        row = df.iloc[max_prob_idx]
        notable.append(
            f"Highest risk cell at ({row['lat']:.2f}, {row['lon']:.2f}) "
            f"with {probabilities[max_prob_idx]:.0%} probability"
        )
    if high_count > total * 0.3:
        notable.append(f"Region has elevated risk: {high_count/total:.0%} of cells are high-risk")

    region_name = region.replace("_", " ").title() if region else "Selected Region"
    region_data = {
        "region_name": region_name,
        "run_id": run_id,
        "stats": {
            "total_cells": total,
            "bounds": {
                "lat_min": float(df["lat"].min()),
                "lat_max": float(df["lat"].max()),
                "lon_min": float(df["lon"].min()),
                "lon_max": float(df["lon"].max()),
            },
        },
        "risk_distribution": {
            "high_risk": high_count,
            "low_risk": low_count,
            "high_risk_pct": round(high_count / total * 100, 1),
            "high_risk_hectares": round(high_count * 5.5 * 5.5 * 100, 0),
        },
        "top_features": top_features,
        "notable_points": notable,
        "hotspots": hotspots,
    }

    # Always generate deterministic sections from model
    brief = generate_policy_brief(region_data)

    # If no API key: use cached narrative sections only (executive_summary, recommendations)
    import os
    if not os.environ.get("OPENAI_API_KEY") and region:
        cached = load_cached_brief(region)
        if cached:
            brief["executive_summary"] = cached.get("executive_summary", brief.get("executive_summary", ""))
            brief["recommendations"] = cached.get("recommendations", brief.get("recommendations", []))
            brief["_cached"] = True

    return brief


@router.get("/suggest-annotations")
async def suggest_annotations(
    run_id: str = Query(...),
    region: str = Query(None),
    n: int = Query(10),
):
    """Suggest cells for human review using uncertainty sampling.

    Returns the N cells where the model is least confident (probability
    closest to 0.5), prioritizing cells that haven't been annotated yet.
    These are the most informative cells to label for active learning.
    """
    run = await registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    df = load_dataset(region=region)
    feature_names = run["feature_names"]
    X, _, _ = prepare_features(df, feature_names)

    probabilities = model.predict_proba(X)[:, 1]

    # Uncertainty = distance from 0.5 (lower = more uncertain)
    uncertainty = np.abs(probabilities - 0.5)

    # Get existing annotations to exclude already-reviewed cells
    await annotation_store.init()
    existing = await annotation_store.list_annotations(run_id)
    annotated_coords = set()
    for ann in existing:
        annotated_coords.add((round(ann["lat"], 3), round(ann["lon"], 3)))

    # Rank by uncertainty, exclude annotated
    candidates = []
    for i in np.argsort(uncertainty):
        lat = float(df.iloc[i]["lat"])
        lon = float(df.iloc[i]["lon"])
        if (round(lat, 3), round(lon, 3)) in annotated_coords:
            continue
        candidates.append({
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "risk_probability": round(float(probabilities[i]), 4),
            "uncertainty": round(float(uncertainty[i]), 4),
            "prediction": int(probabilities[i] > 0.5),
        })
        if len(candidates) >= n:
            break

    return {
        "suggestions": candidates,
        "total_unannotated": len(df) - len(annotated_coords),
        "total_annotated": len(annotated_coords),
        "method": "uncertainty_sampling",
    }


@router.post("/annotate")
async def annotate_cell(
    lat: float,
    lon: float,
    run_id: str,
    prediction: int,
    risk_probability: float,
    verdict: str = Query(..., regex="^(accept|reject)$"),
    note: str = Query(""),
):
    """Record an official's accept/reject verdict on a model prediction."""
    await annotation_store.init()
    annotation_id = await annotation_store.save_annotation(
        lat=lat, lon=lon, run_id=run_id,
        prediction=prediction, risk_probability=risk_probability,
        verdict=verdict, note=note,
    )
    return {"id": annotation_id, "verdict": verdict}


@router.get("/annotations")
async def list_annotations(run_id: str = Query(None)):
    """List all annotations, optionally filtered by run."""
    await annotation_store.init()
    return await annotation_store.list_annotations(run_id)


@router.get("/annotations/stats")
async def annotation_stats(run_id: str = Query(...)):
    """Get annotation counts (accepted/rejected) for a run."""
    await annotation_store.init()
    return await annotation_store.get_stats(run_id)


@router.get("/annotations/cell")
async def get_cell_annotation(lat: float, lon: float, run_id: str):
    """Get the latest annotation for a specific cell."""
    await annotation_store.init()
    result = await annotation_store.get_annotation_for_cell(lat, lon, run_id)
    return result or {}


@router.post("/change-detection")
async def run_change_detection(
    lat_min: float = Query(...),
    lat_max: float = Query(...),
    lon_min: float = Query(...),
    lon_max: float = Query(...),
    year_before: int = Query(2018),
    year_after: int = Query(2023),
):
    """Run satellite image change detection on a bounding box.

    Downloads Sentinel-2 tiles for two years, computes vegetation index change,
    and classifies pixels as forest loss, gain, or stable.
    """
    if year_after <= year_before:
        raise HTTPException(
            status_code=422,
            detail=f"year_after ({year_after}) must be greater than year_before ({year_before})"
        )
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ThreadPoolExecutor(1),
        detect_change,
        lat_min, lat_max, lon_min, lon_max,
        year_before, year_after,
    )
    return result


@router.post("/validate-cv")
async def validate_change_detection(
    lat_min: float = Query(...),
    lat_max: float = Query(...),
    lon_min: float = Query(...),
    lon_max: float = Query(...),
    year_before: int = Query(2018),
    year_after: int = Query(2022),
    sample_points: int = Query(50),
):
    """Validate CV change detection against Hansen ground truth.

    Samples random points, compares ExG vegetation change with known
    deforestation to measure detection accuracy.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ThreadPoolExecutor(1),
        validate_exg_against_hansen,
        lat_min, lat_max, lon_min, lon_max,
        year_before, year_after, sample_points,
    )
    return result


# --- Multi-site review summary ---

class SiteBounds(BaseModel):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class ReviewSummaryRequest(BaseModel):
    run_id: str
    region: str | None = None
    sites: list[SiteBounds]


@router.post("/review-summary")
async def generate_review_summary(req: ReviewSummaryRequest):
    """Generate an aggregated review summary across multiple selected sites."""
    run = await registry.get_run(req.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    model_path = run["model_path"]
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    model = load_model(Path(model_path))
    full_df = load_dataset(region=req.region)
    feature_names = run["feature_names"]

    sites_analysis = []
    total_cells = 0
    total_high_risk = 0

    for site in req.sites:
        mask = (
            (full_df["lat"] >= site.lat_min) & (full_df["lat"] <= site.lat_max) &
            (full_df["lon"] >= site.lon_min) & (full_df["lon"] <= site.lon_max)
        )
        df = full_df[mask]
        if len(df) == 0:
            sites_analysis.append({
                "name": site.name,
                "bounds": site.model_dump(exclude={"name"}),
                "cell_count": 0,
                "high_risk": 0,
                "high_risk_pct": 0,
                "top_hotspot": None,
                "primary_driver": None,
            })
            continue

        X, _, _ = prepare_features(df, feature_names)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        high_count = int((predictions == 1).sum())
        total_cells += len(df)
        total_high_risk += high_count

        # Find top hotspot
        top_hotspot = None
        high_indices = np.where(predictions == 1)[0]
        if len(high_indices) > 0:
            max_idx = high_indices[np.argmax(probabilities[high_indices])]
            row = df.iloc[max_idx]
            top_hotspot = {
                "lat": round(float(row["lat"]), 4),
                "lon": round(float(row["lon"]), 4),
                "risk_probability": round(float(probabilities[max_idx]), 4),
            }

        # Per-site driver from SHAP (site-specific, not global)
        sample_size = min(10, len(df))
        sample_indices = np.random.default_rng(42).choice(len(df), sample_size, replace=False)

        site_shap = np.zeros(len(feature_names))
        for idx in sample_indices:
            x_single = X[idx:idx+1]
            explanation = explain_prediction(model, x_single, feature_names)
            for sv in explanation["shap_values"]:
                feat_idx = feature_names.index(sv["feature"]) if sv["feature"] in feature_names else -1
                if feat_idx >= 0:
                    site_shap[feat_idx] += abs(sv["shap_value"])

        if site_shap.sum() > 0:
            site_shap /= site_shap.sum()

        top_driver_idx = np.argmax(site_shap)
        primary_driver = feature_names[top_driver_idx]
        primary_driver_importance = round(float(site_shap[top_driver_idx]), 4)

        sites_analysis.append({
            "name": site.name,
            "bounds": site.model_dump(exclude={"name"}),
            "cell_count": len(df),
            "high_risk": high_count,
            "high_risk_pct": round(high_count / len(df) * 100, 1) if len(df) > 0 else 0,
            "top_hotspot": top_hotspot,
            "primary_driver": primary_driver,
            "primary_driver_importance": primary_driver_importance,
        })

    # Aggregate summary
    import datetime
    summary = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_run_id": req.run_id,
        "region": req.region,
        "total_sites": len(req.sites),
        "total_cells_analyzed": total_cells,
        "total_high_risk": total_high_risk,
        "overall_risk_pct": round(total_high_risk / total_cells * 100, 1) if total_cells > 0 else 0,
        "sites": sites_analysis,
        "priority_sites": sorted(
            [s for s in sites_analysis if s["high_risk"] > 0],
            key=lambda s: -s["high_risk_pct"],
        ),
    }

    return summary
