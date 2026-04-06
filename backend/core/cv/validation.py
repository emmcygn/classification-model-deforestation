"""Validate CV change detection against Hansen ground truth.

Computes correlation between ExG vegetation change and Hansen-confirmed
forest loss to prove the computer vision signal is real.
"""

import math
import numpy as np
import logging
from core.cv.change_detection import compute_exg, _fetch_s2_tile, _tile_coords, ANALYSIS_ZOOM
from core.data.sources.hansen import fetch_all_hansen

logger = logging.getLogger(__name__)


def validate_exg_against_hansen(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    year_before: int = 2018, year_after: int = 2022,
    sample_points: int = 100,
) -> dict:
    """Validate ExG change detection against Hansen ground truth.

    Samples random points in the bounding box, computes ExG change,
    and checks correlation with Hansen loss data.

    Returns:
        Dict with correlation metrics, confusion matrix, and validation summary.
    """
    rng = np.random.default_rng(42)

    # Sample random points
    lats = rng.uniform(lat_min, lat_max, sample_points)
    lons = rng.uniform(lon_min, lon_max, sample_points)

    exg_changes = []
    hansen_losses = []
    valid_points = 0

    for lat, lon in zip(lats, lons):
        # Get Hansen ground truth
        hansen = fetch_all_hansen(float(lat), float(lon))
        if not hansen["is_land"]:
            continue

        # Get ExG from Sentinel-2 tiles
        zoom = ANALYSIS_ZOOM
        tx, ty = _tile_coords(float(lat), float(lon), zoom)

        before = _fetch_s2_tile(year_before, zoom, tx, ty)
        after = _fetch_s2_tile(year_after, zoom, tx, ty)

        if before is None or after is None:
            continue

        exg_before = compute_exg(before)
        exg_after = compute_exg(after)

        # Get pixel position within tile
        n = 2 ** zoom
        x_float = (float(lon) + 180) / 360 * n
        y_float = (1 - math.log(math.tan(math.radians(float(lat))) + 1 / math.cos(math.radians(float(lat)))) / math.pi) / 2 * n
        px = min(int((x_float - int(x_float)) * 256), 255)
        py = min(int((y_float - int(y_float)) * 256), 255)

        exg_change = float(exg_after[py, px] - exg_before[py, px])
        has_loss = hansen["has_loss"]

        exg_changes.append(exg_change)
        hansen_losses.append(1 if has_loss else 0)
        valid_points += 1

    if valid_points < 10:
        return {
            "valid": False,
            "reason": f"Only {valid_points} valid land points found",
            "valid_points": valid_points,
        }

    exg_arr = np.array(exg_changes)
    hansen_arr = np.array(hansen_losses)

    # Compute correlation
    if hansen_arr.std() > 0 and exg_arr.std() > 0:
        correlation = float(np.corrcoef(exg_arr, hansen_arr)[0, 1])
    else:
        correlation = 0.0

    # Classify ExG changes using a threshold
    # Negative ExG change = vegetation loss
    exg_predicted_loss = (exg_arr < -0.05).astype(int)

    # Confusion matrix
    tp = int(((exg_predicted_loss == 1) & (hansen_arr == 1)).sum())
    fp = int(((exg_predicted_loss == 1) & (hansen_arr == 0)).sum())
    fn = int(((exg_predicted_loss == 0) & (hansen_arr == 1)).sum())
    tn = int(((exg_predicted_loss == 0) & (hansen_arr == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    # Mean ExG change for loss vs no-loss cells
    loss_mask = hansen_arr == 1
    no_loss_mask = hansen_arr == 0
    mean_exg_loss = float(exg_arr[loss_mask].mean()) if loss_mask.any() else 0
    mean_exg_no_loss = float(exg_arr[no_loss_mask].mean()) if no_loss_mask.any() else 0

    return {
        "valid": True,
        "valid_points": valid_points,
        "hansen_loss_count": int(hansen_arr.sum()),
        "hansen_no_loss_count": int((1 - hansen_arr).sum()),
        "correlation": round(correlation, 4),
        "mean_exg_change_loss_cells": round(mean_exg_loss, 4),
        "mean_exg_change_no_loss_cells": round(mean_exg_no_loss, 4),
        "signal_separation": round(abs(mean_exg_loss - mean_exg_no_loss), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "interpretation": (
            f"ExG change shows {'meaningful' if abs(correlation) > 0.1 else 'weak'} correlation "
            f"({correlation:.3f}) with Hansen ground truth. "
            f"Mean ExG change for loss cells: {mean_exg_loss:.4f} vs no-loss: {mean_exg_no_loss:.4f}. "
            f"{'Signal is detectable.' if abs(mean_exg_loss - mean_exg_no_loss) > 0.02 else 'Signal is marginal — RGB-based detection has limitations vs multispectral NDVI.'}"
        ),
        "year_before": year_before,
        "year_after": year_after,
    }
