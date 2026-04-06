"""Fix ExG temporal leak: recompute exg_change using 2018->2019 (pre-target period).

The original exg_change_2018_2022 compared 2018 to 2022 imagery. Since the
target is loss in 2020-2022, the 2022 image observes the loss itself — a
temporal leak. This script replaces it with exg_change_2018_2019, which is
entirely within the feature period and cannot leak target information.

Usage: python -m core.data.fix_exg_column
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

from core.cv.change_detection import compute_exg, _fetch_s2_tile, _tile_coords, ANALYSIS_ZOOM

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

CSVS = [
    DATA_DIR / "palawan_grid.csv",
    DATA_DIR / "sierra_madre_grid.csv",
    DATA_DIR / "mindanao_bukidnon_grid.csv",
]


def compute_exg_for_cell(lat: float, lon: float, year_before: int, year_after: int) -> float:
    """Compute ExG change for a single cell between two years."""
    try:
        tx, ty = _tile_coords(lat, lon, ANALYSIS_ZOOM)
        before = _fetch_s2_tile(year_before, ANALYSIS_ZOOM, tx, ty)
        after = _fetch_s2_tile(year_after, ANALYSIS_ZOOM, tx, ty)

        if before is not None and after is not None:
            exg_b = compute_exg(before)
            exg_a = compute_exg(after)

            # Get pixel position within tile
            n = 2 ** ANALYSIS_ZOOM
            x_float = (lon + 180) / 360 * n
            y_float = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
            px = min(int((x_float - int(x_float)) * 256), 255)
            py = min(int((y_float - int(y_float)) * 256), 255)

            return round(float(exg_a[py, px] - exg_b[py, px]), 4)
    except Exception:
        pass
    return 0.0


def fix_csv(path: Path):
    """Replace exg_change_2018_2022 with exg_change_2018_2019 in a CSV."""
    df = pd.read_csv(path)
    n = len(df)
    print(f"\n{path.name}: {n} cells")

    if "exg_change_2018_2022" not in df.columns:
        print(f"  WARNING: exg_change_2018_2022 column not found, skipping")
        return

    new_exg = []
    for i in range(n):
        lat = float(df.iloc[i]["lat"])
        lon = float(df.iloc[i]["lon"])
        val = compute_exg_for_cell(lat, lon, 2018, 2019)
        new_exg.append(val)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}")

    df = df.drop(columns=["exg_change_2018_2022"])
    df.insert(df.columns.get_loc("high_risk"), "exg_change_2018_2019", new_exg)

    df.to_csv(path, index=False)
    print(f"  Saved with exg_change_2018_2019")

    # Quick leak check
    from scipy.stats import pearsonr
    ly = df["loss_year"].values
    hr = (ly >= 20).astype(int)
    r, p = pearsonr(df["exg_change_2018_2019"], hr)
    print(f"  Leak check: r={r:.4f}, p={p:.2e} (was r=-0.184 with 2022)")


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("Fixing ExG temporal leak: 2018->2022 -> 2018->2019")
    for path in CSVS:
        if path.exists():
            fix_csv(path)
        else:
            print(f"  Skipping {path.name} — not found")
    print("\nDone. All CSVs updated.")


if __name__ == "__main__":
    main()
