import pytest
import numpy as np
from core.cv.change_detection import compute_exg, detect_change

def test_compute_exg():
    """ExG should be positive for green-heavy pixels, negative for red/blue."""
    # Pure green pixel
    green = np.array([[[0, 255, 0]]], dtype=np.float32)
    exg_green = compute_exg(green)
    assert exg_green[0, 0] > 0.5

    # Pure red pixel
    red = np.array([[[255, 0, 0]]], dtype=np.float32)
    exg_red = compute_exg(red)
    assert exg_red[0, 0] < -0.5

    # Balanced pixel
    gray = np.array([[[128, 128, 128]]], dtype=np.float32)
    exg_gray = compute_exg(gray)
    assert abs(exg_gray[0, 0]) < 0.1

def test_exg_batch():
    """ExG should work on full images."""
    img = np.random.randint(0, 255, (256, 256, 3)).astype(np.float32)
    exg = compute_exg(img)
    assert exg.shape == (256, 256)
    assert exg.min() >= -2.1  # ExG range is [-2, 2] for extreme channel ratios
    assert exg.max() <= 2.1

@pytest.mark.network
def test_detect_change_small_area():
    """Integration test with a small bounding box (1-2 tiles).

    Requires network access to Sentinel-2 tile server. Skipped in offline/CI
    environments. Run explicitly with: pytest -m network
    """
    # Small area in Palawan
    try:
        result = detect_change(
            lat_min=9.7, lat_max=9.75,
            lon_min=118.7, lon_max=118.75,
            year_before=2019, year_after=2022,
        )
    except Exception as e:
        pytest.skip(f"Network request failed: {e}")
    assert "loss_pct" in result
    assert "gain_pct" in result
    assert "tiles_analyzed" in result
    assert result["tiles_analyzed"] > 0
    assert "hotspot_tiles" in result
