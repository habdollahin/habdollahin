"""A tiny dose-volume histogram (DVH) tool, built as a vibe-coding learning example.

Plain Python, no dependencies, so every line can be read and understood.
The real project does the same job with NumPy in dosevlm/metrics.py.

Run it:   python3 dvh_tool.py
Test it:  python3 -m unittest test_dvh_tool.py
"""

import math


def _check(doses):
    # Round 3 fix: an empty structure used to crash with ZeroDivisionError.
    if not doses:
        raise ValueError("structure has no voxels")


def mean_dose(doses):
    """Average dose over all voxels in the structure (Gy)."""
    _check(doses)
    return sum(doses) / len(doses)


def dose_at_volume(doses, percent):
    """D_x: the minimum dose received by the hottest `percent`% of the volume.

    Example: D95 = 66 Gy means 95% of the structure gets at least 66 Gy.
    """
    _check(doses)
    hottest_first = sorted(doses, reverse=True)
    # How many voxels make up `percent`% of the volume (at least 1).
    n = max(1, math.ceil(len(doses) * percent / 100))
    return hottest_first[n - 1]


def volume_at_dose(doses, threshold_gy):
    """V_x: percentage of the volume receiving at least `threshold_gy`."""
    _check(doses)
    covered = sum(1 for d in doses if d >= threshold_gy)
    return 100 * covered / len(doses)


def dvh_curve(doses, step_gy=5):
    """Cumulative DVH as (dose, volume %) points from 0 Gy up to the max dose."""
    _check(doses)
    points = []
    dose = 0
    while dose <= max(doses):
        points.append((dose, volume_at_dose(doses, dose)))
        dose += step_gy
    return points


def report(name, doses):
    """Print a short DVH summary for one structure."""
    print(f"== {name} ({len(doses)} voxels) ==")
    print(f"  Mean dose : {mean_dose(doses):6.2f} Gy")
    print(f"  Max dose  : {max(doses):6.2f} Gy")
    print(f"  D95       : {dose_at_volume(doses, 95):6.2f} Gy")
    print(f"  V20       : {volume_at_dose(doses, 20):6.1f} %")
    print("  DVH:")
    for dose, volume in dvh_curve(doses, step_gy=10):
        bar = "#" * round(volume / 5)  # one '#' per 5% of volume
        print(f"    {dose:5.0f} Gy |{bar:<20}| {volume:5.1f}%")


if __name__ == "__main__":
    # Made-up voxel doses (Gy), just for the demo.
    ptv = [68.5, 69.2, 70.1, 70.4, 70.0, 71.3, 69.8, 66.1, 70.7, 72.0]
    lung = [2.0, 5.5, 8.1, 12.4, 18.9, 21.3, 25.0, 4.2, 30.6, 9.9]
    report("PTV", ptv)
    print()
    report("Lung", lung)
