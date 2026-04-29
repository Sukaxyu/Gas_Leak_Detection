import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

DATA_DIR  = "DATA/"
DT        = 0.5    # resample interval (s)
BASELINE  = 300    # initial baseline window (s)
THRESHOLD = 0.002  # minimum relative rise above baseline to arm detection
SLOPE_WIN = 30     # slope averaging window (s)
PERSIST   = 120    # confirmed slope must stay positive for this long (s)


def load_ratio(filepath, col_offset=0):
    """Load CSV, compute I_Gas/I_Ref, resample to DT grid, smooth."""
    df    = pd.read_csv(filepath, header=None)
    t     = pd.to_numeric(df.iloc[:, col_offset + 0], errors="coerce").values
    i_ref = pd.to_numeric(df.iloc[:, col_offset + 2], errors="coerce").values
    i_gas  = pd.to_numeric(df.iloc[:, col_offset + 4], errors="coerce").values
    ratio = i_gas / i_ref
    mask  = np.isfinite(t) & np.isfinite(ratio)
    t, ratio = t[mask], ratio[mask]

    t_new = np.arange(t[0], t[-1], DT)
    ratio = interp1d(t, ratio, kind="linear", fill_value="extrapolate")(t_new)
    ratio = savgol_filter(ratio, window_length=21, polyorder=3)
    return t_new, ratio


def detect(t, ratio, threshold=THRESHOLD, persist=PERSIST, baseline_win=BASELINE):
    """
    Detect Gas onset:
      1. Fixed initial baseline (median of first baseline_win seconds).
      2. Flag each point where relative deviation above baseline > threshold.
      3. Confirm when 70%+ of a rolling persist-window is flagged.
    Returns (confirm_time, onset_time) or (None, None).
    """
    n_base    = int(baseline_win / DT)
    n_persist = int(persist / DT)

    baseline  = np.median(ratio[:n_base])
    deviation = (ratio - baseline) / (np.abs(baseline) + 1e-9)
    flagged   = (deviation > threshold).astype(float)

    # Rolling fraction of flagged points in persist window
    frac = pd.Series(flagged).rolling(n_persist).mean().values

    for i in range(n_base, len(t)):
        if frac[i] >= 0.70:
            confirm_idx = i
            # Backtrack to the start of this window
            onset_idx   = max(0, confirm_idx - n_persist + 1)
            # Refine: find first flagged point in that window
            for j in range(onset_idx, confirm_idx + 1):
                if flagged[j] > 0:
                    onset_idx = j
                    break
            return t[confirm_idx], t[onset_idx]

    return None, None


def run(filepath, onset_true=None, col_offset=0,
        threshold=THRESHOLD, persist=PERSIST):
    print("=" * 50)
    print(f"FILE: {filepath}  (offset={col_offset})")

    t, ratio = load_ratio(filepath, col_offset=col_offset)
    confirm_t, onset_t = detect(t, ratio, threshold=threshold, persist=persist)

    if confirm_t is not None:
        print(f"  Gas detected")
        print(f"  Estimated onset  : {onset_t:.1f} s  ({onset_t/60:.1f} min)")
        print(f"  Confirmed after  : {confirm_t:.1f} s  ({confirm_t/60:.1f} min)")
        if onset_true:
            err = onset_t - onset_true
            print(f"  True onset       : {onset_true} s  (error {err:+.1f} s)")
    else:
        print("  No Gas detected.")
    print()


if __name__ == "__main__":
    # Training concentrations (sanity check)
    run(DATA_DIR + "TEST.csv",          onset_true=696,  col_offset=0)   # 20000ppm
    run(DATA_DIR + "TEST.csv",          onset_true=984,  col_offset=5)   # 2000ppm
    # Validation
    run(DATA_DIR + "Sens1_200ppm.csv",    onset_true=930)                  # 200ppm
    run(DATA_DIR + "Sens1_200ppm_2.csv",  onset_true=984)                  # 200ppm_2
