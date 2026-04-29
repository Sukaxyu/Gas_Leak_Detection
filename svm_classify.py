"""
Gas concentration classification using SVM.
Pipeline:
  1. Load ratio signal from each experiment.
  2. Extract burst-window features around known onset.
  3. Train SVM (3-class: 200 / 2000 / 20000 ppm).
  4. Leave-One-Out cross-validation.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR    = "DATA/"
DT          = 0.5    # resample interval (s)
BURST_WIN   = 300    # feature extraction window after onset (s)
BASELINE_W  = 300    # seconds before onset used as baseline

# ── dataset ────────────────────────────────────────────────────────────────
# (filepath, col_offset, onset_s, label_ppm)
EXPERIMENTS = [
    ("TEST.csv",          0, 696,  20000),
    ("TEST.csv",          5, 984,   2000),
    ("Sens1_200ppm.csv",    0, 930,    200),
    ("Sens1_200ppm_2.csv",  0, 984,    200),
]


def load_ratio(filepath, col_offset=0):
    df    = pd.read_csv(DATA_DIR + filepath, header=None)
    t     = pd.to_numeric(df.iloc[:, col_offset + 0], errors="coerce").values
    i_ref = pd.to_numeric(df.iloc[:, col_offset + 2], errors="coerce").values
    i_gas  = pd.to_numeric(df.iloc[:, col_offset + 4], errors="coerce").values
    ratio = i_gas / i_ref
    mask  = np.isfinite(t) & np.isfinite(ratio)
    t, ratio = t[mask], ratio[mask]
    t_new = np.arange(t[0], t[-1], DT)
    ratio = interp1d(t, ratio, kind="linear", fill_value="extrapolate")(t_new)
    ratio = savgol_filter(ratio, 21, 3)
    return t_new, ratio


def extract_features(t, ratio, onset, burst=BURST_WIN, baseline_w=BASELINE_W):
    """
    Extract scalar features from the burst window [onset, onset+burst].
    Baseline is computed from [onset-baseline_w, onset].

    Features:
      peak_rel   : max relative rise above baseline in burst window
      t_half     : time to reach 50% of peak (response speed)
      slope_30s  : mean slope in first 30s after onset
      slope_60s  : mean slope in 30-60s after onset
      area_120s  : area under normalised curve in first 120s
      curvature  : slope_30s / (slope_60s + eps)  — shape of initial rise
    """
    i_on  = np.searchsorted(t, onset)
    i_end = np.searchsorted(t, onset + burst)
    i_bl0 = max(0, np.searchsorted(t, onset - baseline_w))

    baseline = np.median(ratio[i_bl0:i_on])
    eps      = np.abs(baseline) * 1e-6 + 1e-12

    # normalised burst signal (relative deviation from baseline)
    burst_r = (ratio[i_on:i_end] - baseline) / np.abs(baseline)
    burst_t = t[i_on:i_end] - t[i_on]          # time from 0

    peak_rel = float(np.max(burst_r))

    # time to 50% of peak
    half     = peak_rel * 0.5
    above    = np.where(burst_r >= half)[0]
    t_half   = float(burst_t[above[0]]) if len(above) > 0 else float(burst_t[-1])

    # slopes
    w30 = int(30 / DT)
    w60 = int(60 / DT)
    slope_30s = float(np.mean(np.gradient(burst_r[:w30], DT))) if len(burst_r) >= w30 else 0.0
    slope_60s = float(np.mean(np.gradient(burst_r[w30:w60], DT))) if len(burst_r) >= w60 else 0.0

    # area under curve in first 120s
    w120     = int(120 / DT)
    area_120 = float(np.trapz(burst_r[:w120], dx=DT)) if len(burst_r) >= w120 else 0.0

    curvature = slope_30s / (np.abs(slope_60s) + 1e-9)

    return np.array([peak_rel, t_half, slope_30s, slope_60s, area_120, curvature])


FEAT_NAMES = ["peak_rel", "t_half", "slope_30s", "slope_60s", "area_120s", "curvature"]


def build_dataset():
    X, y, names = [], [], []
    for fpath, offset, onset, label in EXPERIMENTS:
        t, ratio = load_ratio(fpath, col_offset=offset)
        feat = extract_features(t, ratio, onset)
        X.append(feat)
        y.append(label)
        names.append(f"{fpath}[+{offset}] {label}ppm")
        print(f"  {label:6d}ppm  {fpath}[+{offset}]  "
              f"peak={feat[0]:.4f}  t_half={feat[1]:.1f}s  "
              f"slope30={feat[2]:.2e}  area120={feat[4]:.4f}")
    return np.array(X), np.array(y), names


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION")
    X, y, names = build_dataset()

    classes = sorted(set(y))
    print(f"\n  Classes: {classes}")
    print(f"  Samples: {len(y)}")

    # ── Leave-One-Out cross-validation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    loo     = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])

        model = SVC(kernel="rbf", C=10, gamma="scale",
                    class_weight="balanced", random_state=42)
        model.fit(X_train, y[train_idx])

        pred = model.predict(X_test)[0]
        y_true.append(y[test_idx][0])
        y_pred.append(pred)

        status = "OK" if pred == y[test_idx][0] else "WRONG"
        print(f"  [{status}] {names[test_idx[0]]:40s}  "
              f"true={y[test_idx][0]:6d}  pred={pred:6d}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print(classification_report(y_true, y_pred,
                                labels=classes,
                                target_names=[f"{c}ppm" for c in classes]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {[f'{c}ppm' for c in classes]}")
    print(confusion_matrix(y_true, y_pred, labels=classes))


if __name__ == "__main__":
    main()
