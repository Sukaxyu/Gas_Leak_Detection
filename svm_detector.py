"""
SVM-based Gas Leak Detection for FBG Sensors
=============================================
Extracts 10 engineered features from the normalised intensity ratio
I_Gas/I_Ref, trains an RBF-SVM on labelled N2/Gas segments,
and validates on unseen experiments.

Usage
-----
  # Train on defaults, validate on Sens1_200ppm.csv:
  python h2_svm.py

  # Custom training data and validation target:
  python h2_svm.py --val-file DATA/Sens2_2000ppm.csv --val-onset 936

  # Full help:
  python h2_svm.py --help
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="RBF-SVM gas leak detector for FBG sensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    p.add_argument(
        "--data-dir", default="DATA/",
        help="Directory containing CSV data files",
    )
    p.add_argument(
        "--train-files", nargs="+",
        default=["TEST.csv:0:696", "TEST.csv:5:984"],
        metavar="FILE:COL_OFFSET:ONSET_S",
        help=(
            "Training files in FILE:COL_OFFSET:ONSET_S format. "
            "COL_OFFSET=5 selects the second dataset packed in TEST.csv."
        ),
    )
    p.add_argument(
        "--val-file", default="Sens1_200ppm.csv",
        help="CSV file to validate on",
    )
    p.add_argument(
        "--val-onset", type=float, default=1150.0,
        help="True Gas onset time in the validation file (seconds)",
    )
    p.add_argument(
        "--val-col-offset", type=int, default=0,
        help="Column offset for the validation CSV file",
    )

    # ── Model ──
    p.add_argument(
        "--svm-c", type=float, default=2.5,
        help="SVM regularisation parameter C",
    )
    p.add_argument(
        "--prob-threshold", type=float, default=0.80,
        help="Gas probability threshold to trigger detection [0–1]",
    )
    p.add_argument(
        "--gas-label-duration", type=float, default=120.0,
        help="Seconds after onset labelled as Gas for training",
    )

    return p.parse_args()


# ── Constants ──────────────────────────────────────────────────────────────
DT = 0.5   # resampling interval (s)

FEAT_COLS = [
    "k_10s", "energy_30s", "accel", "slope_ratio", "snr",
    "zscore", "accel_persist", "pos_frac", "net_60s", "min_slope_60s",
]


# ── Data loading ───────────────────────────────────────────────────────────
def load_and_resample(filepath, col_offset=0):
    """
    Read CSV, compute I_Gas/I_Ref intensity ratio, resample to DT interval,
    and normalise to each sensor's own 120 s baseline median.

    CSV column order (per sensor block, zero-indexed from col_offset):
      0: time  1: lambda_Ref  2: I_Ref  3: lambda_Gas  4: I_Gas
    """
    df = pd.read_csv(filepath, header=None)
    o  = col_offset

    t     = pd.to_numeric(df.iloc[:, o+0], errors="coerce").values
    i_ref = pd.to_numeric(df.iloc[:, o+2], errors="coerce").values
    i_gas  = pd.to_numeric(df.iloc[:, o+4], errors="coerce").values

    ratio = i_gas / (i_ref + 1e-12)
    mask  = np.isfinite(t) & np.isfinite(ratio)
    t, ratio = t[mask], ratio[mask]

    t_new     = np.arange(t[0], t[-1], DT)
    ratio_new = interp1d(t, ratio, kind="linear",
                         fill_value="extrapolate")(t_new)
    ratio_new = savgol_filter(ratio_new, window_length=21, polyorder=3)

    # Normalise to sensor's own 120 s baseline (makes cross-sensor comparison fair)
    baseline  = np.median(ratio_new[:int(120 / DT)])
    ratio_new = (ratio_new - baseline) / (abs(baseline) + 1e-9)

    return pd.DataFrame({"time": t_new, "ratio": ratio_new})


# ── Feature engineering ────────────────────────────────────────────────────
def extract_features(df):
    """
    Extract 10 dynamic features from the normalised ratio time series.
    All windows are computed at the native DT resolution.
    """
    w10  = int(10  / DT)
    w30  = int(30  / DT)
    w60  = int(60  / DT)
    w300 = int(300 / DT)

    df = df.copy()
    k1 = df["ratio"].diff()

    df["k_10s"]        = df["ratio"].diff(w10)
    df["energy_30s"]   = df["ratio"].rolling(w30).sum().diff()
    df["accel"]        = k1 - k1.rolling(w60).mean()

    k_300              = df["ratio"].diff(w300)
    df["slope_ratio"]  = df["k_10s"] / (k_300.abs() + 1e-9)

    local_std          = df["ratio"].rolling(w60).std().replace(0, np.nan)
    df["snr"]          = df["k_10s"].abs() / local_std

    roll_mean          = df["ratio"].rolling(w300).mean()
    roll_std           = df["ratio"].rolling(w300).std().replace(0, np.nan)
    df["zscore"]       = (df["ratio"] - roll_mean) / roll_std

    df["accel_persist"]= df["accel"].rolling(w10).mean()
    df["pos_frac"]     = k1.gt(0).rolling(w60).mean()
    df["net_60s"]      = df["ratio"].diff(w60)
    df["min_slope_60s"]= df["k_10s"].rolling(w60).min()

    return df.dropna().reset_index(drop=True)


# ── Labelling ──────────────────────────────────────────────────────────────
def label_binary(df, onset, h2_duration):
    """
    Binary labels: 0 = N2 background, 1 = Gas response.
    Gas region: [onset, onset + h2_duration).
    """
    df = df.copy()
    df["label"] = 0
    mask = (df["time"] >= onset) & (df["time"] < onset + h2_duration)
    df.loc[mask, "label"] = 1
    return df


# ── Onset refinement ───────────────────────────────────────────────────────
def refine_onset(df, confirm_idx, lookback_sec=180):
    """
    Back-track from SVM confirmation time to find the acceleration peak,
    which estimates the physical start of Gas absorption.
    """
    win   = int(lookback_sec / DT)
    start = max(0, confirm_idx - win)
    zone  = df.iloc[start:confirm_idx + 1]

    pos = zone[zone["accel"] > 0]
    if not pos.empty:
        return df.loc[pos["accel"].idxmax(), "time"]
    return df.loc[zone["accel"].idxmax(), "time"]


# ── Training ───────────────────────────────────────────────────────────────
def train(train_specs, data_dir, h2_label_duration, svm_c):
    """
    train_specs: list of (filename, col_offset, onset_s) tuples.
    """
    print("=" * 50)
    print("TRAINING")

    parts = []
    for fname, col_offset, onset_s in train_specs:
        seg = extract_features(load_and_resample(data_dir + fname,
                                                 col_offset=col_offset))
        seg = label_binary(seg, onset=onset_s, h2_duration=h2_label_duration)
        parts.append(seg)
        tag = fname + (f" [col+{col_offset}]" if col_offset else "")
        print(f"  {tag}: {len(seg)} samples  "
              f"Gas={seg['label'].sum()}  N2={(seg['label']==0).sum()}")

    train_df = pd.concat(parts, ignore_index=True)
    print(f"  Total: {len(train_df)}  |  "
          f"Gas: {train_df['label'].sum()}  |  "
          f"N2: {(train_df['label']==0).sum()}")

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[FEAT_COLS])
    y = train_df["label"].values

    model = SVC(kernel="rbf", C=svm_c, class_weight="balanced",
                probability=True, random_state=42)
    model.fit(X, y)
    print("  Training complete.")
    return model, scaler


# ── Validation ─────────────────────────────────────────────────────────────
def validate(model, scaler, val_path, col_offset, onset_true,
             prob_threshold, h2_label_duration):
    print("=" * 50)
    print(f"VALIDATING: {val_path}")

    raw = load_and_resample(val_path, col_offset=col_offset)
    df  = extract_features(raw)

    X_val        = scaler.transform(df[FEAT_COLS])
    df["pred"]   = model.predict(X_val)
    df["prob"]   = model.predict_proba(X_val)[:, 1]
    df["prob_max"] = df["prob"].rolling(int(30 / DT)).max()

    hits = df[(df["prob_max"] >= prob_threshold) & (df["time"] > 60)]

    print("-" * 50)
    if not hits.empty:
        confirm_idx  = hits.index[0]
        confirm_time = df.loc[confirm_idx, "time"]
        exact_time   = refine_onset(df, confirm_idx)

        if exact_time >= confirm_time:
            exact_time = confirm_time - 45.0

        print(f"  Gas detected")
        print(f"  SVM confirm time : {confirm_time:.1f} s  "
              f"({confirm_time/60:.2f} min)")
        print(f"  Estimated onset  : {exact_time:.1f} s  "
              f"({exact_time/60:.2f} min)")
        print(f"  Lead time        : {confirm_time - exact_time:.1f} s")

        if onset_true is not None:
            err = exact_time - onset_true
            print(f"  True onset       : {onset_true:.1f} s  "
                  f"(error {err:+.1f} s)")
    else:
        print("  No Gas pattern detected.")

    if onset_true is not None:
        df_lab = label_binary(df, onset=onset_true,
                              h2_duration=h2_label_duration)
        print("\n  Classification report:")
        print(classification_report(df_lab["label"], df["pred"],
                                    target_names=["N2", "Gas"]))
        print("  Confusion matrix:")
        print(confusion_matrix(df_lab["label"], df["pred"]))

    print("-" * 50)
    return df


# ── Main ───────────────────────────────────────────────────────────────────
def parse_train_specs(specs, data_dir):
    """Parse 'FILE:COL_OFFSET:ONSET_S' strings into tuples."""
    result = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"[ERROR] --train-files entry must be FILE:COL_OFFSET:ONSET_S, "
                  f"got: {spec}")
            sys.exit(1)
        fname, col_offset, onset_s = parts[0], int(parts[1]), float(parts[2])
        result.append((fname, col_offset, onset_s))
    return result


def main():
    args = parse_args()

    train_specs = parse_train_specs(args.train_files, args.data_dir)

    model, scaler = train(
        train_specs,
        data_dir=args.data_dir,
        h2_label_duration=args.h2_label_duration,
        svm_c=args.svm_c,
    )

    validate(
        model, scaler,
        val_path=args.data_dir + args.val_file,
        col_offset=args.val_col_offset,
        onset_true=args.val_onset,
        prob_threshold=args.prob_threshold,
        h2_label_duration=args.h2_label_duration,
    )


if __name__ == "__main__":
    main()
