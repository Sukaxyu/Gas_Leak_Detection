# -*- coding: utf-8 -*-
"""
Three-way comparison: Threshold vs SVM vs LSTM-AE
Outputs JSON with all false-alarm details and detection results.
"""
import numpy as np
import pandas as pd
import json, sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

DT = 0.5
CALIB_WIN = 300
PERSIST_WIN = 60
PERSIST_FRAC = 0.65
N2_SKIP_S = 120

EXPERIMENTS = [
    ("DATA/TEST.csv",        ",", 0,  696,  3396, "Sens1-20000ppm"),
    ("DATA/TEST.csv",        ",", 5,  984,  6300, "Sens1-2000ppm"),
    ("DATA/Sens1_200ppm.csv",  ",", 0, 1150, 10500, "Sens1-200ppm"),
    ("DATA/Sens2_2000ppm.csv", ",", 0,  936,  3600, "Sens2-2000ppm"),
]

# ── helpers ────────────────────────────────────────────────────────────────
def load_raw(fpath, sep, col_offset):
    df = pd.read_csv(fpath, sep=sep, header=None)
    o = col_offset
    t      = pd.to_numeric(df.iloc[:, o+0], errors="coerce").values
    i_ref  = pd.to_numeric(df.iloc[:, o+2], errors="coerce").values
    i_gas   = pd.to_numeric(df.iloc[:, o+4], errors="coerce").values
    lam_ref= pd.to_numeric(df.iloc[:, o+1], errors="coerce").values
    lam_gas = pd.to_numeric(df.iloc[:, o+3], errors="coerce").values
    mask = np.isfinite(t)&np.isfinite(i_ref)&np.isfinite(i_gas)&np.isfinite(lam_ref)&np.isfinite(lam_gas)
    t,i_ref,i_gas,lam_ref,lam_gas = t[mask],i_ref[mask],i_gas[mask],lam_ref[mask],lam_gas[mask]
    t_new = np.arange(t[0], t[-1], DT)
    def rs(s):
        s2 = interp1d(t,s,kind="linear",fill_value="extrapolate")(t_new)
        return savgol_filter(s2,21,3)
    return t_new, rs(i_gas), rs(i_ref), rs(lam_gas), rs(lam_ref)

def get_ratio(i_gas, i_ref):
    return i_gas / (i_ref + 1e-12)

# ── THRESHOLD analysis (detailed false alarm report) ──────────────────────
print("="*60)
print("THRESHOLD METHOD — detailed false alarm analysis")
print("="*60)

thr_results = {}
for fpath, sep, o, onset_s, h2_end_s, label in EXPERIMENTS:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(fpath, sep, o)
    ratio = get_ratio(i_gas, i_ref)

    n_cal = int(CALIB_WIN / DT)
    mu    = ratio[:n_cal].mean()
    sig   = ratio[:n_cal].std()
    thr   = mu + 3.0 * sig

    # All flagged points (above threshold)
    flagged_all = np.where(ratio > thr)[0]

    # Persistence-confirmed alarm
    n_per = int(PERSIST_WIN / DT)
    frac  = pd.Series((ratio > thr).astype(float)).rolling(n_per).mean().values
    alarm_idx = None
    for i in range(n_cal, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= PERSIST_FRAC:
            alarm_idx = i; break

    # Points flagged BEFORE onset (false alarms)
    i_onset = int((onset_s - t[0]) / DT)
    fa_points = flagged_all[flagged_all < i_onset]

    print(f"\n  {label}  onset={onset_s}s  mu={mu:.6f}  sigma={sig:.6f}  threshold={thr:.6f}")
    print(f"  Total flagged points above threshold: {len(flagged_all)}")
    print(f"  FALSE ALARM points (before onset at {onset_s}s): {len(fa_points)}")
    if len(fa_points) > 0:
        fa_t = t[fa_points]
        # Group into continuous segments
        gaps = np.where(np.diff(fa_points) > 10)[0]
        seg_starts = np.concatenate([[0], gaps+1])
        seg_ends   = np.concatenate([gaps, [len(fa_points)-1]])
        print(f"  False alarm segments:")
        segs_out = []
        for s, e in zip(seg_starts, seg_ends):
            ts, te = fa_t[s], fa_t[e]
            n_pts = e - s + 1
            print(f"    t = {ts:.1f}s – {te:.1f}s  ({n_pts} points, {n_pts*DT:.1f}s duration)")
            segs_out.append({"t_start": float(ts), "t_end": float(te),
                             "n_points": int(n_pts), "duration_s": float(n_pts*DT)})
    else:
        segs_out = []

    alarm_t = float(t[alarm_idx]) if alarm_idx is not None else None
    thr_results[label] = {
        "mu": float(mu), "sigma": float(sig), "threshold": float(thr),
        "n_total_flagged": int(len(flagged_all)),
        "n_false_alarm_points": int(len(fa_points)),
        "false_alarm_segments": segs_out,
        "alarm_time": alarm_t,
        "latency": float(alarm_t - onset_s) if alarm_t else None,
        "false_alarm_before_onset": alarm_t is not None and alarm_t < onset_s
    }

# ── SVM ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SVM METHOD")
print("="*60)

FEAT_COLS = ["k_10s","energy_30s","accel","slope_ratio","snr",
             "zscore","accel_persist","pos_frac","net_60s","min_slope_60s"]

def svm_features(t, ratio):
    df = pd.DataFrame({"time": t, "ratio": ratio})
    w10=int(10/DT); w30=int(30/DT); w60=int(60/DT); w300=int(300/DT)
    df["k_10s"]     = df["ratio"].diff(w10)
    df["energy_30s"]= df["ratio"].rolling(w30).sum().diff()
    k1              = df["ratio"].diff()
    df["accel"]     = k1 - k1.rolling(w60).mean()
    k_300           = df["ratio"].diff(w300)
    df["slope_ratio"]= df["k_10s"] / (k_300.abs() + 1e-9)
    local_std       = df["ratio"].rolling(w60).std().replace(0, np.nan)
    df["snr"]       = df["k_10s"].abs() / local_std
    rm = df["ratio"].rolling(w300).mean()
    rs2 = df["ratio"].rolling(w300).std().replace(0, np.nan)
    df["zscore"]    = (df["ratio"] - rm) / rs2
    df["accel_persist"] = df["accel"].rolling(w10).mean()
    df["pos_frac"]  = k1.gt(0).rolling(w60).mean()
    df["net_60s"]   = df["ratio"].diff(w60)
    df["min_slope_60s"] = df["k_10s"].rolling(w60).min()
    return df.dropna().reset_index(drop=True)

def baseline_ratio(t, i_gas, i_ref):
    ratio = get_ratio(i_gas, i_ref)
    bl = np.median(ratio[:int(120/DT)])
    return (ratio - bl) / (abs(bl) + 1e-9)

# Train SVM on Sens1-20000ppm + Sens1-2000ppm (same as original code)
train_parts = []
for fpath, sep, o, onset_s, _, label in EXPERIMENTS[:2]:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(fpath, sep, o)
    ratio = baseline_ratio(t, i_gas, i_ref)
    df = svm_features(t, ratio)
    df["label"] = 0
    df.loc[(df["time"] >= onset_s) & (df["time"] < onset_s + 120), "label"] = 1
    train_parts.append(df)
    print(f"  Train: {label}  N2={( df['label']==0).sum()}  Gas={df['label'].sum()}")

train_df = pd.concat(train_parts, ignore_index=True)
svm_scaler = StandardScaler()
X_tr = svm_scaler.fit_transform(train_df[FEAT_COLS])
y_tr = train_df["label"].values
svm = SVC(kernel="rbf", C=2.5, class_weight="balanced",
          probability=True, random_state=42)
svm.fit(X_tr, y_tr)
print("  SVM trained.")

svm_results = {}
for fpath, sep, o, onset_s, h2_end_s, label in EXPERIMENTS:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(fpath, sep, o)
    ratio = baseline_ratio(t, i_gas, i_ref)
    df = svm_features(t, ratio)

    X_val = svm_scaler.transform(df[FEAT_COLS])
    df["prob"] = svm.predict_proba(X_val)[:, 1]
    df["prob_max"] = df["prob"].rolling(int(30/DT)).max()

    # Count false positives before onset
    i_onset_df = df[df["time"] < onset_s]
    fa_df = i_onset_df[i_onset_df["prob"] >= 0.80]
    fa_pts_before = len(fa_df)

    hits = df[(df["prob_max"] >= 0.80) & (df["time"] > 60)]
    alarm_t = None
    if not hits.empty:
        alarm_t = float(hits.iloc[0]["time"])

    lat = float(alarm_t - onset_s) if alarm_t is not None else None
    fa_before = alarm_t is not None and alarm_t < onset_s

    print(f"\n  {label}  onset={onset_s}s")
    print(f"    High-prob points before onset: {fa_pts_before}")
    if alarm_t:
        print(f"    Alarm: {alarm_t:.1f}s  latency={lat:+.1f}s  {'⚠ FALSE ALARM' if fa_before else ''}")
    else:
        print("    No alarm.")

    svm_results[label] = {
        "alarm_time": alarm_t,
        "latency": lat,
        "false_alarm_before_onset": fa_before,
        "n_high_prob_before_onset": int(fa_pts_before),
    }

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY")
print(f"{'Experiment':<18} {'Onset':>6} | {'Thr alarm':>10} {'Thr err':>8} {'Thr FA':>6} | {'SVM alarm':>10} {'SVM err':>8} {'SVM FA':>6} | LSTM err")
print("-"*100)

lstm = {"Sens1-20000ppm": 788, "Sens1-2000ppm": 1069, "Sens1-200ppm": 1232, "Sens2-2000ppm": 1040}
for _, _, _, onset_s, _, label in EXPERIMENTS:
    tr = thr_results[label]
    sv = svm_results[label]
    lm_alarm = lstm.get(label, None)

    thr_s = f"{tr['alarm_time']:.0f}s" if tr['alarm_time'] else "--"
    thr_e = f"{tr['latency']:+.0f}s" if tr['latency'] else "--"
    thr_fa = "⚠FA" if tr['false_alarm_before_onset'] else "ok"

    svm_s = f"{sv['alarm_time']:.0f}s" if sv['alarm_time'] else "--"
    svm_e = f"{sv['latency']:+.0f}s" if sv['latency'] else "--"
    svm_fa = "⚠FA" if sv['false_alarm_before_onset'] else "ok"

    lm_e = f"{lm_alarm-onset_s:+d}s" if lm_alarm else "--"
    print(f"  {label:<16} {onset_s:>6}s | {thr_s:>10} {thr_e:>8} {thr_fa:>6} | {svm_s:>10} {svm_e:>8} {svm_fa:>6} | {lm_e}")

# Save for PDF
out = {"threshold": thr_results, "svm": svm_results,
       "lstm": {l: {"alarm_time": lstm[l], "latency": lstm[l]-o}
                for _,_,_,o,_,l in EXPERIMENTS}}
with open("comparison_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nData saved -> comparison_data.json")
