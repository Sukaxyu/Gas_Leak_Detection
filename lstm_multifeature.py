"""
Multi-Feature LSTM Autoencoder — Gas Anomaly Detection
======================================================

Why multi-feature?
  Using only I_Gas/I_Ref forces us to assume Gas always raises (or always
  lowers) the ratio.  Sens1 rises; Sens2 falls.  With 6 simultaneous features
  the AE learns the joint N2 dynamics of ALL channels.  Any deviation —
  whether a rise, a fall, or a cross-channel decorrelation — inflates the
  reconstruction error.  No direction assumption needed.

Features (6 channels, one vector per timestep)
  I_Gas          Gas-channel intensity
  I_Ref         reference-channel intensity
  λ_Gas          Gas-channel Bragg wavelength
  λ_Ref         reference Bragg wavelength
  ratio         I_Gas / I_Ref
  Δλ            λ_Gas − λ_Ref   (temperature-compensated differential)

Pre-processing
  StandardScaler fitted on N2 training windows only.
  Each feature is independently centred (mean=0) and scaled (std=1).
  → All features contribute equally regardless of physical unit / magnitude.
  This is the standard ML practice.

Architecture
  Encoder : LSTM(6 → HIDDEN) → last hidden → Linear(HIDDEN → LATENT)
  Decoder : Linear(LATENT → HIDDEN) → repeat WIN_LEN times
            → LSTM(HIDDEN → HIDDEN) → Linear(HIDDEN → 6)

Detection (same change-point logic as h2_detect.py)
  1. Calibration : mean recon error on the first CALIB_WIN seconds (N2).
  2. Flag        : point-wise error > CALIB_MEAN × RECON_MULT.
  3. Confirm     : ≥ PERSIST_FRAC of PERSIST_WIN-second window flagged.
  4. No direction assumption — works for Sens1 and Sens2 without modification.

Experiments
  TEST.csv   col0  20 000 ppm  Gas: 696 s → 3396 s
  TEST.csv   col5   2 000 ppm  Gas: 984 s → 6300 s
  Sens1_200ppm col0     200 ppm  Gas:1150 s →10500 s
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Hyper-parameters ───────────────────────────────────────────────────────
DT           = 0.5      # resample interval (s)
WIN_LEN      = 120      # window length in timesteps (= 60 s)
STEP         = 5        # sliding stride in timesteps (= 2.5 s)
HIDDEN       = 64       # LSTM hidden units (larger: 6 features vs 1)
LATENT       = 16       # bottleneck dimension
EPOCHS       = 200
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42

# Detection  (MODERATE — optimised via parameter sweep)
CALIB_WIN    = 300      # initial N2 used for calibration (s)
WIN_LEN      = 60       # window length: 30 s  (↓ from 60 s → earlier first error point)
RECON_MULT   = 2.5      # flag when error > CALIB_MEAN × RECON_MULT  (↓ from 3.0)
PERSIST_WIN  = 60       # persistence window: 60 s  (↓ from 120 s → faster confirmation)
PERSIST_FRAC = 0.65     # fraction of window flagged  (↓ from 0.70)
N2_SKIP_S    = 120      # skip first N seconds of post-Gas N2 (drop phase)

DATA_DIR     = "DATA/"
FEAT_NAMES   = ["I_Gas", "I_Ref", "λ_Gas", "λ_Ref", "ratio", "Δλ"]
N_FEAT       = len(FEAT_NAMES)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Experiment definitions ─────────────────────────────────────────────────
# (filepath, sep, col_offset, h2_onset_s, h2_end_s, label)
EXPERIMENTS = [
    ("TEST.csv",       ",", 0,  696,  3396,  "Sens1 — 20000 ppm"),
    ("TEST.csv",       ",", 5,  984,  6300,  "Sens1 —  2000 ppm"),
    ("Sens1_200ppm.csv", ",", 0, 1150, 10500,  "Sens1 —   200 ppm"),
]


# ── Raw data loading ───────────────────────────────────────────────────────
def load_raw(filepath, sep=",", col_offset=0):
    """
    Load CSV or tab-separated .dat file.
    Returns (t, I_Gas, I_Ref, λ_Gas, λ_Ref) as resampled, smoothed arrays.
    Column layout: time | λ_Ref | I_Ref | λ_Gas | I_Gas  (offset applied)
    """
    df = pd.read_csv(filepath, sep=sep, header=None)
    o  = col_offset

    t      = pd.to_numeric(df.iloc[:, o + 0], errors="coerce").values
    lam_ref = pd.to_numeric(df.iloc[:, o + 1], errors="coerce").values
    i_ref   = pd.to_numeric(df.iloc[:, o + 2], errors="coerce").values
    lam_gas  = pd.to_numeric(df.iloc[:, o + 3], errors="coerce").values
    i_gas    = pd.to_numeric(df.iloc[:, o + 4], errors="coerce").values

    mask = (np.isfinite(t) & np.isfinite(i_ref) &
            np.isfinite(i_gas) & np.isfinite(lam_ref) & np.isfinite(lam_gas))
    t, i_ref, i_gas, lam_ref, lam_gas = (
        t[mask], i_ref[mask], i_gas[mask], lam_ref[mask], lam_gas[mask])

    t_new = np.arange(t[0], t[-1], DT)
    def resample_smooth(sig):
        s = interp1d(t, sig, kind="linear", fill_value="extrapolate")(t_new)
        return savgol_filter(s, window_length=21, polyorder=3)

    return (t_new,
            resample_smooth(i_gas),
            resample_smooth(i_ref),
            resample_smooth(lam_gas),
            resample_smooth(lam_ref))


def build_feature_matrix(i_gas, i_ref, lam_gas, lam_ref):
    """Compute 6-channel feature matrix (N, 6)."""
    ratio = i_gas / (i_ref + 1e-12)
    delta_lam = lam_gas - lam_ref
    return np.stack([i_gas, i_ref, lam_gas, lam_ref, ratio, delta_lam], axis=1)


def make_windows(feat_mat, win=WIN_LEN, step=STEP):
    """Sliding windows → (N_win, win, N_FEAT)."""
    segs = []
    for i in range(0, len(feat_mat) - win + 1, step):
        segs.append(feat_mat[i : i + win])
    return np.array(segs, dtype=np.float32)


# ── LSTM Autoencoder ───────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Seq2seq multivariate LSTM AE.
    Input/output : (batch, WIN_LEN, N_FEAT)
    """
    def __init__(self, n_feat=N_FEAT, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.win = WIN_LEN

        self.enc_lstm   = nn.LSTM(n_feat, hidden, batch_first=True)
        self.enc_linear = nn.Linear(hidden, latent)

        self.dec_linear = nn.Linear(latent, hidden)
        self.dec_lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.dec_out    = nn.Linear(hidden, n_feat)

    def forward(self, x):
        # x : (B, T, F)
        _, (h, _) = self.enc_lstm(x)          # h : (1, B, hidden)
        z = self.enc_linear(h[0])             # z : (B, latent)
        d = self.dec_linear(z).unsqueeze(1).repeat(1, self.win, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_out(d)                 # (B, T, F)


# ── Per-experiment pipeline ────────────────────────────────────────────────
def run_experiment(fpath, sep, col_offset, onset_true, h2_end_s, label,
                   data_prefix=DATA_DIR):
    print(f"\n{'='*62}")
    print(f"EXPERIMENT : {label}")
    print(f"File       : {fpath}  col+{col_offset}  "
          f"onset={onset_true}s  H2_end={h2_end_s}s")

    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
        data_prefix + fpath, sep=sep, col_offset=col_offset)

    feat = build_feature_matrix(i_gas, i_ref, lam_gas, lam_ref)  # (N, 6)

    # ── N2 segments ──────────────────────────────────────────────────────
    i_onset = int((onset_true - t[0]) / DT)
    i_post  = int((h2_end_s  - t[0]) / DT) + int(N2_SKIP_S / DT)
    i_post  = min(i_post, len(t))

    pre_n2  = feat[:i_onset]
    post_n2 = feat[i_post:] if i_post < len(feat) else np.zeros((0, N_FEAT))
    n2_feat = np.concatenate([pre_n2, post_n2], axis=0)

    print(f"  Pre-Gas  N2 : {len(pre_n2)} samples  "
          f"({len(pre_n2)*DT/60:.1f} min)")
    print(f"  Post-Gas N2 : {len(post_n2)} samples  "
          f"({len(post_n2)*DT/60:.1f} min)")

    # ── StandardScaler fitted on N2 only ─────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(n2_feat)
    n2_scaled   = scaler.transform(n2_feat).astype(np.float32)
    full_scaled = scaler.transform(feat).astype(np.float32)

    print(f"  Features scaled (μ≈0, σ≈1 on N2 data)")

    # ── Build training windows ────────────────────────────────────────────
    W = make_windows(n2_scaled)
    print(f"  Training windows : {len(W)}")

    idx  = np.random.permutation(len(W))
    n_tr = int(0.9 * len(W))
    W_tr, W_vl = W[idx[:n_tr]], W[idx[n_tr:]]

    def to_loader(arr, shuffle=True):
        x = torch.tensor(arr)
        return DataLoader(TensorDataset(x), batch_size=BATCH_SIZE,
                          shuffle=shuffle)

    tr_loader = to_loader(W_tr)
    vl_loader = to_loader(W_vl, shuffle=False)

    # ── Train ─────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = LSTMAutoencoder()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    best_val, best_state = float("inf"), None
    history = {"train": [], "val": []}

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (xb,) in tr_loader:
            pred = model(xb)
            loss = crit(pred, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(W_tr)

        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for (xb,) in vl_loader:
                vl_loss += crit(model(xb), xb).item() * len(xb)
        vl_loss /= max(len(W_vl), 1)

        history["train"].append(tr_loss)
        history["val"].append(vl_loss)
        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS}  "
                  f"train={tr_loss:.2e}  val={vl_loss:.2e}")

    model.load_state_dict(best_state)
    print(f"  Best val MSE : {best_val:.2e}")

    # ── Reconstruction error on full trace ────────────────────────────────
    model.eval()
    errors = np.full(len(t), np.nan)
    with torch.no_grad():
        for i in range(0, len(full_scaled) - WIN_LEN + 1, STEP):
            win = torch.tensor(
                full_scaled[i : i + WIN_LEN]).unsqueeze(0)   # (1, T, F)
            rec = model(win)
            # Mean over timesteps AND features → scalar per window
            err = float(((rec - win) ** 2).mean())
            errors[i + WIN_LEN - 1] = err   # causal assignment

    # Smooth (30 s rolling median)
    err_s = (pd.Series(errors).interpolate()
               .rolling(int(30 / DT), min_periods=1).median().values)

    # ── Calibration ───────────────────────────────────────────────────────
    n_calib      = int(CALIB_WIN / DT)
    calib_vals   = err_s[WIN_LEN : n_calib]
    calib_vals   = calib_vals[np.isfinite(calib_vals)]
    calib_mean   = float(np.mean(calib_vals))
    threshold    = calib_mean * RECON_MULT
    print(f"  Calib mean error : {calib_mean:.2e}  "
          f"threshold ({RECON_MULT}×) : {threshold:.2e}")

    # ── Detect (flag + persist, no direction assumption) ──────────────────
    flagged   = (err_s > threshold).astype(float)
    n_persist = int(PERSIST_WIN / DT)
    frac      = pd.Series(flagged).rolling(n_persist).mean().values

    alarm_idx = None
    for i in range(n_calib, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= PERSIST_FRAC:
            alarm_idx = i
            break

    print(f"\n  {'─'*40}")
    if alarm_idx is not None:
        alarm_t = t[alarm_idx]
        error_s = alarm_t - onset_true
        print(f"  Alarm time     : {alarm_t:.1f} s  ({alarm_t/60:.1f} min)")
        print(f"  True onset     : {onset_true} s  ({onset_true/60:.1f} min)")
        print(f"  Detection error: {error_s:+.1f} s  "
              f"({'early' if error_s < 0 else 'late'})")
    else:
        print("  No alarm triggered.")
    print(f"  {'─'*40}")

    return dict(label=label, t=t, feat=feat, err_s=err_s,
                threshold=threshold, onset_true=onset_true,
                alarm_idx=alarm_idx, history=history,
                scaler=scaler, i_gas=i_gas, i_ref=i_ref,
                lam_gas=lam_gas, lam_ref=lam_ref)


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_all(results):
    n   = len(results)
    fig = plt.figure(figsize=(14, 5 * n))
    gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    for k, res in enumerate(results):
        t        = res["t"]
        err_s    = res["err_s"]
        thr      = res["threshold"]
        onset_tr = res["onset_true"]
        al_idx   = res["alarm_idx"]
        ratio    = res["i_gas"] / (res["i_ref"] + 1e-12)

        ax  = fig.add_subplot(gs[k])
        ax2 = ax.twinx()

        ax.plot(t / 60, ratio, color="steelblue", lw=0.7, alpha=0.7,
                label="I_Gas / I_Ref  (ratio)")
        ax2.plot(t / 60, err_s, color="tomato", lw=1.2,
                 label="Reconstruction MSE (6-feature)")
        ax2.axhline(thr, color="red", ls="--", lw=1,
                    label=f"Threshold ({RECON_MULT}× calib)")
        ax.axvspan(0, CALIB_WIN / 60, alpha=0.07, color="gray",
                   label="Calibration zone")
        ax.axvline(onset_tr / 60, color="green", ls="--", lw=1.5,
                   label=f"True onset ({onset_tr}s)")

        if al_idx is not None:
            alarm_t = t[al_idx]
            ax.axvline(alarm_t / 60, color="red", ls=":", lw=1.5,
                       label=f"Alarm ({alarm_t:.0f}s, "
                             f"{alarm_t - onset_tr:+.0f}s)")

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("I_Gas / I_Ref", color="steelblue")
        ax2.set_ylabel("Recon MSE (standardised)", color="tomato")
        ax.set_title(f"{res['label']}  —  6-feature LSTM Autoencoder")

        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper left", ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("Multi-Feature LSTM Autoencoder — Gas Anomaly Detection\n"
                 "(6 channels: I_Gas, I_Ref, λ_Gas, λ_Ref, ratio, Δλ  |  "
                 "StandardScaler on N2)",
                 fontsize=11, y=1.01)
    plt.savefig("multifeature_result.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → multifeature_result.png")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Multi-Feature LSTM Autoencoder — Gas Leak Detection")
    print("Features : I_Gas, I_Ref, λ_Gas, λ_Ref, ratio, Δλ")
    print("Scaler   : StandardScaler (fit on N2, applied to all data)\n")

    results = []
    for fpath, sep, offset, onset, h2_end, label in EXPERIMENTS:
        res = run_experiment(fpath, sep, offset, onset, h2_end, label)
        results.append(res)

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  {'Experiment':<20} {'True onset':>11} {'Alarm time':>11} "
          f"{'Error':>8}")
    print("  " + "─" * 54)
    for res in results:
        if res["alarm_idx"] is not None:
            alarm_t = res["t"][res["alarm_idx"]]
            err     = alarm_t - res["onset_true"]
            print(f"  {res['label']:<20} {res['onset_true']:>8} s   "
                  f"{alarm_t:>8.0f} s  {err:>+7.0f} s")
        else:
            print(f"  {res['label']:<20} {res['onset_true']:>8} s   "
                  f"{'--':>8}     {'--':>8}")

    plot_all(results)
