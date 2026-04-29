"""
Derivative-Space LSTM Autoencoder — Gas Anomaly Detection
=========================================================

Key insight (from user):
  In N2, ALL experiments share the same dynamic signature:
  features are nearly stationary  → first derivatives ≈ 0.
  When Gas enters, signal changes rapidly → derivatives spike.

  Training on DERIVATIVES (not absolute values) makes the model
  truly cross-experiment generalizable:
    - No per-experiment baseline normalisation needed
    - N2 derivative windows from all experiments are pooled directly
    - One global model, one global StandardScaler
    - Sens1 (rising) and Sens2 (falling) both produce large |derivative|
      → same model detects both without direction assumption

Input features (6 first-order derivatives, computed from SG filter):
  dI_Gas/dt,  dI_Ref/dt,  dλ_Gas/dt,  dλ_Ref/dt,  d(ratio)/dt,  d(Δλ)/dt

LOO validation:
  Train  : N2 derivatives from Sens1-20000ppm + Sens1-2000ppm + Sens1-200ppm
  Test   : Sens1-200ppm-2  (completely unseen)
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

# ── Hyper-parameters ──────────────────────────────────────────────────────
DT           = 0.5      # resample interval (s)
WIN_LEN      = 60       # 30 s window
STEP         = 5        # 2.5 s stride
HIDDEN       = 64
LATENT       = 16
EPOCHS       = 200
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42

CALIB_WIN    = 300      # s — calibration window for threshold
RECON_MULT   = 3.0      # flag when error > CALIB_MEAN × RECON_MULT
PERSIST_WIN  = 60       # s
PERSIST_FRAC = 0.65
N2_SKIP_S    = 120      # s to skip after Gas→N2 switch

# SG derivative: smooth + differentiate in one step
SG_WIN    = 21          # Savitzky-Golay window (samples)
SG_ORDER  = 3           # polynomial order
SG_DERIV  = 1           # first derivative

DATA_DIR  = "DATA/"
N_FEAT    = 6
FEAT_NAMES = ["dI_Gas/dt", "dI_Ref/dt", "dλ_Gas/dt",
              "dλ_Ref/dt", "d(ratio)/dt", "d(Δλ)/dt"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── All experiments (for individual + LOO tests) ─────────────────────────
# (filepath, sep, col_offset, h2_onset_s, h2_end_s, label)
ALL_EXPS = [
    ("TEST.csv",          ",", 0,  696,  3396,  "Sens1-20000ppm"),
    ("TEST.csv",          ",", 5,  984,  6300,  "Sens1-2000ppm"),
    ("Sens1_200ppm.csv",    ",", 0, 1150, 10500,  "Sens1-200ppm"),
    ("Sens1_200ppm_2.csv",  ",", 0,  984,  None,  "Sens1-200ppm-2"),
]


# ── I/O ────────────────────────────────────────────────────────────────────
def load_raw(filepath, sep=",", col_offset=0):
    df = pd.read_csv(filepath, sep=sep, header=None)
    o  = col_offset
    t      = pd.to_numeric(df.iloc[:, o+0], errors="coerce").values
    lam_ref = pd.to_numeric(df.iloc[:, o+1], errors="coerce").values
    i_ref   = pd.to_numeric(df.iloc[:, o+2], errors="coerce").values
    lam_gas  = pd.to_numeric(df.iloc[:, o+3], errors="coerce").values
    i_gas    = pd.to_numeric(df.iloc[:, o+4], errors="coerce").values
    mask = (np.isfinite(t) & np.isfinite(i_ref) & np.isfinite(i_gas) &
            np.isfinite(lam_ref) & np.isfinite(lam_gas))
    t, i_ref, i_gas, lam_ref, lam_gas = (
        t[mask], i_ref[mask], i_gas[mask], lam_ref[mask], lam_gas[mask])
    t_new = np.arange(t[0], t[-1], DT)
    def rs(s):
        return interp1d(t, s, kind="linear",
                        fill_value="extrapolate")(t_new)
    return t_new, rs(i_gas), rs(i_ref), rs(lam_gas), rs(lam_ref)


def compute_derivatives(i_gas, i_ref, lam_gas, lam_ref):
    """
    Compute 6 first-order derivatives using Savitzky-Golay.
    SG simultaneously smooths and differentiates → clean, noise-robust.
    Output unit: [signal_unit / s]   (divided by DT)
    Returns array of shape (N, 6).
    """
    def sg_deriv(sig):
        return savgol_filter(sig, window_length=SG_WIN,
                             polyorder=SG_ORDER, deriv=SG_DERIV,
                             delta=DT)

    ratio  = i_gas / (i_ref + 1e-12)
    dlam   = lam_gas - lam_ref

    d_i_gas  = sg_deriv(i_gas)
    d_i_ref = sg_deriv(i_ref)
    d_lh2   = sg_deriv(lam_gas)
    d_lref  = sg_deriv(lam_ref)
    d_ratio = sg_deriv(ratio)
    d_dlam  = sg_deriv(dlam)

    return np.stack([d_i_gas, d_i_ref, d_lh2,
                     d_lref, d_ratio, d_dlam], axis=1)


def make_windows(feat_mat, win=WIN_LEN, step=STEP):
    segs = []
    for i in range(0, len(feat_mat) - win + 1, step):
        segs.append(feat_mat[i : i+win])
    return np.array(segs, dtype=np.float32)


# ── LSTM Autoencoder ───────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat=N_FEAT, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.win        = WIN_LEN
        self.enc_lstm   = nn.LSTM(n_feat, hidden, batch_first=True)
        self.enc_linear = nn.Linear(hidden, latent)
        self.dec_linear = nn.Linear(latent, hidden)
        self.dec_lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.dec_out    = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_linear(h[0])
        d = self.dec_linear(z).unsqueeze(1).repeat(1, self.win, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_out(d)


def train_model(Wtr, Wvl):
    def to_loader(arr, shuffle=True):
        return DataLoader(TensorDataset(torch.tensor(arr)),
                          batch_size=BATCH_SIZE, shuffle=shuffle)
    tr_loader = to_loader(Wtr)
    vl_loader = to_loader(Wvl, shuffle=False)

    torch.manual_seed(SEED)
    model = LSTMAutoencoder()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    best_val, best_state = float("inf"), None

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for (xb,) in tr_loader:
            pred = model(xb); loss = crit(pred, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(Wtr)

        model.eval(); vl_loss = 0.0
        with torch.no_grad():
            for (xb,) in vl_loader:
                vl_loss += crit(model(xb), xb).item() * len(xb)
        vl_loss /= max(len(Wvl), 1)

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 50 == 0 or ep == 1:
            print(f"    Epoch {ep:3d}/{EPOCHS}  "
                  f"train={tr_loss:.2e}  val={vl_loss:.2e}")

    model.load_state_dict(best_state)
    print(f"  Best val MSE : {best_val:.2e}")
    return model


def detect(model, scaler, full_deriv, t, onset_true):
    """Run reconstruction error, calibration, and persistence detection."""
    full_scaled = scaler.transform(full_deriv).astype(np.float32)

    model.eval()
    errors = np.full(len(t), np.nan)
    with torch.no_grad():
        for i in range(0, len(full_scaled) - WIN_LEN + 1, STEP):
            win = torch.tensor(full_scaled[i:i+WIN_LEN]).unsqueeze(0)
            err = float(((model(win) - win) ** 2).mean())
            errors[i + WIN_LEN - 1] = err

    err_s = (pd.Series(errors).interpolate()
               .rolling(int(30 / DT), min_periods=1).median().values)

    n_calib    = int(CALIB_WIN / DT)
    calib_vals = err_s[WIN_LEN : n_calib]
    calib_vals = calib_vals[np.isfinite(calib_vals)]
    calib_mean = float(np.mean(calib_vals))
    threshold  = calib_mean * RECON_MULT

    flagged   = (err_s > threshold).astype(float)
    n_persist = int(PERSIST_WIN / DT)
    frac      = pd.Series(flagged).rolling(n_persist).mean().values

    alarm_idx = None
    for i in range(n_calib, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= PERSIST_FRAC:
            alarm_idx = i
            break

    if alarm_idx is not None:
        alarm_t  = t[alarm_idx]
        err_det  = alarm_t - onset_true
        print(f"  Alarm : {alarm_t:.1f} s  |  "
              f"onset={onset_true} s  |  error={err_det:+.1f} s  "
              f"({'early' if err_det < 0 else 'late'})")
    else:
        print("  No alarm triggered.")

    return err_s, threshold, alarm_idx, calib_mean


# ══════════════════════════════════════════════════════════════════════════
# PART A : per-experiment self-supervised (same as before, derivative space)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("PART A — Per-experiment (derivative space, self-calibrated)")
print("=" * 66)

results_A = []
for fpath, sep, offset, onset_s, h2_end_s, label in ALL_EXPS[:3]:
    print(f"\n  {label}")
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
        DATA_DIR + fpath, sep=sep, col_offset=offset)
    deriv = compute_derivatives(i_gas, i_ref, lam_gas, lam_ref)   # (N, 6)

    i_onset = int((onset_s  - t[0]) / DT)
    i_post  = (int((h2_end_s - t[0]) / DT) + int(N2_SKIP_S / DT)
               if h2_end_s else len(t))
    i_post  = min(i_post, len(t))

    n2_deriv = np.concatenate([deriv[:i_onset],
                                deriv[i_post:] if i_post < len(deriv)
                                else np.zeros((0, N_FEAT))], axis=0)

    scaler = StandardScaler().fit(n2_deriv)
    n2_sc  = scaler.transform(n2_deriv).astype(np.float32)
    W      = make_windows(n2_sc)
    idx    = np.random.permutation(len(W))
    ntr    = int(0.9 * len(W))
    Wtr, Wvl = W[idx[:ntr]], W[idx[ntr:]]
    print(f"    N2 windows: {len(W)}  (train={len(Wtr)}, val={len(Wvl)})")

    model = train_model(Wtr, Wvl)
    err_s, thr, al_idx, cm = detect(model, scaler, deriv, t, onset_s)

    results_A.append(dict(label=label, t=t, i_gas=i_gas, i_ref=i_ref,
                          err_s=err_s, threshold=thr, onset_true=onset_s,
                          alarm_idx=al_idx))


# ══════════════════════════════════════════════════════════════════════════
# PART B — LOO: train on 3 experiments, test on unseen Sens1-200ppm-2
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 66)
print("PART B — LOO cross-experiment (derivative space)")
print("  Train: Sens1-20000ppm + Sens1-2000ppm + Sens1-200ppm  (N2 only)")
print("  Test : Sens1-200ppm-2  (completely unseen)")
print("=" * 66)

all_n2_deriv = []
for fpath, sep, offset, onset_s, h2_end_s, label in ALL_EXPS[:3]:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
        DATA_DIR + fpath, sep=sep, col_offset=offset)
    deriv = compute_derivatives(i_gas, i_ref, lam_gas, lam_ref)

    i_onset = int((onset_s  - t[0]) / DT)
    i_post  = (int((h2_end_s - t[0]) / DT) + int(N2_SKIP_S / DT)
               if h2_end_s else len(t))
    i_post  = min(i_post, len(t))

    n2_d = np.concatenate([deriv[:i_onset],
                            deriv[i_post:] if i_post < len(deriv)
                            else np.zeros((0, N_FEAT))], axis=0)
    all_n2_deriv.append(n2_d)
    print(f"  {label:<16}  N2 deriv samples: {len(n2_d)}")

# Pool and scale
n2_pool = np.concatenate(all_n2_deriv, axis=0)
print(f"\n  Pooled N2 : {len(n2_pool)} samples  ({len(n2_pool)*DT/60:.1f} min)")
print(f"  Note: no per-experiment baseline normalisation needed")
print(f"        derivatives are scale-free by construction\n")

scaler_loo = StandardScaler().fit(n2_pool)
n2_sc = scaler_loo.transform(n2_pool).astype(np.float32)
W   = make_windows(n2_sc)
idx = np.random.permutation(len(W))
ntr = int(0.9 * len(W))
Wtr, Wvl = W[idx[:ntr]], W[idx[ntr:]]
print(f"  Training windows : {len(W)}  (train={len(Wtr)}, val={len(Wvl)})")

model_loo = train_model(Wtr, Wvl)

# Apply to unseen Sens1-200ppm-2
fpath, sep, offset, onset_s, h2_end_s, label = ALL_EXPS[3]
print(f"\n  Detecting on : {label}  (onset={onset_s} s)")
t2, i_gas_2, i_ref_2, lam_gas_2, lam_ref_2 = load_raw(
    DATA_DIR + fpath, sep=sep, col_offset=offset)
deriv2 = compute_derivatives(i_gas_2, i_ref_2, lam_gas_2, lam_ref_2)

err_s2, thr2, al_idx2, cm2 = detect(model_loo, scaler_loo, deriv2, t2, onset_s)

result_B = dict(label=label, t=t2, i_gas=i_gas_2, i_ref=i_ref_2,
                err_s=err_s2, threshold=thr2, onset_true=onset_s,
                alarm_idx=al_idx2)


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════
def plot_result(ax, res, subtitle=""):
    t, err_s = res["t"], res["err_s"]
    thr, onset_tr = res["threshold"], res["onset_true"]
    al_idx = res["alarm_idx"]
    ratio = res["i_gas"] / (res["i_ref"] + 1e-12)

    ax2 = ax.twinx()
    ax.plot(t / 60, ratio, color="steelblue", lw=0.7, alpha=0.6,
            label="ratio I_Gas/I_Ref")
    ax2.plot(t / 60, err_s, color="tomato", lw=1.2,
             label="Reconstruction MSE")
    ax2.axhline(thr, color="red", ls="--", lw=1,
                label=f"Threshold ({RECON_MULT}× calib mean)")
    ax.axvspan(0, CALIB_WIN / 60, alpha=0.07, color="gray",
               label="Calibration zone")
    ax.axvline(onset_tr / 60, color="green", ls="--", lw=1.5,
               label=f"True onset ({onset_tr}s)")
    if al_idx is not None:
        alarm_t = t[al_idx]
        ax.axvline(alarm_t / 60, color="red", ls=":", lw=1.5,
                   label=f"Alarm ({alarm_t:.0f}s, {alarm_t-onset_tr:+.0f}s)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("I_Gas/I_Ref", color="steelblue")
    ax2.set_ylabel("Recon MSE (deriv space)", color="tomato")
    ax.set_title(f"{res['label']}  {subtitle}", fontsize=10)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)


n = len(results_A) + 1
fig = plt.figure(figsize=(14, 5 * n))
gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

for k, res in enumerate(results_A):
    plot_result(fig.add_subplot(gs[k]), res, "(per-experiment AE)")

plot_result(fig.add_subplot(gs[n-1]), result_B,
            "(LOO — model never saw this experiment)")

fig.suptitle(
    "Derivative-Space LSTM Autoencoder — Gas Anomaly Detection\n"
    "Input: d/dt of [I_Gas, I_Ref, λ_Gas, λ_Ref, ratio, Δλ]  |  "
    "No per-experiment baseline needed",
    fontsize=11, y=1.01)
plt.savefig("derivative_ae_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → derivative_ae_result.png")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print("SUMMARY")
print(f"  {'Experiment':<20} {'Onset':>8} {'Alarm':>8} {'Error':>8}  Mode")
print("  " + "─" * 58)
for res in results_A:
    if res["alarm_idx"] is not None:
        alarm_t = res["t"][res["alarm_idx"]]
        err = alarm_t - res["onset_true"]
        print(f"  {res['label']:<20} {res['onset_true']:>6}s  "
              f"{alarm_t:>6.0f}s  {err:>+6.0f}s   per-exp")
    else:
        print(f"  {res['label']:<20} {res['onset_true']:>6}s      --       --   per-exp")

if result_B["alarm_idx"] is not None:
    alarm_t = result_B["t"][result_B["alarm_idx"]]
    err = alarm_t - result_B["onset_true"]
    print(f"  {result_B['label']:<20} {result_B['onset_true']:>6}s  "
          f"{alarm_t:>6.0f}s  {err:>+6.0f}s   LOO (unseen)")
else:
    print(f"  {result_B['label']:<20} {result_B['onset_true']:>6}s      --       --   LOO (unseen)")
