# -*- coding: utf-8 -*-
"""
Cross-Concentration Validation
================================
Train : N2 from Sens1-20000ppm + Sens1-2000ppm  (weights only)
Test  : Sens1-200ppm  (weights pre-trained; only calibration threshold uses own N2)
Also  : Sens2-2000ppm (cross-sensor)

This addresses the reviewer concern: model weights never see 200ppm or Sens2 data.
Only the detection threshold (2.5 * calib_mean) uses the first CALIB_WIN seconds
of the test experiment's own N2 -- which is the standard in-deployment calibration step.
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

DT           = 0.5
WIN_LEN      = 60
STEP         = 5
HIDDEN       = 64
LATENT       = 16
EPOCHS       = 200
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42
CALIB_WIN    = 300
RECON_MULT   = 2.5
PERSIST_WIN  = 60
PERSIST_FRAC = 0.65
N2_SKIP_S    = 120
N_FEAT       = 6
N_BL         = int(300 / DT)   # for per-channel baseline normalisation

DATA_DIR = "DATA/"
torch.manual_seed(SEED); np.random.seed(SEED)

# Training experiments (N2 only, for weights)
TRAIN_EXPS = [
    ("TEST.csv", ",", 0,  696, 3396, "Sens1-20000ppm"),
    ("TEST.csv", ",", 5,  984, 6300, "Sens1-2000ppm"),
]

# Test experiments (weights pre-trained; threshold uses own N2 first CALIB_WIN s)
TEST_EXPS = [
    ("Sens1_200ppm.csv",  ",", 0, 1150, 10500, "Sens1-200ppm  [unseen concentration]"),
    ("Sens2_2000ppm.csv", ",", 0,  936,  3600, "Sens2-2000ppm [unseen sensor]"),
]

# ── helpers ────────────────────────────────────────────────────────────────
def load_raw(fpath, sep, col_offset):
    df = pd.read_csv(fpath, sep=sep, header=None)
    o  = col_offset
    t      = pd.to_numeric(df.iloc[:, o+0], errors="coerce").values
    lam_ref= pd.to_numeric(df.iloc[:, o+1], errors="coerce").values
    i_ref  = pd.to_numeric(df.iloc[:, o+2], errors="coerce").values
    lam_gas = pd.to_numeric(df.iloc[:, o+3], errors="coerce").values
    i_gas   = pd.to_numeric(df.iloc[:, o+4], errors="coerce").values
    mask = np.isfinite(t)&np.isfinite(i_ref)&np.isfinite(i_gas)&\
           np.isfinite(lam_ref)&np.isfinite(lam_gas)
    t,i_ref,i_gas,lam_ref,lam_gas = t[mask],i_ref[mask],i_gas[mask],lam_ref[mask],lam_gas[mask]
    t_new = np.arange(t[0], t[-1], DT)
    def rs(s):
        s2 = interp1d(t, s, kind="linear", fill_value="extrapolate")(t_new)
        return savgol_filter(s2, 21, 3)
    return t_new, rs(i_gas), rs(i_ref), rs(lam_gas), rs(lam_ref)

def build_feats(i_gas, i_ref, lam_gas, lam_ref):
    ratio = i_gas / (i_ref + 1e-12)
    dlam  = lam_gas - lam_ref
    return np.stack([i_gas, i_ref, lam_gas, lam_ref, ratio, dlam], axis=1)

def bl_norm(feat, n_bl=N_BL):
    """Per-channel fractional deviation from initial baseline median."""
    out = np.empty_like(feat)
    for c in range(feat.shape[1]):
        med = np.median(feat[:n_bl, c])
        out[:, c] = (feat[:, c] - med) / (abs(med) + 1e-9)
    return out

def make_windows(feat):
    return np.array([feat[i:i+WIN_LEN]
                     for i in range(0, len(feat)-WIN_LEN+1, STEP)],
                    dtype=np.float32)

# ── Model ──────────────────────────────────────────────────────────────────
class LSTMAe(nn.Module):
    def __init__(self):
        super().__init__()
        self.win        = WIN_LEN
        self.enc_lstm   = nn.LSTM(N_FEAT, HIDDEN, batch_first=True)
        self.enc_linear = nn.Linear(HIDDEN, LATENT)
        self.dec_linear = nn.Linear(LATENT, HIDDEN)
        self.dec_lstm   = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.dec_out    = nn.Linear(HIDDEN, N_FEAT)
    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_linear(h[0])
        d = self.dec_linear(z).unsqueeze(1).repeat(1, self.win, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_out(d)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Collect N2 from training experiments, fit global scaler
# ══════════════════════════════════════════════════════════════════════════
print("="*64)
print("CROSS-CONCENTRATION / CROSS-SENSOR VALIDATION")
print("  Pre-train weights on : Sens1-20000ppm + Sens1-2000ppm  (N2 only)")
print("  Test (unseen) on     : Sens1-200ppm  &  Sens2-2000ppm")
print("  Calibration threshold: each test experiment's own first 300s N2")
print("="*64)

all_n2 = []
for fpath, sep, o, onset_s, h2_end_s, label in TRAIN_EXPS:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(DATA_DIR+fpath, sep, o)
    feat  = build_feats(i_gas, i_ref, lam_gas, lam_ref)
    fnorm = bl_norm(feat)
    i_onset = int((onset_s  - t[0]) / DT)
    i_post  = int((h2_end_s - t[0]) / DT) + int(N2_SKIP_S / DT)
    i_post  = min(i_post, len(t))
    n2 = np.concatenate([fnorm[:i_onset],
                         fnorm[i_post:] if i_post < len(fnorm)
                         else np.zeros((0, N_FEAT))], axis=0)
    all_n2.append(n2)
    print(f"  {label}: {len(n2)} N2 samples  ({len(n2)*DT/60:.1f} min)")

n2_pool = np.concatenate(all_n2, axis=0)
scaler  = StandardScaler().fit(n2_pool)
n2_sc   = scaler.transform(n2_pool).astype(np.float32)

W   = make_windows(n2_sc)
idx = np.random.permutation(len(W))
ntr = int(0.9 * len(W))
Wtr, Wvl = W[idx[:ntr]], W[idx[ntr:]]
print(f"\n  Pooled N2: {len(n2_pool)} samples | Windows: {len(W)} "
      f"(train={len(Wtr)}, val={len(Wvl)})")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Train LSTM-AE on pooled N2 (weights only)
# ══════════════════════════════════════════════════════════════════════════
def to_loader(arr, sh=True):
    return DataLoader(TensorDataset(torch.tensor(arr)),
                      batch_size=BATCH_SIZE, shuffle=sh)

tr_ld = to_loader(Wtr); vl_ld = to_loader(Wvl, False)
torch.manual_seed(SEED)
model = LSTMAe()
opt   = torch.optim.Adam(model.parameters(), lr=LR)
crit  = nn.MSELoss()
best_val, best_st = float("inf"), None

print("\n  Training on pooled N2 ...")
for ep in range(1, EPOCHS+1):
    model.train(); tl = 0.0
    for (xb,) in tr_ld:
        p = model(xb); loss = crit(p, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * len(xb)
    tl /= len(Wtr)
    model.eval(); vl = 0.0
    with torch.no_grad():
        for (xb,) in vl_ld:
            vl += crit(model(xb), xb).item() * len(xb)
    vl /= max(len(Wvl), 1)
    if vl < best_val:
        best_val = vl
        best_st  = {k: v.clone() for k, v in model.state_dict().items()}
    if ep % 50 == 0 or ep == 1:
        print(f"    ep {ep:3d}  train={tl:.2e}  val={vl:.2e}")

model.load_state_dict(best_st)
print(f"  Best val MSE: {best_val:.2e}  (on pooled N2)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Apply pre-trained model to unseen test experiments
#         Only the calibration threshold uses own N2 (first CALIB_WIN s)
# ══════════════════════════════════════════════════════════════════════════
results = []
for fpath, sep, o, onset_s, h2_end_s, label in TEST_EXPS:
    print(f"\n{'─'*60}")
    print(f"  TEST: {label}   onset={onset_s}s")
    print(f"  [Model weights: pre-trained on Sens1-20000ppm + Sens1-2000ppm N2]")
    print(f"  [Calibration threshold: this experiment's own first {CALIB_WIN}s N2]")

    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(DATA_DIR+fpath, sep, o)
    feat  = build_feats(i_gas, i_ref, lam_gas, lam_ref)
    fnorm = bl_norm(feat)
    full_sc = scaler.transform(fnorm).astype(np.float32)

    model.eval()
    errors = np.full(len(t), np.nan)
    with torch.no_grad():
        for i in range(0, len(full_sc)-WIN_LEN+1, STEP):
            win = torch.tensor(full_sc[i:i+WIN_LEN]).unsqueeze(0)
            errors[i+WIN_LEN-1] = float(((model(win)-win)**2).mean())

    err_s = (pd.Series(errors).interpolate()
               .rolling(int(30/DT), min_periods=1).median().values)

    # Calibration: use THIS experiment's own first CALIB_WIN s N2
    n_cal     = int(CALIB_WIN / DT)
    cal_vals  = err_s[WIN_LEN:n_cal]
    cal_vals  = cal_vals[np.isfinite(cal_vals)]
    cal_mean  = float(np.mean(cal_vals))
    threshold = cal_mean * RECON_MULT
    print(f"  Calib mean MSE (own N2): {cal_mean:.2e}  "
          f"threshold ({RECON_MULT}x): {threshold:.2e}")

    flagged   = (err_s > threshold).astype(float)
    n_per     = int(PERSIST_WIN / DT)
    frac      = pd.Series(flagged).rolling(n_per).mean().values

    alarm_idx = None
    for i in range(n_cal, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= PERSIST_FRAC:
            alarm_idx = i; break

    if alarm_idx is not None:
        alarm_t = t[alarm_idx]
        det_err = alarm_t - onset_s
        print(f"  Alarm: {alarm_t:.1f}s  |  onset={onset_s}s  |  "
              f"latency={det_err:+.1f}s  "
              f"({'EARLY - FALSE ALARM' if det_err < 0 else 'late'})")
    else:
        print("  No alarm triggered.")

    ratio_plot = i_gas / (i_ref + 1e-12)
    results.append(dict(label=label, t=t, ratio=ratio_plot, err_s=err_s,
                        threshold=threshold, onset_true=onset_s,
                        alarm_idx=alarm_idx))

# ══════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════
n  = len(results)
fig = plt.figure(figsize=(13, 5*n))
gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.5)

for k, res in enumerate(results):
    ax  = fig.add_subplot(gs[k])
    ax2 = ax.twinx()
    t_m = res["t"] / 60
    bl  = np.median(res["ratio"][:N_BL])
    rn  = (res["ratio"] - bl) / (abs(bl) + 1e-9)

    ax.plot(t_m, rn, color="#2166ac", lw=0.9, alpha=0.7,
            label="Normalised ratio")
    ax2.plot(t_m, res["err_s"], color="#d6604d", lw=1.3,
             label="Recon MSE")
    ax2.axhline(res["threshold"], color="#b2182b", ls="--", lw=1.2,
                label=f"Threshold ({RECON_MULT}x calib)")
    ax.axvspan(0, CALIB_WIN/60, alpha=0.1, color="grey",
               label="Calibration zone")
    ax.axvline(res["onset_true"]/60, color="#4dac26", ls="--", lw=1.5,
               label=f"True onset ({res['onset_true']}s)")
    if res["alarm_idx"] is not None:
        at = res["t"][res["alarm_idx"]]
        ax.axvline(at/60, color="#d01c8b", ls=":", lw=2,
                   label=f"Alarm ({at:.0f}s, {at-res['onset_true']:+.0f}s)")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Normalised ratio", color="#2166ac")
    ax2.set_ylabel("Recon MSE", color="#d6604d")
    ax.set_title(f"{res['label']}\n"
                 f"[Weights: pre-trained on Sens1-20000ppm+Sens1-2000ppm N2  |  "
                 f"Threshold: own first {CALIB_WIN}s N2]", fontsize=9)
    l1,lb1 = ax.get_legend_handles_labels()
    l2,lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)

fig.suptitle("Cross-Concentration & Cross-Sensor Validation\n"
             "LSTM-AE weights pre-trained on 20000+2000 ppm N2 — "
             "applied to unseen 200 ppm & Sens2 sensor",
             fontsize=10, y=1.01)
plt.savefig("cross_concentration_result.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("\nPlot saved -> cross_concentration_result.png")
