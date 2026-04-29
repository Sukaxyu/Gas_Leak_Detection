"""
Leave-One-Out Cross-Experiment Validation
==========================================
Training  : N2 segments from Sens1-20000ppm, Sens1-2000ppm, Sens1-200ppm
Test      : Sens1_200ppm_2  (completely unseen — model never trained on it)

Key challenge
  Each experiment has different absolute signal levels (different sensor
  coupling, fibre state, etc.).  We cannot feed raw intensities from three
  different measurements into one StandardScaler.

Solution: per-channel baseline normalisation
  For every experiment, divide each channel by its own initial-median baseline:
      ch_norm = (ch - median(ch[:N_BL])) / (|median(ch[:N_BL])| + ε)
  This converts absolute values → fractional deviations from each channel's
  own starting point.  All experiments now live in the same ±δ space.
  ONE global StandardScaler is then fitted on the pooled normalised N2 data.

Onset of Sens1_200ppm_2 (from user): ~984 s
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

# ── Hyper-parameters (MODERATE, same as main script) ──────────────────────
DT           = 0.5
WIN_LEN      = 60       # 30 s
STEP         = 5        # 2.5 s stride
HIDDEN       = 64
LATENT       = 16
EPOCHS       = 200
BATCH_SIZE   = 64
LR           = 1e-3
SEED         = 42

CALIB_WIN    = 300      # s
N_BL         = int(300 / DT)   # baseline window for per-channel normalisation
RECON_MULT   = 2.5
PERSIST_WIN  = 60       # s
PERSIST_FRAC = 0.65
N2_SKIP_S    = 120      # s to skip after Gas → N2 switch

N_FEAT       = 6
FEAT_NAMES   = ["I_Gas", "I_Ref", "λ_Gas", "λ_Ref", "ratio", "Δλ"]

DATA_DIR     = "DATA/"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Training experiments (N2 used only) ────────────────────────────────────
TRAIN_EXPS = [
    ("TEST.csv",       ",", 0,  696,  3396,  "Sens1-20000ppm"),
    ("TEST.csv",       ",", 5,  984,  6300,  "Sens1-2000ppm"),
    ("Sens1_200ppm.csv", ",", 0, 1150, 10500,  "Sens1-200ppm"),
]

# ── Test experiment (unseen) ───────────────────────────────────────────────
TEST_EXP = ("Sens1_200ppm_2.csv", ",", 0, 984, None, "Sens1-200ppm-2 (unseen)")


# ── I/O helpers ───────────────────────────────────────────────────────────
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
        s2 = interp1d(t, s, kind="linear", fill_value="extrapolate")(t_new)
        return savgol_filter(s2, window_length=21, polyorder=3)
    return t_new, rs(i_gas), rs(i_ref), rs(lam_gas), rs(lam_ref)


def build_feats(i_gas, i_ref, lam_gas, lam_ref):
    ratio = i_gas / (i_ref + 1e-12)
    dlam  = lam_gas - lam_ref
    return np.stack([i_gas, i_ref, lam_gas, lam_ref, ratio, dlam], axis=1)


def bl_normalise(feat_mat, n_bl=N_BL):
    """Per-channel normalisation: (x − median_baseline) / |median_baseline|."""
    out = np.empty_like(feat_mat)
    for c in range(feat_mat.shape[1]):
        med = np.median(feat_mat[:n_bl, c])
        out[:, c] = (feat_mat[:, c] - med) / (abs(med) + 1e-9)
    return out


def make_windows(feat_mat):
    segs = []
    for i in range(0, len(feat_mat) - WIN_LEN + 1, STEP):
        segs.append(feat_mat[i : i+WIN_LEN])
    return np.array(segs, dtype=np.float32)


# ── LSTM Autoencoder (same architecture as main script) ───────────────────
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


# ── Step 1 : Collect N2 windows from all training experiments ─────────────
print("=" * 64)
print("LOO CROSS-EXPERIMENT VALIDATION")
print("  Train on : Sens1-20000ppm, Sens1-2000ppm, Sens1-200ppm  (N2 only)")
print("  Test  on : Sens1-200ppm-2  (completely unseen)")
print("=" * 64)

all_n2_norm = []          # list of baseline-normalised N2 feature arrays

for fpath, sep, offset, onset_s, h2_end_s, label in TRAIN_EXPS:
    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
        DATA_DIR + fpath, sep=sep, col_offset=offset)
    feat  = build_feats(i_gas, i_ref, lam_gas, lam_ref)     # (N, 6)
    fnorm = bl_normalise(feat)                              # fractional deviation

    # N2 segments
    i_onset = int((onset_s  - t[0]) / DT)
    i_post  = int((h2_end_s - t[0]) / DT) + int(N2_SKIP_S / DT)
    i_post  = min(i_post, len(t))
    pre_n2  = fnorm[:i_onset]
    post_n2 = fnorm[i_post:] if i_post < len(fnorm) else np.zeros((0, N_FEAT))
    n2      = np.concatenate([pre_n2, post_n2], axis=0)
    all_n2_norm.append(n2)

    print(f"  {label:<16}  N2 samples: {len(n2):6d}  "
          f"({len(n2)*DT/60:.1f} min)")

# ── Step 2 : Fit a single global StandardScaler on pooled N2 ──────────────
n2_pool = np.concatenate(all_n2_norm, axis=0)
scaler  = StandardScaler()
scaler.fit(n2_pool)
print(f"\n  Pooled N2 : {len(n2_pool)} samples  ({len(n2_pool)*DT/60:.1f} min)")
print(f"  Global StandardScaler fitted on pooled N2.")

# Build training windows
n2_scaled = scaler.transform(n2_pool).astype(np.float32)
W = make_windows(n2_scaled)
idx  = np.random.permutation(len(W))
ntr  = int(0.9 * len(W))
Wtr  = W[idx[:ntr]]
Wvl  = W[idx[ntr:]]
print(f"  Training windows : {len(W)}  (train={len(Wtr)}, val={len(Wvl)})")

# ── Step 3 : Train ────────────────────────────────────────────────────────
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

print("\n  Training …")
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
        print(f"    Epoch {ep:3d}/{EPOCHS}  train={tr_loss:.2e}  val={vl_loss:.2e}")

model.load_state_dict(best_state)
print(f"  Best val MSE : {best_val:.2e}")

# ── Step 4 : Apply to unseen Sens1_200ppm_2 ────────────────────────────────
fpath, sep, offset, onset_s, _, label = TEST_EXP
print(f"\n{'─'*64}")
print(f"  Applying to UNSEEN experiment: {label}")
print(f"  True onset : {onset_s} s")

t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
    DATA_DIR + fpath, sep=sep, col_offset=offset)
feat  = build_feats(i_gas, i_ref, lam_gas, lam_ref)
fnorm = bl_normalise(feat)                        # same per-channel normalisation
full_scaled = scaler.transform(fnorm).astype(np.float32)   # global scaler

# Reconstruction error on full trace
model.eval()
errors = np.full(len(t), np.nan)
with torch.no_grad():
    for i in range(0, len(full_scaled) - WIN_LEN + 1, STEP):
        win = torch.tensor(full_scaled[i:i+WIN_LEN]).unsqueeze(0)
        err = float(((model(win) - win) ** 2).mean())
        errors[i + WIN_LEN - 1] = err

err_s = (pd.Series(errors).interpolate()
           .rolling(int(30 / DT), min_periods=1).median().values)

# Calibration (on own initial N2)
n_calib    = int(CALIB_WIN / DT)
calib_vals = err_s[WIN_LEN : n_calib]
calib_vals = calib_vals[np.isfinite(calib_vals)]
calib_mean = float(np.mean(calib_vals))
calib_p90  = float(np.percentile(calib_vals, 90))
# LOO: use 90th-percentile × RECON_MULT as threshold
# The cross-experiment noise floor is wider than per-experiment,
# so we anchor to p90 (not mean) to avoid false positives in N2.
threshold  = calib_p90 * RECON_MULT
print(f"  Calib mean error : {calib_mean:.2e}  "
      f"p90 : {calib_p90:.2e}  "
      f"threshold ({RECON_MULT}× p90) : {threshold:.2e}")

# Detection
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
    err_det = alarm_t - onset_s
    print(f"  Alarm time      : {alarm_t:.1f} s  ({alarm_t/60:.1f} min)")
    print(f"  True onset      : {onset_s} s  ({onset_s/60:.1f} min)")
    print(f"  Detection error : {err_det:+.1f} s  "
          f"({'early' if err_det < 0 else 'late'})")
else:
    print("  No alarm triggered.")
print(f"  {'─'*40}")

# ── Step 5 : Plot ─────────────────────────────────────────────────────────
ratio_plot = i_gas / (i_ref + 1e-12)

fig, ax = plt.subplots(figsize=(13, 5))
ax2 = ax.twinx()

ax.plot(t / 60, ratio_plot, color="steelblue", lw=0.8, alpha=0.7,
        label="I_Gas / I_Ref  (ratio)")
ax2.plot(t / 60, err_s, color="tomato", lw=1.2,
         label="Reconstruction MSE (6-feature)")
ax2.axhline(threshold, color="red", ls="--", lw=1,
            label=f"Threshold ({RECON_MULT}× p90 calib)")
ax.axvspan(0, CALIB_WIN / 60, alpha=0.07, color="gray",
           label="Calibration zone")
ax.axvline(onset_s / 60, color="green", ls="--", lw=1.5,
           label=f"True onset ({onset_s}s)")

if alarm_idx is not None:
    ax.axvline(alarm_t / 60, color="red", ls=":", lw=1.5,
               label=f"Alarm ({alarm_t:.0f}s, {alarm_t-onset_s:+.0f}s)")

ax.set_xlabel("Time (min)")
ax.set_ylabel("I_Gas / I_Ref", color="steelblue")
ax2.set_ylabel("Recon MSE (standardised)", color="tomato")
ax.set_title(
    "LOO Validation — Sens1-200ppm-2 (unseen)\n"
    "Model trained on N2 from Sens1-20000ppm + Sens1-2000ppm + Sens1-200ppm",
    fontsize=11)

l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loo_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → loo_result.png")
