"""
Gas Leak Detection — LSTM Autoencoder (One-Class Anomaly Detection)
===================================================================
Principle
---------
  Train an LSTM Autoencoder exclusively on NORMAL (N2) signal windows.
  The model compresses each 60-second window into a low-dimensional
  latent code and then reconstructs the original signal.  Because it
  has only seen N2, it reconstructs N2 accurately (small error).
  When Gas is introduced the signal character changes; the model
  cannot reconstruct it well → reconstruction MSE spikes → alarm.

N2 training data (both pre- and post-Gas segments)
---------
  Pre-Gas  N2: t < h2_onset_s
  Post-Gas N2: t > h2_end_s + N2_SKIP_S   (skip the drop phase after Gas removal)
  Both segments are concatenated as training data.

Detection logic
---------
  1. Calibration : mean recon error on first CALIB_WIN seconds (initial N2).
  2. Flag        : error > CALIB_MEAN * RECON_MULT  AND  signal is rising
                   (directionality: drops from N2 purge do NOT trigger alarm).
  3. Confirm     : ≥ PERSIST_FRAC of PERSIST_WIN-second rolling window flagged.
  4. Report      : confirmation time as the alarm time (no backtracking).

Experiments
-----------
  TEST.csv   col0  20 000 ppm  Gas: 696 s → 3396 s    post-N2: 3516 s → end
  TEST.csv   col5   2 000 ppm  Gas: 984 s → 6300 s    post-N2: 6420 s → end
  Sens1_200ppm col0     200 ppm  Gas:1150 s →10500 s    post-N2:10620 s → end
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves file without a GUI window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Hyperparameters ────────────────────────────────────────────────────────
DATA_DIR    = "DATA/"
DT          = 0.5       # resample interval (s)
WIN_LEN     = 120       # sliding-window length (timesteps = 60 s)
STEP        = 5         # stride (timesteps = 2.5 s)
HIDDEN      = 32        # LSTM hidden units
LATENT      = 8         # bottleneck size
EPOCHS      = 200
BATCH_SIZE  = 64
LR          = 1e-3
SEED        = 42

# Detection parameters
CALIB_WIN    = 300      # initial N2 window used for calibration (s)
RECON_MULT   = 2.0      # flag when error > CALIB_MEAN * RECON_MULT
PERSIST_WIN  = 120      # rolling persistence window (s)
PERSIST_FRAC = 0.70     # fraction of window that must be flagged
RISE_WIN     = 60       # window (s) for checking signal is rising (not falling)
N2_SKIP_S    = 120      # skip first N2_SKIP_S after Gas→N2 switch (drop phase)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Experiment definitions ─────────────────────────────────────────────────
# (filepath, col_offset, h2_onset_s, h2_end_s, label)
# h2_end_s: when Gas is switched off (N2 purge begins)
EXPERIMENTS = [
    ("TEST.csv",       0,  696,  3396,  "20000 ppm"),  # Gas: 696s, +45min → 3396s
    ("TEST.csv",       5,  984,  6300,  "2000 ppm"),   # Gas: 984s → 105min=6300s
    ("Sens1_200ppm.csv", 0, 1150, 10500,  "200 ppm"),    # Gas: ~1150s → 175min=10500s
]


# ── Signal loading ─────────────────────────────────────────────────────────
def load_ratio(filepath, col_offset=0):
    """Read CSV, compute I_Gas/I_Ref, resample to DT grid, smooth."""
    df    = pd.read_csv(DATA_DIR + filepath, header=None)
    t     = pd.to_numeric(df.iloc[:, col_offset + 0], errors="coerce").values
    i_ref = pd.to_numeric(df.iloc[:, col_offset + 2], errors="coerce").values
    i_gas  = pd.to_numeric(df.iloc[:, col_offset + 4], errors="coerce").values
    ratio = i_gas / i_ref
    mask  = np.isfinite(t) & np.isfinite(ratio)
    t, ratio = t[mask], ratio[mask]
    t_new = np.arange(t[0], t[-1], DT)
    ratio = interp1d(t, ratio, kind="linear", fill_value="extrapolate")(t_new)
    ratio = savgol_filter(ratio, window_length=21, polyorder=3)
    # Normalise to initial baseline
    baseline = np.median(ratio[:int(CALIB_WIN / DT)])
    ratio = (ratio - baseline) / (np.abs(baseline) + 1e-9)
    return t_new, ratio


def make_windows(signal, win=WIN_LEN, step=STEP):
    return np.array([signal[i:i + win] for i in range(0, len(signal) - win + 1, step)],
                    dtype=np.float32)


# ── LSTM Autoencoder ───────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Seq2seq LSTM AE.
      Encoder : LSTM(1→HIDDEN) → last hidden → Linear(HIDDEN→LATENT)
      Decoder : Linear(LATENT→HIDDEN) → repeat → LSTM(HIDDEN→HIDDEN) → Linear(HIDDEN→1)
    """
    def __init__(self):
        super().__init__()
        self.enc_lstm   = nn.LSTM(1, HIDDEN, batch_first=True)
        self.enc_linear = nn.Linear(HIDDEN, LATENT)
        self.dec_linear = nn.Linear(LATENT, HIDDEN)
        self.dec_lstm   = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
        self.dec_out    = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_linear(h[0])
        d = self.dec_linear(z).unsqueeze(1).repeat(1, WIN_LEN, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_out(d)


# ── Per-experiment pipeline ────────────────────────────────────────────────
def run_experiment(fpath, col_offset, onset_true, h2_end_s, label):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {label}  ({fpath}[+{col_offset}])  onset={onset_true}s")

    t, ratio = load_ratio(fpath, col_offset)

    # ── 1. Build N2 training windows: pre-Gas + post-Gas segments ──────────
    # Pre-Gas: everything before Gas onset
    i_onset   = int((onset_true - t[0]) / DT)
    pre_n2    = ratio[:i_onset]

    # Post-Gas: after Gas ends + skip N2_SKIP_S (signal still dropping)
    i_post    = int((h2_end_s - t[0]) / DT) + int(N2_SKIP_S / DT)
    i_post    = min(i_post, len(ratio))
    post_n2   = ratio[i_post:] if i_post < len(ratio) else np.array([])

    # Concatenate and build windows
    n2_signal = np.concatenate([pre_n2, post_n2])
    W = make_windows(n2_signal)

    n_pre  = len(pre_n2)
    n_post = len(post_n2)
    print(f"  Pre-Gas  N2: {n_pre} samples ({n_pre*DT/60:.1f} min)")
    print(f"  Post-Gas N2: {n_post} samples ({n_post*DT/60:.1f} min)")
    print(f"  Total windows: {len(W)}")

    # Shuffle / split 90-10
    idx = np.random.permutation(len(W))
    n_tr = int(0.9 * len(W))
    W_tr, W_vl = W[idx[:n_tr]], W[idx[n_tr:]]

    def to_loader(arr, shuffle=True):
        x = torch.tensor(arr).unsqueeze(-1)
        return DataLoader(TensorDataset(x), batch_size=BATCH_SIZE, shuffle=shuffle)

    tr_loader = to_loader(W_tr)
    vl_loader = to_loader(W_vl, shuffle=False)

    # ── 2. Train ──────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = LSTMAutoencoder()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    best_val, best_state, history = float("inf"), None, {"train": [], "val": []}

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
            print(f"  Epoch {ep:3d}/{EPOCHS}  train={tr_loss:.2e}  val={vl_loss:.2e}")

    model.load_state_dict(best_state)
    print(f"  Best val MSE: {best_val:.2e}")

    # ── 3. Compute reconstruction error on full trace ─────────────────────
    model.eval()
    errors = np.full(len(ratio), np.nan)
    with torch.no_grad():
        for i in range(0, len(ratio) - WIN_LEN + 1, STEP):
            win = torch.tensor(ratio[i:i + WIN_LEN],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            err = float(((model(win) - win) ** 2).mean())
            errors[i + WIN_LEN - 1] = err     # causal: assign to last point

    # Smooth (30 s rolling median)
    err_s = pd.Series(errors).interpolate().rolling(int(30 / DT), min_periods=1).median().values

    # ── 4. Calibration ────────────────────────────────────────────────────
    n_calib = int(CALIB_WIN / DT)
    calib_errors = err_s[WIN_LEN:n_calib]
    calib_errors = calib_errors[np.isfinite(calib_errors)]
    calib_mean   = float(np.mean(calib_errors))
    threshold    = calib_mean * RECON_MULT
    print(f"  Calib mean error: {calib_mean:.2e}  |  threshold ({RECON_MULT}x): {threshold:.2e}")

    # ── 5. Directionality: only flag when signal is actually rising ────────
    # Use 60s rolling slope: positive = rising (Gas), negative = falling (N2 purge)
    rise_w = int(RISE_WIN / DT)
    signal_slope = pd.Series(ratio).diff(rise_w).values   # net change over RISE_WIN
    rising = signal_slope > 0

    # ── 6. Detection: error above threshold AND signal rising ──────────────
    flagged   = ((err_s > threshold) & rising).astype(float)
    n_persist = int(PERSIST_WIN / DT)
    frac      = pd.Series(flagged).rolling(n_persist).mean().values

    alarm_idx = None
    for i in range(n_calib, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= PERSIST_FRAC:
            alarm_idx = i
            break

    print(f"\n  --- DETECTION RESULT ---")
    if alarm_idx is not None:
        alarm_t = t[alarm_idx]
        err_s_detect = alarm_t - onset_true
        print(f"  Alarm confirmed : {alarm_t:.1f} s  ({alarm_t/60:.1f} min)")
        print(f"  True onset      : {onset_true} s  ({onset_true/60:.1f} min)")
        print(f"  Detection error : {err_s_detect:+.1f} s  "
              f"({'early — possible pre-onset sensitivity' if err_s_detect < 0 else 'late'})")
    else:
        print("  No Gas alarm triggered.")

    return dict(label=label, t=t, ratio=ratio, err_s=err_s,
                threshold=threshold, calib_mean=calib_mean,
                onset_true=onset_true, alarm_idx=alarm_idx,
                history=history)


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_all(results):
    n = len(results)
    fig = plt.figure(figsize=(14, 4 * n))
    gs  = gridspec.GridSpec(n, 1, figure=fig, hspace=0.5)

    for k, res in enumerate(results):
        t        = res["t"]
        ratio    = res["ratio"]
        err_s    = res["err_s"]
        thr      = res["threshold"]
        onset_tr = res["onset_true"]
        al_idx   = res["alarm_idx"]

        ax  = fig.add_subplot(gs[k])
        ax2 = ax.twinx()

        ax.plot(t / 60, ratio, color="steelblue", lw=0.8, alpha=0.7,
                label="Signal (normalised ratio)")
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
        ax.set_ylabel("Normalised ratio", color="steelblue")
        ax2.set_ylabel("Recon MSE", color="tomato")
        ax.set_title(f"{res['label']}  —  {res['t'].shape[0]*DT/60:.0f} min total")

        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc="upper left",
                  ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("LSTM Autoencoder — Gas Anomaly Detection (in-deployment calibration)",
                 fontsize=12, y=1.01)
    plt.savefig("autoencoder_result.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → autoencoder_result.png")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LSTM Autoencoder — Gas Leak Detection")
    print("Each experiment trains on its own N2 baseline, then detects onset.\n")

    results = []
    for fpath, offset, onset, h2_end, label in EXPERIMENTS:
        res = run_experiment(fpath, offset, onset, h2_end, label)
        results.append(res)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  {'Experiment':<12} {'True onset':>11} {'Est. onset':>11} {'Error':>8}")
    print("  " + "-" * 46)
    for res in results:
        if res["alarm_idx"] is not None:
            alarm_t = res["t"][res["alarm_idx"]]
            err = alarm_t - res["onset_true"]
            print(f"  {res['label']:<12} {res['onset_true']:>8} s   {alarm_t:>8.0f} s  {err:>+7.0f} s")
        else:
            print(f"  {res['label']:<12} {res['onset_true']:>8} s        -- s      -- s")

    plot_all(results)
