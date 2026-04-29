"""
LSTM Autoencoder — FBG Gas Leak Detection
================================================
Trains a per-experiment LSTM-AE on pre-Gas N2 baseline only,
then detects Gas leaks via reconstruction MSE threshold.

Usage
-----
  # Run with defaults (reads experiments.json):
  python h2_paper_figure.py

  # Custom parameters:
  python h2_paper_figure.py --epochs 300 --threshold-mult 3.0 --output my_result.png

  # Custom experiment config:
  python h2_paper_figure.py --config my_experiments.json

  # Full help:
  python h2_paper_figure.py --help

Training data: pre-Gas N2 ONLY (t=0 to onset_s).
No post-Gas data is used — mirrors real deployment where only
the initial baseline gas segment is available before injection.
"""

import argparse
import json
import sys
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


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="LSTM-AE gas leak detection for FBG sensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    p.add_argument(
        "--data-dir", default="DATA/",
        help="Directory containing the CSV data files",
    )
    p.add_argument(
        "--config", default="experiments.json",
        help="JSON file defining the experiment list (see experiments.json for format)",
    )

    # ── Detection logic ──
    p.add_argument(
        "--window-sec", type=float, default=30.0,
        help="Sliding window length in seconds (converted to samples at DT=0.5 s)",
    )
    p.add_argument(
        "--calib-sec", type=float, default=300.0,
        help="Duration of the N2 calibration zone at the start of each experiment (seconds)",
    )
    p.add_argument(
        "--threshold-mult", type=float, default=2.5,
        help="Alarm threshold = calib_mean_MSE × threshold-mult",
    )
    p.add_argument(
        "--persist-sec", type=float, default=60.0,
        help="Persistence confirmation window length (seconds)",
    )
    p.add_argument(
        "--persist-frac", type=float, default=0.65,
        help="Fraction of the persistence window that must be flagged to trigger an alarm [0–1]",
    )

    # ── Model architecture ──
    p.add_argument(
        "--hidden", type=int, default=64,
        help="LSTM hidden state size",
    )
    p.add_argument(
        "--latent", type=int, default=16,
        help="Bottleneck latent vector dimension",
    )

    # ── Training ──
    p.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam optimiser learning rate",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # ── Output ──
    p.add_argument(
        "--output", default="detection_result.png",
        help="Filename for the output figure (PNG)",
    )
    p.add_argument(
        "--summary", default="detection_summary.txt",
        help="Filename for the plain-text detection latency summary",
    )

    return p.parse_args()


# ── Constants (fixed by sensor physics / data format) ─────────────────────
DT     = 0.5   # resampling interval (s) — matches sensor acquisition rate
N_FEAT = 6     # number of input channels (fixed by feature set)
STEP   = 5     # window stride (samples)


# ── Load experiment config ─────────────────────────────────────────────────
def load_experiments(config_path):
    """
    Load experiment list from a JSON file.

    Expected format:
    [
      {
        "file":          "TEST.csv",
        "sep":           ",",
        "col_offset":    0,
        "onset_s":       696,
        "h2_end_s":      3396,
        "concentration": "20 000 ppm",
        "sensor":        "Sens1"
      },
      ...
    ]
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            exps = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        print("  Create an experiments.json file or use --config to specify another path.")
        print("  See the bundled experiments.json for the required format.")
        sys.exit(1)

    required = {"file", "sep", "col_offset", "onset_s", "h2_end_s",
                "concentration", "sensor"}
    for i, e in enumerate(exps):
        missing = required - e.keys()
        if missing:
            print(f"[ERROR] experiments.json entry {i} is missing keys: {missing}")
            sys.exit(1)

    return [
        (e["file"], e["sep"], e["col_offset"],
         e["onset_s"], e["h2_end_s"],
         e["concentration"], e["sensor"])
        for e in exps
    ]


# ── I/O ────────────────────────────────────────────────────────────────────
def load_raw(fpath, sep=",", col_offset=0):
    df = pd.read_csv(fpath, sep=sep, header=None)
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


def make_windows(feat, win_len):
    return np.array([feat[i:i+win_len]
                     for i in range(0, len(feat)-win_len+1, STEP)],
                    dtype=np.float32)


# ── Model ──────────────────────────────────────────────────────────────────
class LSTMAe(nn.Module):
    def __init__(self, win_len, hidden, latent):
        super().__init__()
        self.win        = win_len
        self.enc_lstm   = nn.LSTM(N_FEAT, hidden, batch_first=True)
        self.enc_linear = nn.Linear(hidden, latent)
        self.dec_linear = nn.Linear(latent, hidden)
        self.dec_lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.dec_out    = nn.Linear(hidden, N_FEAT)

    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_linear(h[0])
        d = self.dec_linear(z).unsqueeze(1).repeat(1, self.win, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_out(d)


# ── Per-experiment detection ───────────────────────────────────────────────
def run(fpath, sep, col_offset, onset_s, h2_end_s, conc, sensor, args):
    win_len     = int(args.window_sec / DT)
    calib_win   = int(args.calib_sec  / DT)

    label = f"{sensor} — {conc}"
    print(f"\n{'─'*56}")
    print(f"  {label}   onset={onset_s}s")

    t, i_gas, i_ref, lam_gas, lam_ref = load_raw(
        args.data_dir + fpath, sep=sep, col_offset=col_offset)
    feat = build_feats(i_gas, i_ref, lam_gas, lam_ref)

    # Training data: pre-Gas N2 ONLY
    i_onset = int((onset_s - t[0]) / DT)
    n2      = feat[:i_onset]

    scaler  = StandardScaler().fit(n2)
    n2_sc   = scaler.transform(n2).astype(np.float32)
    full_sc = scaler.transform(feat).astype(np.float32)

    W   = make_windows(n2_sc, win_len)
    idx = np.random.permutation(len(W))
    ntr = int(0.9 * len(W))
    Wtr, Wvl = W[idx[:ntr]], W[idx[ntr:]]

    def loader(arr, sh=True):
        return DataLoader(TensorDataset(torch.tensor(arr)),
                          batch_size=args.batch_size, shuffle=sh)

    tr_ld = loader(Wtr); vl_ld = loader(Wvl, False)

    torch.manual_seed(args.seed)
    model = LSTMAe(win_len, args.hidden, args.latent)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit  = nn.MSELoss()
    best_val, best_st = float("inf"), None

    for ep in range(1, args.epochs + 1):
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

    # Reconstruction error over full experiment
    model.eval()
    errors = np.full(len(t), np.nan)
    with torch.no_grad():
        for i in range(0, len(full_sc) - win_len + 1, STEP):
            win = torch.tensor(full_sc[i:i+win_len]).unsqueeze(0)
            errors[i+win_len-1] = float(((model(win)-win)**2).mean())

    err_s = (pd.Series(errors).interpolate()
               .rolling(int(30/DT), min_periods=1).median().values)

    # Calibration threshold
    cal_vals = err_s[win_len:calib_win]
    cal_vals = cal_vals[np.isfinite(cal_vals)]
    cal_mean = float(np.mean(cal_vals))
    thr      = cal_mean * args.threshold_mult

    # Persistence-confirmed alarm
    flagged  = (err_s > thr).astype(float)
    n_per    = int(args.persist_sec / DT)
    frac     = pd.Series(flagged).rolling(n_per).mean().values

    alarm_idx = None
    for i in range(calib_win, len(t)):
        if np.isfinite(frac[i]) and frac[i] >= args.persist_frac:
            alarm_idx = i; break

    if alarm_idx is not None:
        alarm_t = t[alarm_idx]
        det_err = alarm_t - onset_s
        print(f"  → Alarm {alarm_t:.0f}s  (onset {onset_s}s  |  Δt={det_err:+.0f}s)")
    else:
        print("  → No alarm")

    ratio_raw = i_gas / (i_ref + 1e-12)
    return dict(label=label, conc=conc, sensor=sensor,
                t=t, ratio=ratio_raw, err_s=err_s,
                threshold=thr, cal_mean=cal_mean,
                onset_true=onset_s, alarm_idx=alarm_idx,
                win_len=win_len, calib_win=calib_win)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate range constraints
    if not (0.0 < args.persist_frac <= 1.0):
        print("[ERROR] --persist-frac must be in (0, 1]"); sys.exit(1)
    if args.threshold_mult <= 0:
        print("[ERROR] --threshold-mult must be positive"); sys.exit(1)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    experiments = load_experiments(args.config)

    print("=" * 56)
    print("LSTM Autoencoder — FBG Gas Leak Detection")
    print(f"  window     : {args.window_sec:.0f} s")
    print(f"  calib      : {args.calib_sec:.0f} s")
    print(f"  threshold  : {args.threshold_mult}× calib mean")
    print(f"  persistence: {args.persist_sec:.0f} s  ×  {args.persist_frac:.0%}")
    print(f"  model      : LSTM({args.hidden}) → z({args.latent})")
    print(f"  epochs     : {args.epochs}  |  lr={args.lr}  |  batch={args.batch_size}")
    print("=" * 56)

    results = []
    for exp in experiments:
        results.append(run(*exp, args=args))

    # ── Publication figure ────────────────────────────────────────────────
    n = len(results)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(7 * ncols, 4.5 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.45, wspace=0.35)

    C_SIGNAL = "#2166ac"
    C_MSE    = "#d6604d"
    C_THR    = "#b2182b"
    C_ONSET  = "#4dac26"
    C_ALARM  = "#d01c8b"

    for k, res in enumerate(results):
        row, col = divmod(k, ncols)
        ax  = fig.add_subplot(gs[row, col])
        ax2 = ax.twinx()

        t_min   = res["t"] / 60
        bl      = np.median(res["ratio"][:res["calib_win"]])
        ratio_n = (res["ratio"] - bl) / (abs(bl) + 1e-9)

        ax.plot(t_min, ratio_n, color=C_SIGNAL, lw=0.9, alpha=0.75,
                label=r"Normalised ratio $I_{Gas}/I_{Ref}$")
        ax2.plot(t_min, res["err_s"], color=C_MSE, lw=1.3,
                 label="Reconstruction MSE")
        ax2.axhline(res["threshold"], color=C_THR, ls="--", lw=1.2,
                    label=f"Threshold ({args.threshold_mult}× calib. mean)")
        ax.axvspan(0, args.calib_sec/60, alpha=0.12, color="grey",
                   label="Calibration zone")
        ax.axvline(res["onset_true"]/60, color=C_ONSET, ls="--", lw=1.5,
                   label=f"True Gas onset ({res['onset_true']} s)")

        if res["alarm_idx"] is not None:
            alarm_t = res["t"][res["alarm_idx"]]
            det_err = alarm_t - res["onset_true"]
            ax.axvline(alarm_t/60, color=C_ALARM, ls=":", lw=2.0,
                       label=f"Alarm ({alarm_t:.0f} s, Δt={det_err:+.0f} s)")

        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel("Normalised ratio (a.u.)", color=C_SIGNAL, fontsize=8)
        ax2.set_ylabel("Reconstruction MSE", color=C_MSE, fontsize=8)
        ax.set_title(f"({chr(97+k)})  {res['label']}",
                     fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(axis="y", labelcolor=C_SIGNAL, labelsize=8)
        ax2.tick_params(axis="y", labelcolor=C_MSE,   labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=7, loc="upper left",
                  ncol=1, framealpha=0.85)
        ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle(
        "LSTM Autoencoder Gas Leak Detection — 6-Feature Multi-Channel Analysis\n"
        r"Features: $I_{Gas}$, $I_{Ref}$, $\lambda_{Gas}$, $\lambda_{Ref}$, "
        r"$I_{Gas}/I_{Ref}$, $\Delta\lambda$  |  "
        f"Trained on pre-Gas N₂ only  |  "
        f"WIN={args.window_sec:.0f} s  PERSIST={args.persist_sec:.0f} s  "
        f"MULT={args.threshold_mult}  FRAC={args.persist_frac}",
        fontsize=9.5, y=1.02)

    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved → {args.output}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("DETECTION SUMMARY")
    print(f"{'Experiment':<22} {'Conc.':>9} {'Onset':>8} "
          f"{'Alarm':>8} {'Latency':>9}")
    print("─" * 64)

    lines = []
    for res in results:
        if res["alarm_idx"] is not None:
            alarm_t = res["t"][res["alarm_idx"]]
            lat     = alarm_t - res["onset_true"]
            row = (f"  {res['label']:<20} {res['conc']:>9}  "
                   f"{res['onset_true']:>6} s  {alarm_t:>6.0f} s  "
                   f"{lat:>+7.0f} s")
        else:
            row = (f"  {res['label']:<20} {res['conc']:>9}  "
                   f"{res['onset_true']:>6} s       --       --")
        print(row); lines.append(row)

    lats = [res["t"][res["alarm_idx"]] - res["onset_true"]
            for res in results if res["alarm_idx"] is not None]
    if lats:
        print("─" * 64)
        print(f"  {'Mean latency':>42}  {np.mean(lats):>+7.1f} s")
        print(f"  {'Min  latency':>42}  {np.min(lats):>+7.1f} s")
        print(f"  {'Max  latency':>42}  {np.max(lats):>+7.1f} s")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("LSTM-AE — Gas Detection Results\n")
        f.write(f"window={args.window_sec:.0f}s  calib={args.calib_sec:.0f}s  "
                f"mult={args.threshold_mult}  persist={args.persist_sec:.0f}s  "
                f"frac={args.persist_frac}  epochs={args.epochs}\n")
        f.write("Features: I_Gas, I_Ref, lam_Gas, lam_Ref, ratio, delta_lam\n")
        f.write("Training: pre-Gas N2 only (t=0 to onset)\n\n")
        f.write(f"{'Experiment':<22} {'Conc':>9} {'Onset':>8} "
                f"{'Alarm':>8} {'Latency':>9}\n")
        f.write("─" * 60 + "\n")
        for line in lines:
            f.write(line + "\n")
        if lats:
            f.write("─" * 60 + "\n")
            f.write(f"  Mean latency: {np.mean(lats):+.1f} s\n")
            f.write(f"  Min  latency: {np.min(lats):+.1f} s\n")
            f.write(f"  Max  latency: {np.max(lats):+.1f} s\n")

    print(f"Summary saved → {args.summary}")


if __name__ == "__main__":
    main()
