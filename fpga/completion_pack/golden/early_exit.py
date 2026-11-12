#!/usr/bin/env python3
"""
early_exit.py - Confidence-Gated Early Exit (CGEE) analysis.

Idea: the SNN is causal, so the output spike counts after step t are exactly what a hardware run
would have after t steps. We can therefore replay ANY exit policy offline from the recorded
cumulative counts (results_*.npz from snn_golden.py) - no re-simulation needed.

Policy (same rule as rtl/early_exit_ctrl.v):
    after every timestep t:  margin = (best count) - (second best count)
    exit if  t >= t_min  AND  margin >= M      (M >= 1)
    otherwise run to T (=32).   Prediction = argmax of counts, ties -> lowest class index.

Usage:
    python early_exit.py results_fixed_poisson.npz            # sweep + plot + csv
    python early_exit.py --selftest                           # unit test of the logic
"""
import argparse, csv, sys
import numpy as np


def early_exit_eval(cum, labels, t_min, margin):
    """cum:(N,T,C) cumulative output spike counts after each step. Returns (accuracy, mean_steps, steps[N], pred[N])."""
    N, T, C = cum.shape
    assert margin >= 1
    srt = np.sort(cum, axis=2)
    marg = srt[:, :, -1] - srt[:, :, -2]
    ok = (marg >= margin) & (np.arange(1, T + 1)[None, :] >= t_min)
    first = np.where(ok.any(1), ok.argmax(1), T - 1)          # index of exit step (or last step)
    pred = cum[np.arange(N), first].argmax(1)
    steps = first + 1
    return float((pred == labels).mean()), float(steps.mean()), steps, pred


def sweep(cum, labels, t_mins=(1, 2, 4, 6, 8), margins=range(1, 13)):
    base_acc = float((cum[:, -1].argmax(1) == labels).mean())
    T = cum.shape[1]
    rows = []
    for tm in t_mins:
        for m in margins:
            acc, st, _, _ = early_exit_eval(cum, labels, tm, m)
            rows.append(dict(t_min=tm, margin=m, accuracy=acc, mean_steps=st,
                             speedup=T / st, acc_drop_pts=100 * (base_acc - acc)))
    return base_acc, rows


def pareto(rows):
    """Keep points not dominated (higher accuracy AND fewer steps)."""
    pts = sorted(rows, key=lambda r: (r["mean_steps"], -r["accuracy"]))
    best, out = -1, []
    for r in pts:
        if r["accuracy"] > best:
            out.append(r)
            best = r["accuracy"]
    return out


def plot(base_acc, rows, T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    for tm in sorted({r["t_min"] for r in rows}):
        rr = sorted([r for r in rows if r["t_min"] == tm], key=lambda r: r["mean_steps"])
        ax.plot([r["mean_steps"] for r in rr], [100 * r["accuracy"] for r in rr], "o-", ms=3, lw=1, alpha=.55,
                label=f"t_min={tm}")
    p = pareto(rows)
    ax.plot([r["mean_steps"] for r in p], [100 * r["accuracy"] for r in p], "k-", lw=2.2, label="Pareto front")
    ax.axhline(100 * base_acc, color="crimson", ls="--", label=f"full {T}-step baseline ({100*base_acc:.2f}%)")
    ax.set_xlabel("average timesteps used per image (lower = faster / less energy)")
    ax.set_ylabel("CIFAR-10 accuracy (%)")
    ax.set_title("Confidence-Gated Early Exit: accuracy vs. timesteps")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def selftest():
    # 3 images, 3 classes, T=6. Hand-computed expectations.
    cum = np.array([
        [[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0]],      # clear class 0: margin hits 2 at t=2
        [[0,0,0],[0,1,0],[1,2,0],[1,3,1],[2,3,1],[2,4,2]],      # class 1, margin 2 first at t=3 (2 vs 1?) -> see below
        [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],      # silent: never exits, pred=class 0 (tie->lowest)
    ])
    labels = np.array([0, 1, 2])
    acc, mean_steps, steps, pred = early_exit_eval(cum, labels, t_min=1, margin=2)
    # image0: margins 0,1,2 -> exit t=3?  counts after t=3 are [2,0,0]: margin 2 -> step 3
    # image1: margins 0,1,1,2(3-1) -> t=4 has [1,3,1] margin 2 -> step 4
    assert list(steps) == [3, 4, 6], steps
    assert list(pred) == [0, 1, 0], pred
    assert abs(acc - 2 / 3) < 1e-9 and abs(mean_steps - 13 / 3) < 1e-9
    _, _, steps_t, _ = early_exit_eval(cum, labels, t_min=5, margin=2)       # t_min delays the exit
    assert list(steps_t) == [5, 6, 6], steps_t   # img1 at t=5 is [2,3,1] (margin 1) -> waits to t=6
    print("early_exit selftest: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prefix", default="early_exit")
    a = ap.parse_args()
    if a.selftest or not a.npz:
        selftest()
        sys.exit(0)
    d = np.load(a.npz)
    cum, labels = d["cum_counts"], d["labels"]
    base, rows = sweep(cum, labels)
    T = cum.shape[1]
    with open(f"{a.prefix}_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(base, rows, T, f"{a.prefix}_pareto.png")
    print(f"baseline ({T} steps): {100*base:.2f}%   N={len(labels)}")
    print("Pareto-optimal operating points (t_min, margin, acc, avg steps, speedup):")
    for r in pareto(rows):
        print(f"  t_min={r['t_min']} M={r['margin']:2d}  acc={100*r['accuracy']:.2f}%  "
              f"steps={r['mean_steps']:.2f}  speedup={r['speedup']:.2f}x  drop={r['acc_drop_pts']:+.2f} pts")
    print(f"wrote {a.prefix}_sweep.csv and {a.prefix}_pareto.png")
