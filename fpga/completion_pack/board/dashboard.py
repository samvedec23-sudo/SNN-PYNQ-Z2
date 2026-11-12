#!/usr/bin/env python3
"""
dashboard.py - presentation dashboard for the SNN demo (static PNG, animated GIF, or live in Jupyter).

Data source contract (software today, FPGA later - same format):
    stream(...) yields, once per timestep t = 1..T:
        out_spikes : (10,)  0/1 spikes of the output layer
        layer_rate : {layer_name: spikes_this_step / neurons}        (for the activity heat-map)
KPIs shown are only what you pass in `kpi` - nothing is invented. Measure them, then fill them in.

CLI:  python dashboard.py --root <SNN_FPGA_V6_CIFAR10> --cifar <cifar-10-batches-py> --index 0 --gif demo.gif
"""
import argparse, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "golden"))
import snn_golden as g
import early_exit as ee

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BG, PANEL, FG, GRID = "#0b1020", "#131a30", "#e8ecf8", "#2a3358"
ACCENT, ACCENT2, WARN = "#35e0c1", "#7aa2ff", "#ffb454"
LAYER_LABELS = ["conv0", "res1.a", "res1.b", "conv1", "res2.a", "res2.b", "conv2", "res3.a", "res3.b", "out"]


# ----------------------------------------------------------------------------- software backend
def stream(net, img01, T=32, encoding="poisson", norm="unit", seed=0):
    """Software backend: runs the golden model one timestep at a time (img01: (3,32,32) in [0,1])."""
    x = img01[None].astype(np.float32)
    for n in g.LIF_NAMES:
        net.lif[n].mem = None
    for frame in g.make_frames(x, T, encoding, norm, seed):
        if net.mode != "float" and encoding == "direct":
            frame = np.clip(np.round(frame / net.in_scale), -127, 127).astype(np.float32)
        rec = {}
        out = net.step(frame, lambda name, s: rec.__setitem__(name, float(s.mean())))
        yield out[0].astype(np.int32), rec


def collect(gen, T):
    out = np.zeros((T, 10), np.int32)
    rate = np.zeros((len(g.LIF_NAMES), T))
    for t, (o, r) in enumerate(gen):
        out[t] = o
        rate[:, t] = [r[n] for n in g.LIF_NAMES]
    return out, rate


# ----------------------------------------------------------------------------- drawing
def _style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_title(title, color=FG, fontsize=10, loc="left", fontweight="bold", pad=6)


def draw_frame(fig, img_u8, out_spk, rate, t, T=32, exit_step=None, kpi=None, label=None, title="Spiking ResNet on Zynq-7020 (PYNQ-Z2)"):
    """Draw the dashboard state after timestep t (1..T). Whole history is passed in; only [:t] is shown."""
    fig.clf()
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.28], width_ratios=[0.8, 1.5, 1.3],
                  hspace=0.45, wspace=0.28, left=0.04, right=0.98, top=0.84, bottom=0.05)
    fig.text(0.04, 0.945, title, color=FG, fontsize=17, fontweight="bold")
    fig.text(0.04, 0.905, "INT8 weights  |  Q16 integer LIF  |  Confidence-Gated Early Exit" +
             (f"  |  {label}" if label else ""), color=ACCENT2, fontsize=9.5)

    cum = out_spk[:t].cumsum(0)
    exited = exit_step is not None and t >= exit_step
    cur = cum[(exit_step if exited else t) - 1]          # state the controller acted on
    lead = int(cur.argmax())
    srt = np.sort(cur)
    margin = int(srt[-1] - srt[-2])

    # input image + prediction
    ax = fig.add_subplot(gs[0:2, 0])
    _style(ax, "Input")
    ax.imshow(img_u8.transpose(1, 2, 0), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(("PREDICTION: " if exited else "leading: ") + CLASSES[lead].upper() if cur.sum() else "waiting for spikes...",
                  color=ACCENT if exited else WARN, fontsize=12, fontweight="bold")

    # class race
    ax = fig.add_subplot(gs[0:2, 1])
    _style(ax, "Output spike count per class (the 'race')")
    cols = plt.cm.tab10(np.arange(10))
    order = np.argsort(-cur)
    for c in range(10):
        hi = c == lead
        ax.plot(np.arange(1, t + 1), cum[:, c], color=ACCENT if hi else cols[c], lw=3 if hi else 1.1,
                alpha=1 if hi else 0.55, zorder=3 if hi else 2)
    for rank, c in enumerate(order[:3]):
        ax.text(0.02, 0.95 - rank * 0.07, f"{CLASSES[c]:<10s}{cur[c]:>3d}", transform=ax.transAxes, color=ACCENT if rank == 0 else FG,
                fontsize=9, family="monospace", va="top", zorder=5,
                bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_xlim(1, T); ax.set_ylim(0, max(4, int(out_spk.cumsum(0).max()) + 1))
    ax.set_xlabel("timestep", color=FG, fontsize=9); ax.grid(color=GRID, alpha=.5)
    if exit_step is not None and t >= exit_step:
        ax.axvline(exit_step, color=WARN, ls="--", lw=2)
        ax.text(exit_step + 0.4, ax.get_ylim()[1] * 0.06, f"EARLY EXIT @ t={exit_step}\n{T / exit_step:.1f}x fewer timesteps",
                color=WARN, fontsize=10, fontweight="bold")

    # output raster
    ax = fig.add_subplot(gs[0, 2])
    _style(ax, "Output-layer spike raster")
    from matplotlib.colors import ListedColormap
    ax.imshow(out_spk[:t].T, cmap=ListedColormap([PANEL, ACCENT]), aspect="auto", extent=[0.5, t + .5, 9.5, -.5], vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xlim(0.5, T + .5); ax.set_yticks(range(10)); ax.set_yticklabels([c[:5] for c in CLASSES], fontsize=7)

    # layer activity
    ax = fig.add_subplot(gs[1, 2])
    _style(ax, "Spike activity per layer (spikes / neuron)")
    hm = np.full(rate.shape, np.nan)
    hm[:, :t] = rate[:, :t]
    im = ax.imshow(hm, aspect="auto", cmap="magma", vmin=0, vmax=max(0.3, float(np.nanmax(rate))), interpolation="nearest",
                   extent=[0.5, T + .5, len(LAYER_LABELS) - .5, -.5])
    ax.set_yticks(range(len(LAYER_LABELS))); ax.set_yticklabels(LAYER_LABELS, fontsize=7)
    ax.set_xlabel("timestep", color=FG, fontsize=8)

    # KPI strip
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    kpi = dict(kpi or {})
    cells = [("timesteps used", f"{exit_step if exited else t} / {T}"),
             ("confidence margin", f"{margin} spikes")] + [(k, str(v)) for k, v in kpi.items()]
    for i, (k, v) in enumerate(cells):
        x = (i + 0.5) / len(cells)
        ax.text(x, 0.64, v, ha="center", va="center", color=ACCENT, fontsize=15, fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.2, k, ha="center", va="center", color=FG, fontsize=8.5, transform=ax.transAxes)
    return fig


def save_png(path, img_u8, out_spk, rate, T=32, exit_step=None, kpi=None, label=None):
    fig = plt.figure(figsize=(14, 7.2), dpi=110)
    draw_frame(fig, img_u8, out_spk, rate, T, T, exit_step, kpi, label)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def save_gif(path, img_u8, out_spk, rate, T=32, exit_step=None, kpi=None, label=None, fps=6, dpi=70):
    from PIL import Image
    import io
    frames = []
    last = (exit_step + 4) if exit_step else T            # hold a few frames after the exit so it is readable
    fig = plt.figure(figsize=(14, 7.2), dpi=dpi)
    for t in range(1, min(T, last) + 1):
        draw_frame(fig, img_u8, out_spk, rate, t, T, exit_step, kpi, label)
        buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=int(1000 / fps), loop=0)


def live_jupyter(gen, img_u8, T=32, t_min=2, margin=2, kpi=None, label=None):
    """In a PYNQ/Jupyter notebook: redraws after every timestep as results arrive; stops at the early exit."""
    from IPython.display import display, clear_output
    out = np.zeros((T, 10), np.int32)
    rate = np.zeros((len(LAYER_LABELS), T))
    fig = plt.figure(figsize=(14, 7.2), dpi=90)
    for t, (o, r) in enumerate(gen, start=1):
        out[t - 1] = o
        rate[:, t - 1] = [r[n] for n in g.LIF_NAMES]
        c = out[:t].cumsum(0)[None]
        _, _, st, _ = ee.early_exit_eval(c, np.array([0]), t_min, margin)
        done = (c[0, -1].max() - np.sort(c[0, -1])[-2] >= margin and t >= t_min) or t == T
        draw_frame(fig, img_u8, out, rate, t, T, t if done else None, kpi, label)
        clear_output(wait=True); display(fig)
        if done:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cifar", help="folder with test_batch; omit to use a synthetic image")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mode", default="fixed")
    ap.add_argument("--encoding", default="poisson")
    ap.add_argument("--norm", default="unit")
    ap.add_argument("--t_min", type=int, default=2)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--png", default="dashboard.png")
    ap.add_argument("--gif", default=None)
    a = ap.parse_args()
    scale = 1 / 64 if a.encoding == "direct" else 1.0
    net = g.Net(g.Params(a.root, scale), a.mode, scale)
    if a.cifar:
        X, Y = g.load_cifar_test(a.cifar, a.index + 1)
        img, lab = X[a.index], f"CIFAR-10 test #{a.index} (true: {CLASSES[Y[a.index]]})"
    else:
        yy, xx = np.mgrid[0:32, 0:32] / 31.0
        img = np.stack([xx, yy, 0.5 * np.ones_like(xx)]).astype(np.float32)
        lab = "SYNTHETIC test image (layout demo)"
    out, rate = collect(stream(net, img, 32, a.encoding, a.norm), 32)
    _, _, st, _ = ee.early_exit_eval(out.cumsum(0)[None], np.array([0]), a.t_min, a.margin)
    ex = int(st[0]) if st[0] < 32 else None
    u8 = (img * 255).astype(np.uint8)
    save_png(a.png, u8, out, rate, 32, ex, {"latency": "measure me", "power": "measure me"}, lab)
    print("wrote", a.png, "| early exit at step:", ex)
    if a.gif:
        save_gif(a.gif, u8, out, rate, 32, ex, {"latency": "measure me", "power": "measure me"}, lab)
        print("wrote", a.gif)
