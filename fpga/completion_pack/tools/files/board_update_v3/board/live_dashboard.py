#!/usr/bin/env python3
"""
live_dashboard.py - real-time "review day" dashboard for the SNN on the PYNQ-Z2.

   LEFT   : the CIFAR-10 test image + a picture of the board's LEDs (what the physical LEDs show)
   MIDDLE : spike raster, one block of rows per layer, over the timesteps so far
   RIGHT  : live class bars (share of output spikes) + lead-margin gauge toward the early-exit threshold

Every timestep the output spikes are sent to the FPGA (ee_axil); the FPGA decides when to stop.
With the LED bitstream (v3) the FPGA also measures the time of the first output spike in hardware and the
board's LEDs follow the run:  LD0..LD3 = leading class as a 4-bit number (LD0 = least significant bit),
RGB LD4 green = first output spike arrived, RGB LD5 green = early exit / red = ran the full 32 steps.

HONESTY NOTES (also printed on the dashboard):
  * "confidence" = share of output spikes so far. It is NOT a calibrated probability.
  * mode LIVE   : the ARM CPU computes every timestep while you watch (about 0.5 s per step on the board).
    mode REPLAY : spike trains were computed earlier and are replayed at a chosen speed. The FPGA decision is still
    live, but the spikes are not computed live. The dashboard says which mode you are in.
"""
import io, sys, time, pathlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "golden"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import snn_golden as g

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
BG, PANEL, FG, GRID = "#0b1020", "#131a30", "#e8ecf8", "#2a3358"
ACCENT, ACCENT2, WARN, BAD = "#35e0c1", "#7aa2ff", "#ffb454", "#ff5d73"
LAYER_LABELS = ["conv0", "res1.a", "res1.b", "conv1", "res2.a", "res2.b", "conv2", "res3.a", "res3.b", "output"]
LAYER_COLORS = ["#7aa2ff", "#6ec6ff", "#4fd8e6", "#35e0c1", "#7be495", "#b6e36b", "#ffd166", "#ffb454", "#ff8c6b", "#ff5d73"]
K_CHANNELS = 8                       # channels shown per layer (output layer shows all 10 classes)


# ----------------------------------------------------------------------------- data source (software)
def stream_full(net, img01, T=32, encoding="poisson", norm="unit", seed=0, k=K_CHANNELS):
    """Yield per timestep: (out_spikes (10,), {layer: mean firing rate}, {layer: fraction of each shown channel firing})."""
    x = img01[None].astype(np.float32)
    for n in g.LIF_NAMES:
        net.lif[n].mem = None
    picks = {}
    for frame in g.make_frames(x, T, encoding, norm, seed):
        if net.mode != "float" and encoding == "direct":
            frame = np.clip(np.round(frame / net.in_scale), -127, 127).astype(np.float32)
        rec = {}
        out = net.step(frame, lambda name, s: rec.__setitem__(name, s))
        rates, samp = {}, {}
        for n in g.LIF_NAMES:
            s = rec[n][0]                                   # (C,H,W) or (C,)
            C = s.shape[0]
            flat = s.reshape(C, -1)
            if n not in picks:
                picks[n] = np.arange(C) if n == "lif_out" else np.unique(np.linspace(0, C - 1, min(k, C)).astype(int))
            rates[n] = float(s.mean())
            samp[n] = flat[picks[n]].mean(1).astype(np.float32)      # fraction of the channel's neurons firing
        yield out[0].astype(np.int32), rates, samp


def make_cache(net, X, Y, idxs, T=32, encoding="poisson", norm="unit", path="demo_cache.npz", progress=print):
    """Pre-compute spike trains for chosen images (run once; then REPLAY is instant)."""
    outs, rts, ras, labs = [], [], [], []
    for n_i, idx in enumerate(idxs):
        o_l, r_l, s_l = [], [], []
        for o, r, s in stream_full(net, X[idx], T, encoding, norm, seed=int(idx)):
            o_l.append(o); r_l.append([r[n] for n in g.LIF_NAMES])
            row = np.zeros((len(g.LIF_NAMES), 10), np.float32)
            for li, n in enumerate(g.LIF_NAMES):
                row[li, :len(s[n])] = s[n]
            s_l.append(row)
        outs.append(np.array(o_l)); rts.append(np.array(r_l).T); ras.append(np.array(s_l).transpose(1, 2, 0))   # (L,10,T)
        labs.append(-1 if Y is None else int(Y[idx]))
        progress("cached image %d (%d/%d)" % (idx, n_i + 1, len(idxs)))
    ks = np.array([10 if n == "lif_out" else min(K_CHANNELS, {"lif0": 64, "lif1": 128, "lif2": 256}.get(n, 64 if "res1" in n else 128 if "res2" in n else 256)) for n in g.LIF_NAMES])
    np.savez(path, idxs=np.array(idxs), out=np.array(outs), rate=np.array(rts), ras=np.array(ras), labels=np.array(labs),
             imgs=(X[list(idxs)] * 255).astype(np.uint8), ks=ks, T=T)
    return path


def replay_iter(cache, i):
    """Same tuple format as stream_full, but from a cache file (REPLAY mode)."""
    out, rate, ras, ks = cache["out"][i], cache["rate"][i], cache["ras"][i], cache["ks"]
    for t in range(out.shape[0]):
        yield out[t], {n: float(rate[li, t]) for li, n in enumerate(g.LIF_NAMES)}, \
            {n: ras[li, :ks[li], t] for li, n in enumerate(g.LIF_NAMES)}


# ----------------------------------------------------------------------------- drawing
def _style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_title(title, color=FG, fontsize=10, loc="left", fontweight="bold", pad=6)


def render(st):
    """st: dict with img_u8, out (T,10), ras {layer:(K,T)}, rate (L,T), t, T, exit_step, margin_needed, first_step,
    first_src, led_code, rgb {rgb0_g,rgb1_g,rgb1_r}, mode, label, extra_kpi {name:value}. Returns PNG bytes."""
    T, t = st["T"], st["t"]
    fig = Figure(figsize=(16, 8.6), dpi=88)
    fig.patch.set_facecolor(BG)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 0.82, 0.2], width_ratios=[0.85, 1.75, 1.15],
                  hspace=0.42, wspace=0.26, left=0.04, right=0.985, top=0.855, bottom=0.04)
    fig.text(0.04, 0.945, "Spiking ResNet on Zynq-7020 (PYNQ-Z2)  |  live", color=FG, fontsize=18, fontweight="bold")
    fig.text(0.04, 0.905, "INT8 weights  |  Q16 integer LIF  |  FPGA-decided early exit  |  " + st.get("label", ""),
             color=ACCENT2, fontsize=10)
    mode = st.get("mode", "LIVE")
    fig.text(0.985, 0.945, ("\u25cf LIVE: ARM computes each timestep" if mode == "LIVE" else "\u25cf REPLAY: precomputed spikes, FPGA decision live"),
             color=ACCENT if mode == "LIVE" else WARN, fontsize=10.5, fontweight="bold", ha="right")

    out = st["out"][:t]
    cum = out.cumsum(0)
    exited = st["exit_step"] is not None
    snap = cum[(st["exit_step"] if exited else t) - 1] if t > 0 else np.zeros(10, int)
    total = int(snap.sum())
    lead = int(snap.argmax())
    srt = np.sort(snap)
    margin_now = int(srt[-1] - srt[-2])
    M = st["margin_needed"]

    # ---- left top: image
    ax = fig.add_subplot(gs[0, 0]); _style(ax, "Input image")
    ax.imshow(st["img_u8"].transpose(1, 2, 0), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(("PREDICTION: " + CLASSES[lead].upper()) if exited else (("leading: " + CLASSES[lead]) if total else "waiting for output spikes..."),
                  color=ACCENT if exited else WARN, fontsize=12, fontweight="bold")

    # ---- left bottom: virtual board LEDs
    ax = fig.add_subplot(gs[1, 0]); _style(ax, "Board LEDs (what you see on the PYNQ-Z2)")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
    code = st["led_code"]
    for b in range(4):
        on = (code >> b) & 1
        ax.add_patch(Circle((1.3 + b * 1.45, 8.3), 0.5, fc=ACCENT if on else "#1b2342", ec=GRID, lw=1.5))
        ax.text(1.3 + b * 1.45, 7.3, "LD%d" % b, color=FG, fontsize=7, ha="center")
    ax.text(0.3, 6.45, "class code (LD0 = lowest bit): %d = %s" % (code, CLASSES[code] if code < 10 else "-"), color=FG, fontsize=7.5)
    rgb = st["rgb"]
    ax.add_patch(Circle((7.6, 8.3), 0.5, fc="#35e07a" if rgb["rgb0_g"] else "#1b2342", ec=GRID, lw=1.5))
    ax.text(7.6, 7.3, "LD4 first\nspike", color=FG, fontsize=6.5, ha="center", va="top")
    fc5 = "#35e07a" if rgb["rgb1_g"] else (BAD if rgb["rgb1_r"] else "#1b2342")
    ax.add_patch(Circle((9.2, 8.3), 0.5, fc=fc5, ec=GRID, lw=1.5))
    ax.text(9.2, 7.3, "LD5 exit\ngrn=early\nred=full", color=FG, fontsize=6.5, ha="center", va="top")
    for i, c in enumerate(CLASSES):
        ax.text(0.3 + (i // 5) * 4.9, 5.2 - (i % 5) * 0.95, "%d %-10s %s" % (i, c, format(i, "04b")),
                color=ACCENT if (i == lead and total) else FG, fontsize=7.2, family="monospace")
    ax.text(0.3, 0.1, "binary shown LD3..LD0", color=ACCENT2, fontsize=6.5)

    # ---- middle: raster
    ax = fig.add_subplot(gs[0:2, 1]); _style(ax, "Spike raster per layer (rows = feature-map channels; dot size = fraction firing)")
    Ks = [st["ras"][n].shape[0] for n in g.LIF_NAMES]
    SP = [1.0] * (len(Ks) - 1) + [2.6]                 # output layer gets taller rows so the class names fit
    y0, yticks, ylabels = 0.0, [], []
    for li, n in enumerate(g.LIF_NAMES):
        K, sp = Ks[li], SP[li]
        ax.axhspan(y0 - 0.5 * sp, y0 + (K - 0.5) * sp, color="#182040" if li % 2 == 0 else PANEL, zorder=0)
        a = st["ras"][n][:, :t]
        yy, xx = np.nonzero(a > 0)
        if len(xx):
            ax.scatter(xx + 1, y0 + sp * yy, s=3 + 15 * np.sqrt(np.clip(a[yy, xx], 0, 1)) + (6 if li == len(Ks) - 1 else 0),
                       color=LAYER_COLORS[li], marker="o", linewidths=0, alpha=0.95, zorder=3)
        mr = 100 * float(st["rate"][li, :t].mean()) if t else 0.0
        yticks.append(y0 + sp * (K - 1) / 2.0); ylabels.append("%s  %.1f%%" % (LAYER_LABELS[li], mr))
        out_y0 = y0
        y0 += sp * K + 1
    ax.set_xlim(0.5, T + 3.2); ax.set_ylim(y0 - 1, -1.5)
    for ci, cn in enumerate(CLASSES):
        ax.text(T + 0.9, out_y0 + SP[-1] * ci, cn[:5], color=LAYER_COLORS[-1], fontsize=6.5, va="center")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("timestep  (labels: layer, mean firing rate so far)", color=FG, fontsize=8.5)
    ax.grid(axis="x", color=GRID, alpha=0.5)
    if exited:
        es = st["exit_step"]
        ax.axvline(es + 0.5, color=WARN, ls="--", lw=2)
        txt = ("EARLY EXIT @ t=%d" % es) if es < T else "no early exit: all %d steps" % T
        right = es > 0.6 * T
        ax.text(es - 0.3 if right else es + 1.0, -0.9, txt, color=WARN, fontsize=10, fontweight="bold", va="center",
                ha="right" if right else "left")
    if st.get("first_step"):
        ax.axvline(st["first_step"], color="#35e07a", ls=":", lw=1.6)
        ax.text(st["first_step"] + 0.3, y0 - 1.2, "first output spike", color="#35e07a", fontsize=8, va="bottom",
                bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.85, pad=1.2), zorder=6)

    # ---- right top: class bars (share of output spikes)
    ax = fig.add_subplot(gs[0, 2]); _style(ax, "Class confidence = share of output spikes")
    share = snap / total if total else np.zeros(10)
    colors = [ACCENT if (i == lead and total) else "#5a6fb0" for i in range(10)]
    ax.barh(np.arange(10), share, color=colors, height=0.72)
    ax.set_yticks(np.arange(10)); ax.set_yticklabels(CLASSES, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0%", "50%", "100%"])
    for i in range(10):
        if total:
            ax.text(share[i] + 0.02, i, "%d%%  (%d)" % (round(100 * share[i]), snap[i]), color=FG, fontsize=7.5, va="center")
    if not total:
        ax.text(0.5, 4.5, "no output spikes yet", color=WARN, fontsize=10, ha="center")
    ax.set_xlabel("share of output spikes so far (not a calibrated probability)", color=ACCENT2, fontsize=7.5)

    # ---- right bottom: margin gauge + first-spike info
    ax = fig.add_subplot(gs[1, 2]); _style(ax, "Early-exit rule: lead over runner-up")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    frac = min(1.0, margin_now / float(M)) if M else 0
    ax.barh([0.72], [1.0], height=0.16, color="#1b2342", edgecolor=GRID)
    ax.barh([0.72], [frac], height=0.16, color=WARN if (frac >= 1 or exited) else ACCENT)
    ax.text(0.5, 0.88, "lead %d / %d spikes needed" % (margin_now, M), color=FG, fontsize=11, ha="center", fontweight="bold")
    fs = st.get("first_step", 0)
    src = st.get("first_src", "")
    ax.text(0.04, 0.45, ("first output spike: step %d  (%s)" % (fs, src)) if fs else "first output spike: not yet",
            color="#35e07a" if fs else FG, fontsize=9.5)
    if exited:
        ax.text(0.04, 0.27, "FPGA decision: STOP at step %d of %d  (%.1fx fewer timesteps)" % (st["exit_step"], T, T / float(st["exit_step"])) if st["exit_step"] < T
                else "FPGA decision: no early exit - used all %d steps" % T, color=WARN, fontsize=9.5, fontweight="bold")
    else:
        ax.text(0.04, 0.27, "FPGA decision: still running ...", color=FG, fontsize=9.5)
    ax.text(0.04, 0.09, "rule: exit when lead >= %d spikes (and step >= %d)" % (M, st.get("t_min", 1)), color=ACCENT2, fontsize=8)

    # ---- KPI strip
    ax = fig.add_subplot(gs[2, :]); ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    cells = [("timestep", "%d / %d" % (st["exit_step"] if exited else t, T))]
    cells += [(k, str(v)) for k, v in st.get("extra_kpi", {}).items()]
    for i, (k, v) in enumerate(cells):
        x = (i + 0.5) / len(cells)
        ax.text(x, 0.64, v, ha="center", va="center", color=ACCENT, fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.2, k, ha="center", va="center", color=FG, fontsize=8.5, transform=ax.transAxes)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return buf.getvalue()


# ----------------------------------------------------------------------------- the demo loop
def jupyter_display(inplace=True):
    """Returns show(png). inplace=True updates ONE output cell without flicker (use for a plain cell);
    inplace=False clears and redraws (use inside an ipywidgets Output widget). Backend independent."""
    from IPython.display import display, Image, clear_output
    h = {"handle": None}

    def show(png):
        im = Image(data=png)
        if not inplace:
            clear_output(wait=True)
            display(im)
        elif h["handle"] is None:
            h["handle"] = display(im, display_id=True)
        else:
            h["handle"].update(im)
    return show


def run_image(img_u8, steps_iter, fpga, T=32, show=None, mode="LIVE", step_delay=0.0, use_leds=True, label="", keep_frames=None):
    """Drive one image through the FPGA-in-the-loop demo. `steps_iter` yields (out_spikes, rates, samples) per timestep."""
    L = len(g.LIF_NAMES)
    out = np.zeros((T, 10), np.int32)
    rate = np.zeros((L, T))
    ras = {n: None for n in g.LIF_NAMES}
    fpga.start()
    if use_leds:
        fpga.set_leds(0)
    exit_step, pred, first_sw = None, None, 0
    t0 = time.perf_counter()
    t_first_wall = None
    for t, (o, r, s) in enumerate(steps_iter, start=1):
        t_step = time.perf_counter()
        out[t - 1] = o
        rate[:, t - 1] = [r[n] for n in g.LIF_NAMES]
        for n in g.LIF_NAMES:
            if ras[n] is None:
                ras[n] = np.zeros((len(s[n]), T), np.float32)
            ras[n][:, t - 1] = s[n]
        if not first_sw and o.any():
            first_sw = t
            t_first_wall = time.perf_counter() - t0
        done, p, steps = fpga.step(o)
        cum = out[:t].cumsum(0)[-1]
        if done and exit_step is None:
            exit_step, pred = steps, p
        led = pred if done else int(cum.argmax())
        if use_leds:
            fpga.set_leds(led if cum.sum() or done else 0)
        seen, fstep = fpga.first_spike()
        first_step, src = (fstep, "measured by FPGA") if (seen and fpga.has_extras) else (first_sw, "software")
        st = dict(img_u8=img_u8, out=out, ras={n: (ras[n][:, :] if ras[n] is not None else np.zeros((1, T))) for n in g.LIF_NAMES},
                  rate=rate, t=t, T=T, exit_step=exit_step, margin_needed=fpga.margin, t_min=fpga.t_min,
                  first_step=first_step, first_src=src, led_code=led if (cum.sum() or done) else 0,
                  rgb=fpga.rgb_state(done, steps), mode=mode, label=label,
                  extra_kpi={"first spike": ("t=%d" % first_step) if first_step else "-",
                             "elapsed": "%.1f s" % (time.perf_counter() - t0),
                             "FPGA decision": ("no early exit" if exit_step >= T else "STOP @%d" % exit_step) if exit_step else "running",
                             "LEDs": "on board" if (use_leds and fpga.has_extras) else "virtual only"})
        png = render(st)
        if keep_frames is not None:
            keep_frames.append(png)
        if show:
            show(png)
        if step_delay:
            time.sleep(max(0.0, step_delay - (time.perf_counter() - t_step)))
        if done:
            break
    return dict(exit_step=exit_step, pred=pred, first_step=first_step, elapsed=time.perf_counter() - t0, steps_run=t,
                first_wall=t_first_wall, out=out[:t])


def save_gif(pngs, path, fps=5, hold=6):
    from PIL import Image
    fr = [Image.open(io.BytesIO(p)).convert("P", palette=Image.ADAPTIVE) for p in pngs]
    fr[0].save(str(path), save_all=True, append_images=fr[1:] + [fr[-1]] * hold, duration=int(1000 / fps), loop=0)
