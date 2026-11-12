#!/usr/bin/env python3
"""
pynq_ee_demo.py - FPGA-in-the-loop Confidence-Gated Early Exit.

  ARM CPU : runs the spiking network layers (golden model)           -> output spikes of each timestep
  FPGA    : ee_axil block counts the spikes and DECIDES when to stop -> done / predicted class / steps

For every image the script cross-checks the FPGA's decision against the Python model. They must agree 100%.

Run on the PYNQ-Z2 (Jupyter terminal):
    sudo python3 pynq_ee_demo.py --root ~/snn/SNN_FPGA_V6_CIFAR10 --bit ~/pynq/overlays/snn/snn.bit \
         --cifar ~/snn/cifar-10-batches-py --n 20 --t_min 2 --margin 3
Test without hardware (software stand-in for the FPGA block):
    python3 pynq_ee_demo.py --root <...> --mock --n 3
"""
import argparse, csv, sys, time, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "golden"))
sys.path.insert(0, str(HERE))
import snn_golden as g
import early_exit as ee
import dashboard as dash

MAGIC = 0xEE01
REG_CTRL, REG_STATUS = 0x00, 0x04


class FpgaEE:
    """Thin driver for the ee_axil register block (see rtl/ee_axil.v for the register map)."""
    def __init__(self, mmio, t_min, margin, t_max=32):
        self.m = mmio
        self.cfg = (t_min << 12) | (margin << 18) | (t_max << 24)
        s = self.status()
        if (s >> 16) != MAGIC:
            raise RuntimeError(f"Wrong block / wrong address: STATUS read 0x{s:08x}, expected 0xEE01xxxx. "
                               "Check the base address in Vivado's Address Editor (use --base).")

    def status(self):
        return self.m.read(REG_STATUS)

    def start(self):
        self.m.write(REG_CTRL, self.cfg | (1 << 11))

    def step(self, spikes10):
        word = int(sum(int(b) << c for c, b in enumerate(spikes10)))
        self.m.write(REG_CTRL, self.cfg | (1 << 10) | word)
        s = self.status()
        return bool(s & 1), (s >> 1) & 0xF, (s >> 5) & 0x3F


class MockMMIO:
    """Software stand-in with the same register protocol (for testing this script without a board)."""
    def __init__(self):
        self.cnt = [0] * 10; self.t = 0; self.done = 0; self.pred = 0; self.steps = 0; self.cfg = 0

    def write(self, off, val):
        assert off == REG_CTRL
        self.cfg = val
        tmin, marg, tmax = (val >> 12) & 63, max((val >> 18) & 63, 1), (val >> 24) & 63
        if val & (1 << 11):
            self.cnt = [0] * 10; self.t = 0; self.done = 0; self.pred = 0; self.steps = 0
        elif (val & (1 << 10)) and not self.done:
            for c in range(10):
                self.cnt[c] += (val >> c) & 1
            self.t += 1
            top = max(self.cnt); arg = self.cnt.index(top)
            second = max(v for i, v in enumerate(self.cnt) if i != arg)
            if (self.t >= tmin and top - second >= marg) or self.t >= tmax:
                self.done, self.pred, self.steps = 1, arg, self.t

    def read(self, off):
        assert off == REG_STATUS
        return (MAGIC << 16) | (self.steps << 5) | (self.pred << 1) | self.done


def open_hw(a):
    if a.mock:
        return MockMMIO()
    from pynq import Overlay, MMIO
    ol = Overlay(a.bit)
    base = a.base
    if base is None:
        cands = [k for k, v in ol.ip_dict.items() if "ee_axil" in (k + str(v.get("type", ""))).lower()]
        if not cands:
            raise SystemExit("Could not find ee_axil in the overlay. Run: print(list(ol.ip_dict)) and pass --base 0x....")
        base = ol.ip_dict[cands[0]]["phys_addr"]
        print(f"found {cands[0]} at 0x{base:08x}")
    return MMIO(base, 0x1000)


def synthetic_images(n):
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n):
        yy, xx = np.mgrid[0:32, 0:32] / 31.0
        c = rng.random(3)
        out.append(np.clip(np.stack([xx * c[0], yy * c[1], (xx + yy) / 2 * c[2]]) + 0.1 * rng.random((3, 32, 32)), 0, 1).astype(np.float32))
    return np.stack(out), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bit"); ap.add_argument("--base", type=lambda s: int(s, 0))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--cifar"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", default="fixed"); ap.add_argument("--encoding", default="poisson"); ap.add_argument("--norm", default="unit")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--t_min", type=int, default=2); ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--stop_at_exit", action="store_true", help="really stop computing when the FPGA says done (real speed-up)")
    ap.add_argument("--power_w", type=float, help="board power you MEASURED with a USB power meter (optional)")
    ap.add_argument("--dash_index", type=int, default=0)
    ap.add_argument("--csv", default="board_results.csv"); ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    if not a.mock and not a.bit:
        raise SystemExit("give --bit <overlay.bit> or use --mock")

    scale = 1 / 64 if a.encoding == "direct" else 1.0
    net = g.Net(g.Params(a.root, scale), a.mode, scale)
    X, Y = g.load_cifar_test(a.cifar, a.n) if a.cifar else synthetic_images(a.n)
    fpga = FpgaEE(open_hw(a), a.t_min, a.margin, a.T)
    print(f"FPGA block found (magic OK). Running {a.n} images, t_min={a.t_min}, margin={a.margin}, T={a.T}\n")

    rows, keep = [], {}
    for idx in range(a.n):
        it = dash.stream(net, X[idx], a.T, a.encoding, a.norm, seed=idx)
        outs, rates, durs = [], [], []
        fpga.start(); done = False; pl = (0, 0)
        io_t = 0.0
        for t in range(1, a.T + 1):
            t0 = time.perf_counter(); o, r = next(it); durs.append(time.perf_counter() - t0)
            outs.append(o); rates.append([r[n] for n in g.LIF_NAMES])
            if not done:
                t1 = time.perf_counter(); done, p, s = fpga.step(o); io_t += time.perf_counter() - t1
                if done:
                    pl = (s, p)
            if done and a.stop_at_exit:
                break
        out = np.array(outs); cum = out.cumsum(0)
        acc_py, _, st_py, pr_py = ee.early_exit_eval(cum[None], np.array([-1]), a.t_min, a.margin)
        steps_py, pred_py = int(st_py[0]), int(pr_py[0])
        if not done:
            raise RuntimeError("FPGA never asserted done (it must at t_max).")
        match = (pl[0] == steps_py and pl[1] == pred_py)
        full = not a.stop_at_exit
        lab = int(Y[idx]) if Y is not None else -1
        rows.append(dict(idx=idx, label=lab, fpga_steps=pl[0], fpga_pred=pl[1], python_steps=steps_py, python_pred=pred_py,
                         match=int(match), full_pred=int(cum[-1].argmax()) if full else "",
                         t_ee_s=round(sum(durs[:pl[0]]), 4), t_full_s=round(sum(durs), 4) if full else "", fpga_io_s=round(io_t, 5)))
        print(f"img {idx:3d}  FPGA: exit@{pl[0]:2d} -> {dash.CLASSES[pl[1]]:<10s} | python: exit@{steps_py:2d} -> "
              f"{dash.CLASSES[pred_py]:<10s} | {'MATCH' if match else '*** MISMATCH ***'}"
              + (f" | true: {dash.CLASSES[lab]}" if lab >= 0 else ""))
        if idx == a.dash_index:
            full_out = np.zeros((a.T, 10), np.int32); full_out[:len(out)] = out
            rt = np.zeros((len(g.LIF_NAMES), a.T)); rt[:, :len(rates)] = np.array(rates).T
            keep = dict(img=(X[idx] * 255).astype(np.uint8), out=full_out, rate=rt, ex=pl[0] if pl[0] < a.T else None, lab=lab)

    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    n = len(rows); matches = sum(r["match"] for r in rows)
    avg_steps = np.mean([r["fpga_steps"] for r in rows])
    print("\n================ SUMMARY ================")
    print(f"FPGA decision == Python decision : {matches}/{n}  {'(PERFECT)' if matches == n else '(INVESTIGATE!)'}")
    print(f"average timesteps used           : {avg_steps:.2f} of {a.T}  ->  {a.T / avg_steps:.2f}x fewer")
    if Y is not None:
        print(f"accuracy with early exit         : {100 * np.mean([r['fpga_pred'] == r['label'] for r in rows]):.1f}%")
        if not a.stop_at_exit:
            print(f"accuracy, full {a.T} steps         : {100 * np.mean([r['full_pred'] == r['label'] for r in rows]):.1f}%")
    t_ee = np.mean([r["t_ee_s"] for r in rows])
    if not a.stop_at_exit:
        t_full = np.mean([r["t_full_s"] for r in rows])
        print(f"compute time/image: full {t_full:.2f}s  vs  early-exit {t_ee:.2f}s  ({t_full / t_ee:.2f}x faster)")
    print(f"FPGA register I/O per image      : {1000 * np.mean([r['fpga_io_s'] for r in rows]):.2f} ms")
    print(f"wrote {a.csv}")

    if keep:
        kpi = {"avg steps": f"{avg_steps:.1f}/{a.T}", "FPGA==Python": f"{matches}/{n}",
               "time/img": f"{t_ee:.2f} s"}
        if a.power_w:
            kpi["energy/img (measured P)"] = f"{a.power_w * t_ee:.2f} J"
        lab = f"board run, image #{a.dash_index}" + (f" (true: {dash.CLASSES[keep['lab']]})" if keep["lab"] >= 0 else " (synthetic)")
        out = pathlib.Path(a.outdir)
        dash.save_png(out / "board_dashboard.png", keep["img"], keep["out"], keep["rate"], a.T, keep["ex"], kpi, lab)
        dash.save_gif(out / "board_demo.gif", keep["img"], keep["out"], keep["rate"], a.T, keep["ex"], kpi, lab)
        print("wrote board_dashboard.png and board_demo.gif")


if __name__ == "__main__":
    main()
