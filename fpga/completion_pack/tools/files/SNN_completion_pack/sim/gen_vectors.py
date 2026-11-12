#!/usr/bin/env python3
"""Generate RTL test vectors whose expected values come from the golden model (snn_golden.py / early_exit.py)."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "golden"))
import snn_golden as g
import early_exit as ee

HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(2026)


def hx(v, bits):
    return f"{int(v) & ((1 << bits) - 1):0{(bits + 3) // 4}x}"


def gen_lif(n=20000):
    # realistic range + edge cases (threshold boundary, saturation, rounding ties)
    acc = rng.integers(-(1 << 18), 1 << 18, n)
    m = rng.integers(-(1 << 19), 1 << 19, n)
    b = rng.integers(-(1 << 19), 1 << 19, n)
    mem = rng.integers(-(1 << 21), 1 << 21, n)
    k = 0
    for mv in (g.THR_Q - 1, g.THR_Q, g.THR_Q + 1, 0, -g.THR_Q):                      # threshold boundary
        mem[k] = mv; acc[k] = 0; b[k] = 0; k += 1
    mem[k], acc[k], m[k], b[k] = (1 << 31) - 5, (1 << 19) - 1, (1 << 20) - 1, (1 << 30); k += 1   # + saturation
    mem[k], acc[k], m[k], b[k] = -(1 << 31) + 5, -(1 << 19), (1 << 20) - 1, -(1 << 30); k += 1     # - saturation
    mem[k], acc[k], m[k], b[k] = 7, 3, 128, 0; k += 1                                 # exact rounding tie region
    # 400 vectors whose NEW membrane lands within +-2 LSB of the threshold (tests the strict '>' spike compare)
    for j in range(400):
        idx = k + j
        acc[idx], m[idx] = 0, 0
        mem[idx] = rng.integers(-(1 << 17), 1 << 17)
        decay = (int(mem[idx]) * g.BETA_Q + (1 << 15)) >> 16
        reset = int(mem[idx]) > g.THR_Q
        b[idx] = g.THR_Q - decay + (g.THR_Q if reset else 0) + rng.integers(-2, 3)
    y = ((acc.astype(np.int64) * m + 128) >> 8) + b
    lif = g.LIF(True)
    lif.mem = mem.astype(np.int64).copy()
    spk = lif.step(y)
    with open(HERE / "lif_vectors.txt", "w") as f:
        for i in range(n):
            f.write(f"{hx(acc[i],20)} {hx(m[i],21)} {hx(b[i],32)} {hx(mem[i],32)} {hx(lif.mem[i],32)} {int(spk[i])}\n")
    print(f"lif_vectors.txt: {n} vectors, spikes={int(spk.sum())}, saturated={(np.abs(lif.mem) >= (1<<31)-1).sum()}")


def gen_ee(n=3000, T=32, C=10):
    ff = open(HERE / "ee_first.txt", "w")
    with open(HERE / "ee_vectors.txt", "w") as f:
        exits = []
        for _ in range(n):
            true = rng.integers(0, C)
            p = np.full(C, rng.uniform(0.02, 0.2)); p[true] = rng.uniform(0.15, 0.7)
            if rng.random() < 0.3:                                      # ambiguous image: a rival class
                p[(true + 1 + rng.integers(0, C - 1)) % C] = p[true] * rng.uniform(0.7, 1.0)
            if rng.random() < 0.05:
                p[:] = 0.0                                              # silent output layer
            spikes = (rng.random((T, C)) < p).astype(np.int32)
            cum = spikes.cumsum(0)[None]
            tmin = int(rng.choice([1, 2, 4, 8])); marg = int(rng.integers(0, 9))
            _, _, steps, pred = ee.early_exit_eval(cum, np.array([true]), tmin, max(marg, 1))
            words = [int((s * (1 << np.arange(C))).sum()) for s in spikes]
            f.write(f"{tmin:x} {marg:x} " + " ".join(f"{w:03x}" for w in words) + f" {int(steps[0]):x} {int(pred[0]):x}\n")
            exits.append(int(steps[0]))
            first = next((t + 1 for t in range(int(steps[0])) if spikes[t].any()), 0)     # 1-based, 0 = none
            ff.write(f"{first:x}\n")
    ff.close()
    print(f"ee_vectors.txt: {n} images, exit-step histogram: 32-step runs={exits.count(32)}, mean={np.mean(exits):.1f}")


if __name__ == "__main__":
    gen_lif(); gen_ee()
