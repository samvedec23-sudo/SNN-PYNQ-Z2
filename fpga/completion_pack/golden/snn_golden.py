#!/usr/bin/env python3
"""
snn_golden.py  -  Reference ("golden") model of CIFAR10SNN_V4 for FPGA verification.

Three numeric modes, each one step closer to the hardware:
  float : float32 weights, float BN, float LIF            (should reproduce ~77% from training)
  int8  : INT8 weights (dequantised), float BN, float LIF (isolates weight-quantisation error)
  fixed : integer-only pipeline == what the RTL must compute, bit for bit
          acc(int32) -> y_q16 = ((acc*Mq + 128) >> 8) + Bq -> LIF in Q16

Network (from your screenshots / layer_config.json):
  conv0 -> BN -> LIF -> MaxPool2 -> res1 -> conv1 -> BN -> LIF -> MaxPool2 -> res2
        -> conv2 -> BN -> LIF -> MaxPool2 -> res3 -> flatten(4096) -> FC(10) -> LIF_out
  res block: x -> conv1 -> BN1 -> LIF1 -> conv2 -> BN2 -> LIF2 -> (+ x)      (add AFTER LIF2)
LIF (snnTorch Leaky, subtract reset, reset_delay=True, beta=0.85, thr=1.0):
  R[t-1] = (U[t-1] > thr);  U[t] = beta*U[t-1] + I[t] - R[t-1]*thr;  S[t] = (U[t] > thr)
"""
import argparse, json, pickle, time
from pathlib import Path
import numpy as np

T_DEFAULT = 32
BETA, THR, BN_EPS = 0.85, 1.0, 1e-5
QF = 16                       # fractional bits of membrane / BN offset / threshold
MF = 24                       # fractional bits of the per-channel multiplier Mq
BETA_Q = int(round(BETA * (1 << QF)))      # 55706
THR_Q = 1 << QF                            # 65536
MEM_MIN, MEM_MAX = -(2**31), 2**31 - 1     # membrane register is 32-bit signed, saturating

CONV_LAYERS = ["conv0", "res1__conv1", "res1__conv2", "conv1", "res2__conv1", "res2__conv2",
               "conv2", "res3__conv1", "res3__conv2"]
BN_OF = {"conv0": "bn0", "conv1": "bn1", "conv2": "bn2"}
for _r in (1, 2, 3):
    BN_OF[f"res{_r}__conv1"] = f"res{_r}__bn1"
    BN_OF[f"res{_r}__conv2"] = f"res{_r}__bn2"
LIF_NAMES = ["lif0", "res1__lif1", "res1__lif2", "lif1", "res2__lif1", "res2__lif2",
             "lif2", "res3__lif1", "res3__lif2", "lif_out"]


# ----------------------------------------------------------------------------- parameters
class Params:
    def __init__(self, root, input_scale=1.0):
        root = Path(root)
        f32, i8 = root / "weights_float32", root / "weights_int8_experiment"
        man = {m["file"]: m["scale"] for m in json.loads((i8 / "manifest.json").read_text())}
        self.layers = {}
        for name in CONV_LAYERS + ["fc"]:
            wf = np.load(f32 / f"{name}__weight.npy").astype(np.float32)
            wq = np.load(i8 / f"{name}__weight.npy")
            assert wq.dtype == np.int8, name
            s = man[f"{name}__weight.npy"]
            if name == "fc":
                A = np.ones(10, np.float64)
                B = np.load(f32 / "fc__bias.npy").astype(np.float64)
            else:
                bn = BN_OF[name]
                g, b, m, v = (np.load(f32 / f"{bn}__{k}.npy").astype(np.float64)
                              for k in ("weight", "bias", "running_mean", "running_var"))
                A = g / np.sqrt(v + BN_EPS)
                B = b - m * A
            in_s = input_scale if name == "conv0" else 1.0
            self.layers[name] = dict(
                shape=wf.shape, wf=wf.reshape(wf.shape[0], -1),
                wq=wq.reshape(wq.shape[0], -1).astype(np.float32), s=s, A=A, B=B,
                Mq=np.round(A * s * in_s * (1 << MF)).astype(np.int64),
                Bq=np.round(B * (1 << QF)).astype(np.int64))


def check_bn_against_export(root):
    """Cross-check our BN->Q16 numbers against your exported bn_fixed/*.npy (verifies EPS=1e-5)."""
    root = Path(root)
    P = Params(root)
    d = root / "hardware_params" / "bn_fixed"
    rep = {}
    for name, bn in BN_OF.items():
        fa, fb = d / f"{bn}_a_fixed.npy", d / f"{bn}_b_fixed.npy"
        if not fa.exists():
            continue
        L = P.layers[name]
        ea = np.abs(np.load(fa).astype(np.int64) - np.round(L["A"] * 65536).astype(np.int64)).max()
        eb = np.abs(np.load(fb).astype(np.int64) - np.round(L["B"] * 65536).astype(np.int64)).max()
        rep[bn] = (int(ea), int(eb))
    return rep


# ----------------------------------------------------------------------------- building blocks
def conv3x3(x, Wmat):
    """x:(B,C,H,W) float32, Wmat:(Cout, C*9) -> (B,Cout,H,W).  zero padding 1."""
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, :, i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=2)   # (B,C,9,H,W)
    cols = cols.transpose(0, 3, 4, 1, 2).reshape(B * H * W, C * 9)
    return (cols @ Wmat.T).reshape(B, H, W, -1).transpose(0, 3, 1, 2)


def maxpool2(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


class LIF:
    def __init__(self, fixed):
        self.fixed, self.mem = fixed, None

    def step(self, cur):
        if self.mem is None:
            self.mem = np.zeros(cur.shape, np.int64 if self.fixed else np.float32)
        if self.fixed:
            reset = self.mem > THR_Q
            self.mem = np.clip(((self.mem * BETA_Q + (1 << (QF - 1))) >> QF) + cur - reset * THR_Q, MEM_MIN, MEM_MAX)
            return (self.mem > THR_Q).astype(np.float32)
        reset = self.mem > THR
        self.mem = BETA * self.mem + cur - reset * THR
        return (self.mem > THR).astype(np.float32)


class Net:
    """Time-major execution (one timestep through all layers) so early exit is exact."""
    def __init__(self, params, mode="fixed", input_scale=1.0):
        assert mode in ("float", "int8", "fixed")
        self.P, self.mode, self.in_scale = params, mode, input_scale
        self.fixed = mode == "fixed"
        self.lif = {n: LIF(self.fixed) for n in LIF_NAMES}

    def bn_conv(self, name, x):
        L = self.P.layers[name]
        is_fc = name == "fc"
        if self.mode == "float":
            y = x @ L["wf"].T if is_fc else conv3x3(x, L["wf"])
            sh = (1, -1) + (1,) * (y.ndim - 2)
            return y * L["A"].reshape(sh).astype(np.float32) + L["B"].reshape(sh).astype(np.float32)
        acc = x @ L["wq"].T if is_fc else conv3x3(x, L["wq"])
        sh = (1, -1) + (1,) * (acc.ndim - 2)
        if self.mode == "int8":
            s = L["s"] * (self.in_scale if name == "conv0" else 1.0)
            return acc * s * L["A"].reshape(sh).astype(np.float32) + L["B"].reshape(sh).astype(np.float32)
        assert np.abs(acc).max() < 2**24, "float32 accumulator no longer exact"
        a = acc.astype(np.int64)
        return ((a * L["Mq"].reshape(sh) + 128) >> 8) + L["Bq"].reshape(sh)

    def res_block(self, r, x, rec):
        s1 = self.lif[f"{r}__lif1"].step(self.bn_conv(f"{r}__conv1", x)); rec(f"{r}__lif1", s1)
        s2 = self.lif[f"{r}__lif2"].step(self.bn_conv(f"{r}__conv2", s1)); rec(f"{r}__lif2", s2)
        return s2 + x

    def step(self, frame, rec):
        s = self.lif["lif0"].step(self.bn_conv("conv0", frame)); rec("lif0", s)
        x = maxpool2(s)
        x = self.res_block("res1", x, rec)
        s = self.lif["lif1"].step(self.bn_conv("conv1", x)); rec("lif1", s)
        x = maxpool2(s)
        x = self.res_block("res2", x, rec)
        s = self.lif["lif2"].step(self.bn_conv("conv2", x)); rec("lif2", s)
        x = maxpool2(s)
        x = self.res_block("res3", x, rec)
        out = self.lif["lif_out"].step(self.bn_conv("fc", x.reshape(x.shape[0], -1)))
        rec("lif_out", out)
        return out


# ----------------------------------------------------------------------------- data + encoding
MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)


def load_cifar_test(cifar_dir, n):
    d = pickle.load(open(Path(cifar_dir) / "test_batch", "rb"), encoding="bytes")
    raw = d[b"data"]                      # original file: numpy array; compact board file: raw bytes
    arr = np.frombuffer(raw, np.uint8) if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, np.uint8)
    x = arr.reshape(-1, 3072)[:n].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return x, np.array(d[b"labels"][:n])


def make_frames(x01, T, encoding, norm, seed):
    """Yield one input frame per timestep. x01 in [0,1], shape (B,3,32,32)."""
    rng = np.random.default_rng(seed)
    if encoding == "poisson":                       # rate coding, binary spikes (like snntorch spikegen.rate)
        for _ in range(T):
            yield (rng.random(x01.shape, dtype=np.float32) < x01).astype(np.float32)
    else:                                           # direct: same analog image every step
        x = (x01 - MEAN) / STD if norm == "cifar" else x01
        for _ in range(T):
            yield x


def run_batch(net, x01, T=T_DEFAULT, encoding="poisson", norm="unit", seed=0):
    """Returns cum_counts (B,T,10), final_mem (B,10), spikes_per_step {layer:(B,T)}, neurons {layer:n}."""
    B = x01.shape[0]
    cum = np.zeros((B, T, 10), np.int32)
    run = np.zeros((B, 10), np.int32)
    spk = {n: np.zeros((B, T), np.int64) for n in LIF_NAMES}
    neurons = {}
    for n in LIF_NAMES:
        net.lif[n].mem = None
    for t, frame in enumerate(make_frames(x01, T, encoding, norm, seed)):
        if net.mode != "float" and encoding == "direct":     # quantise analog input to int8
            frame = np.clip(np.round(frame / net.in_scale), -127, 127).astype(np.float32)

        def rec(name, s, t=t):
            spk[name][:, t] = s.reshape(B, -1).sum(1)
            neurons[name] = int(np.prod(s.shape[1:]))
        out = net.step(frame, rec)
        run += out.astype(np.int32)
        cum[:, t] = run
    return cum, net.lif["lif_out"].mem.copy(), spk, neurons


def predict(cum_counts, final_t=None):
    c = cum_counts[:, -1 if final_t is None else final_t - 1]
    return c.argmax(1)                  # ties -> lowest class index (hardware must do the same)


# ----------------------------------------------------------------------------- CLI
def selftest(root):
    print("== 1. BN fixed-point cross-check vs your exported bn_fixed/*.npy (max |diff| in LSB: A, B)")
    rep = check_bn_against_export(root)
    for k, v in rep.items():
        print(f"   {k:10s} A:{v[0]}  B:{v[1]}")
    print("   -> small values (0-1 LSB) mean EPS=1e-5 is right." if rep else "   (bn_fixed .npy not found - skipped)")
    print("== 2. float-INT8 vs fixed-point equivalence on synthetic images (T=8)")
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0.5, 0.25, (4, 3, 32, 32)).astype(np.float32), 0, 1)
    P = Params(root)
    t0 = time.time()
    ci, _, si, nn = run_batch(Net(P, "int8"), x, T=8, encoding="poisson", seed=5)
    cf, _, sf, _ = run_batch(Net(P, "fixed"), x, T=8, encoding="poisson", seed=5)
    print(f"   ran 2 modes x 4 images x 8 steps in {time.time()-t0:.1f}s")
    print("   layer         neurons  spikes(int8)  spikes(fixed)  rel.diff")
    for n in LIF_NAMES:
        a, b = si[n].sum(), sf[n].sum()
        print(f"   {n:12s} {nn[n]:8d} {a:12d} {b:13d}  {abs(a-b)/max(a,1):.4f}")
    print(f"   output counts identical: {(ci == cf).all()}   mean |diff|: {np.abs(ci-cf).mean():.3f}")
    print("== 3. mean spike rate per neuron per step (sparsity -> zero-skip potential)")
    for n in LIF_NAMES:
        print(f"   {n:12s} {sf[n].sum()/(4*8*nn[n]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to SNN_FPGA_V6_CIFAR10 folder")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cifar", help="folder containing 'test_batch' (cifar-10-batches-py)")
    ap.add_argument("--mode", default="fixed", choices=["float", "int8", "fixed"])
    ap.add_argument("--encoding", default="poisson", choices=["poisson", "direct"])
    ap.add_argument("--norm", default="unit", choices=["unit", "cifar"], help="direct encoding only")
    ap.add_argument("--input_scale", type=float, default=1 / 64, help="int8 step of analog input (direct only)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.root)
    scale = a.input_scale if a.encoding == "direct" else 1.0
    net = Net(Params(a.root, scale), a.mode, scale)
    X, Y = load_cifar_test(a.cifar, a.n)
    C, S = [], {n: [] for n in LIF_NAMES}
    t0 = time.time()
    for i in range(0, a.n, a.batch):
        c, _, s, _ = run_batch(net, X[i:i + a.batch], a.T, a.encoding, a.norm, a.seed + i)
        C.append(c)
        for k in S:
            S[k].append(s[k])
        acc = (predict(np.concatenate(C)) == Y[:i + len(c)]).mean() * 100
        print(f"\r{i + len(c)}/{a.n}  running acc {acc:.2f}%", end="")
    C = np.concatenate(C)
    print(f"\nmode={a.mode} enc={a.encoding}: accuracy {100 * (predict(C) == Y).mean():.2f}% "
          f"on {a.n} images, {time.time() - t0:.0f}s")
    out = a.out or f"results_{a.mode}_{a.encoding}.npz"
    np.savez(out, cum_counts=C, labels=Y, **{f"spk_{k}": np.concatenate(v) for k, v in S.items()})
    print("saved", out, "-> now run early_exit.py on it")


if __name__ == "__main__":
    main()
