# Spiking ResNet on PYNQ-Z2 — completion pack

Contents
```
golden/snn_golden.py      bit-accurate reference model (float / int8 / fixed) + CIFAR-10 evaluation
golden/early_exit.py      Confidence-Gated Early Exit (CGEE) sweep -> accuracy-vs-timesteps curve
rtl/lif_bn_unit.v         BN requantise + LIF neuron datapath      (simulated, bit-exact vs golden)
rtl/early_exit_ctrl.v     early-exit controller for the output layer, v2 timing-optimised (simulated, bit-exact vs golden)
rtl/ee_axil.v             AXI4-Lite register block so the ARM CPU can talk to early_exit_ctrl (simulated)
sim/                      testbenches + vector generator + run_sim.sh (Icarus Verilog or Vivado xsim)
board/dashboard.py        demo dashboard: PNG / GIF / live Jupyter on the PYNQ board
tools/make_board_zip.py   builds ONE zip with everything the board needs (code + weights + small CIFAR file)
board/pynq_ee_demo.py     runs on the board: ARM computes layers, FPGA decides the early exit (has --mock mode)
```

## 0. What I found in your project (read this first)

| Finding | Why it matters |
|---|---|
| Part is `xc7z020clg400-1` (PYNQ-Z2 compatible). PL clock is **50 MHz**, DMA is 32-bit, no scatter-gather | Raise FCLK0 to 100 MHz and widen the DMA/HP path to 64-bit before benchmarking |
| The block design contains PS7, AXI DMA, GPIO, SmartConnect — **no accelerator instance**; the DMA stream ports are unconnected | The board currently cannot run your network. `cifar10_snn_accelerator.v` is a pass-through placeholder |
| `conv3x3_int8_engine.v` is referenced from `C:/Users/samwi/Downloads/` — **it is not inside the project folder** | Copy it into `…srcs/sources_1/new/` or it will be lost / fail on another PC |
| Only a toy 2x2 conv test passes | LIF, pooling, residual add, layer control, memory plan are not built yet |
| 1,959,616 INT8 weights = **~1.96 MB**; xc7z020 has ~0.63 MB of BRAM | Weights cannot all live on-chip. Stream them from DDR (or place only early layers on-chip) |
| Work per image: ~115 M accumulate-ops per timestep x 32 timesteps | Needs a parallel engine; plan on ~1 s/image order of magnitude (estimate — measure it) |
| The note says accuracy 77.12% "measured using random input spikes", but the DMA plan packs INT8 **pixels** | Input encoding is unresolved: Poisson spikes or analog pixels every step? Settle in step 1 |
| Training script / `.pth` are not in the zip | I rebuilt the network from your screenshots + `layer_config.json`; step 1 validates that |

Good news: your exported **Q16 BN files match my recomputation with 0 LSB error**, so BN folding and EPS=1e-5 are right.

## 1. Step-by-step plan (each gate gives you something presentable)

### Gate 1 — Make the software reference trustworthy (do first, ~half a day)
```bash
# get CIFAR-10 python version once: https://www.cs.toronto.edu/~kriz/cifar.html  -> cifar-10-batches-py/
cd golden
python snn_golden.py --root <path>/SNN_FPGA_V6_CIFAR10 --selftest
python snn_golden.py --root <...> --cifar <cifar-10-batches-py> --mode float --encoding poisson --n 500
python snn_golden.py --root <...> --cifar <cifar-10-batches-py> --mode float --encoding direct --norm cifar --n 500
```
The **float** result must land near your 77%. Whichever encoding/normalisation does is the one your model was
trained with — keep it for everything else. (If neither does, check how your training notebook decodes the
output: spike-count argmax vs. final membrane. The golden model uses spike-count argmax.)
Then run `--mode int8` and `--mode fixed` with the same settings: report the accuracy drop from INT8 and from
fixed-point. This is a real, citable result for your report.

### Gate 2 — Novelty experiment in software (~half a day)
```bash
python snn_golden.py ... --mode fixed --n 1000 --out results_fixed.npz
python early_exit.py results_fixed.npz        # -> early_exit_sweep.csv, early_exit_pareto.png
```
Because the network is causal, one 32-step run lets you replay every (t_min, margin) policy exactly.
Report: baseline accuracy, then the Pareto table "accuracy vs. average timesteps".

### Gate 3 — RTL building blocks (done for you; just run)
```bash
cd sim && ./run_sim.sh      # needs iverilog (sudo apt install iverilog)
```
Or add `rtl/*.v` + `sim/tb_*.v` to your Vivado project, copy `lif_vectors.txt` / `ee_vectors.txt` to the xsim
run directory, and run behavioural simulation.

### Gate 4 — Put something real on the board (the hard part)
Recommended scope, in order — stop at whatever level your deadline allows, and say so honestly in the report:
1. **ARM-only baseline on the PYNQ-Z2** — run `golden/` on the board for 2-3 images and time it. This is your
   "software baseline" for speedup numbers.
2. **Stage 1 on the PL** (conv0 + res1 = only ~75 KB of weights, fits in BRAM). Engine = your conv3x3 engine
   (spike inputs make it accumulate-only) + `lif_bn_unit` + maxpool + residual add. PS runs stages 2-3.
   Validate every layer's spikes against the golden model (dump golden spikes with `run_batch`).
3. **Whole network on the PL**, weights streamed from DDR through the DMA / HP port, time-major
   (one timestep through all layers, so early exit is exact). Stretch goal.
4. Instantiate `early_exit_ctrl` on the output-layer spikes; its `done` flag stops the timestep loop.
5. Before benchmarking: FCLK0 -> 100 MHz, DMA data width -> 64 bit, check timing closure (WNS >= 0).

### Gate 5 — Measurements to put in the report (measure; don't estimate)
| Metric | How |
|---|---|
| Accuracy: float / INT8 / fixed | `snn_golden.py` (Gate 1) |
| Accuracy vs avg timesteps (CGEE) | `early_exit.py` (Gate 2) |
| RTL == golden | `sim/run_sim.sh` PASS lines |
| Latency per image, with and without early exit | `time.perf_counter()` around the run on the board, >= 50 images |
| Resource use (LUT / FF / BRAM / DSP) | Vivado utilization report after implementation |
| Power | USB/inline power meter (idle vs. running) = *measured*; Vivado power report = *estimated* — label which |
| Spike sparsity per layer | `spk_*` arrays in the results `.npz` |

### Gate 6 — Make it impressive
* `python board/dashboard.py --root ... --cifar ... --index 3 --gif demo.gif` -> animated demo for slides.
* In a PYNQ notebook use `dashboard.live_jupyter(...)`: the audience watches the class race and the early exit.
* Fill the `latency` / `power` tiles with your *measured* values (they say "measure me" until you do).
* Poster / slides: architecture diagram, the Pareto plot, the resource table, the live demo, and a
  "verification" slide (golden vs RTL, 0 mismatches over 20,000 neuron updates + 3,000 images).

## 2. The novelty, stated honestly
**Confidence-Gated Early Exit (CGEE) for an INT8 fixed-point spiking ResNet on Zynq-7020**: stop the timestep
loop as soon as the output layer's top-1 spike count leads top-2 by >= M (after >= t_min steps). Hardware cost is a
few comparators; the benefit is fewer timesteps per image (latency and energy).

Dynamic-timestep / early-exit ideas already exist in the SNN literature, so do **not** claim you invented them.
Claim what is true and measurable: *a bit-exact verified hardware implementation, evaluated on your own network, with the full
accuracy-vs-timesteps trade-off on a Zynq-7020.* Search "early exit spiking neural network" /
"dynamic timestep SNN" and cite 2-3 papers in related work.

Secondary, optional additions (measure the sparsity first, it is in your `.npz` files): zero-skipping in the
accumulate engine, and an on-chip LFSR Poisson encoder so the image is sent once instead of 32 frames.

## 2b. Timing fix (v2 controller) - read if your Vivado timing report was negative
The first version of `early_exit_ctrl.v` passed simulation but **failed timing at 50 MHz (WNS -15.7 ns, 11 failing
endpoints = done + pred + steps_used)**: it found the leading class with a serial compare chain, then the runner-up
with a second serial chain, then subtracted. v2 evaluates "is class i ahead of all others by >= margin" for all
classes in parallel and takes the decision one clock after the counters update.
Same ports, same decisions (3,000-image test); `done/pred/steps_used` now appear 2 clocks (not 1) after `step_valid`
- invisible to the CPU, which reads much later. Estimated path: ~10 ns on a 20 ns clock (rough model, calibrated
against the failing report); **only Vivado's Report Timing Summary counts - WNS must be >= 0.**
Simulation cannot find this kind of problem: it does not model delays. Always check timing after implementation.

## 3. Verification ledger (what has actually been run)
| Item | Status |
|---|---|
| BN Q16 files vs recomputation | verified, 0 LSB difference |
| Fixed-point vs float-INT8 LIF pipeline | agrees to <0.5% spikes per layer on synthetic images |
| `lif_bn_unit.v` vs golden | 20,000 vectors pass; 4 deliberately injected bugs all caught |
| `early_exit_ctrl.v` (v2) vs golden | 3,000 images pass incl. back-to-back step pulses; 5 injected bugs caught (a 6th change was behaviour-equivalent) |
| Timing of v2 | **estimated only** (~10 ns path, rough model); real result = your Vivado timing report |
| `ee_axil.v` (AXI4-Lite wrapper) | 3,000 images driven through random-wait-state AXI reads/writes; AXI ordering monitors; 4 injected bugs all caught |
| `pynq_ee_demo.py` | logic tested against a software stand-in (`--mock`) only; **never run against real hardware** |
| Dashboard | rendered and visually checked (static PNG + GIF) |
| **Real CIFAR-10 accuracy of golden model** | **NOT run** (no dataset in my sandbox) — Gate 1 |
| **Anything on the physical board / Vivado synthesis / timing** | **NOT run** — I have no Vivado or board |
| Early-exit numbers on your network | **NOT available yet** — the "mean steps" printed by the testbench uses synthetic spike patterns |
