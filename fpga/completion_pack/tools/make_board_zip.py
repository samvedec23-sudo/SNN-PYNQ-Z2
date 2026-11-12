#!/usr/bin/env python3
"""
make_board_zip.py - builds ONE zip with everything the PYNQ board needs (small enough for Jupyter's upload button).

  python make_board_zip.py --root "D:\\...\\SNN_FPGA_V6_CIFAR10" --cifar "D:\\...\\cifar-10-batches-py" --n 500

Creates snn_board_files.zip containing:
  golden/ board/                         (this pack's code)
  weights/weights_float32, weights/weights_int8_experiment   (the only two folders the model reads)
  cifar/test_batch                       (first N test images in a compact, Python-version-independent format)
"""
import argparse, pathlib, pickle, zipfile
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="SNN_FPGA_V6_CIFAR10 folder (has weights_float32 and weights_int8_experiment)")
ap.add_argument("--cifar", required=True, help="cifar-10-batches-py folder (has test_batch)")
ap.add_argument("--n", type=int, default=500, help="how many test images to put on the board")
ap.add_argument("--out", default="snn_board_files.zip")
a = ap.parse_args()

pack = pathlib.Path(__file__).resolve().parents[1]
root = pathlib.Path(a.root)
for need in ("weights_float32", "weights_int8_experiment"):
    if not (root / need).is_dir():
        raise SystemExit(f"cannot find {root / need} - check --root")

d = pickle.load(open(pathlib.Path(a.cifar) / "test_batch", "rb"), encoding="bytes")
small = {b"data": np.asarray(d[b"data"][:a.n], np.uint8).tobytes(), b"labels": [int(v) for v in d[b"labels"][:a.n]]}
tmp = pathlib.Path("test_batch.tmp")
pickle.dump(small, open(tmp, "wb"), protocol=2)               # protocol 2: readable by any Python on the board

with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("golden", "board"):
        for f in (pack / sub).rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(pack).as_posix())
    for sub in ("weights_float32", "weights_int8_experiment"):
        for f in (root / sub).rglob("*"):
            if f.is_file():
                z.write(f, f"weights/{sub}/{f.relative_to(root / sub).as_posix()}")
    z.write(tmp, "cifar/test_batch")
tmp.unlink()
size = pathlib.Path(a.out).stat().st_size / 1e6
print(f"wrote {a.out}  ({size:.1f} MB)  - upload this ONE file with Jupyter's Upload button")
if size > 24:
    print("WARNING: bigger than ~25 MB; Jupyter may refuse it. Lower --n.")
