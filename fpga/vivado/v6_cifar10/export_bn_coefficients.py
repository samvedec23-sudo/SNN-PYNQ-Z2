
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PARAMS = ROOT / "hardware_export"
OUTPUT = ROOT / "hardware_params" / "bn_coefficients"

# PyTorch BatchNorm formula:
# y = scale * x + offset
# scale  = gamma / sqrt(running_var + eps)
# offset = beta - running_mean * scale

# Use the exact eps value from the trained model.
# This default is PyTorch BatchNorm2d's usual default,
# but confirm it matches your original model before use.
EPS = 1e-5

OUTPUT.mkdir(parents=True, exist_ok=True)

bn_names = [
    "bn0", "bn1", "bn2",
    "res1__bn1", "res1__bn2",
    "res2__bn1", "res2__bn2",
    "res3__bn1", "res3__bn2",
]

manifest = {}

for name in bn_names:
    prefix = name + "__"

    gamma = np.load(PARAMS / f"{prefix}weight.npy")
    beta = np.load(PARAMS / f"{prefix}bias.npy")
    mean = np.load(PARAMS / f"{prefix}running_mean.npy")
    var = np.load(PARAMS / f"{prefix}running_var.npy")

    scale = gamma / np.sqrt(var + EPS)
    offset = beta - mean * scale

    np.save(OUTPUT / f"{name}__scale.npy", scale.astype(np.float32))
    np.save(OUTPUT / f"{name}__offset.npy", offset.astype(np.float32))

    manifest[name] = {
        "channels": int(gamma.size),
        "eps": EPS,
        "scale_file": f"{name}__scale.npy",
        "offset_file": f"{name}__offset.npy",
        "dtype": "float32",
        "status": "floating-point coefficients; not yet fixed-point"
    }

    print(f"{name}: {gamma.size} channels exported")

with open(OUTPUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nOutput directory: {OUTPUT}")
print("IMPORTANT: confirm EPS matches the trained model.")
print("These coefficients are not yet FPGA-ready fixed-point values.")
