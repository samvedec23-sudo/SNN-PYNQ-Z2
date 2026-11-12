
from pathlib import Path
import json
import numpy as np

ROOT = Path(r"D:\SNN_Project\SNN_FPGA_V6_CIFAR10")
SOURCE = ROOT / "weights_int8_experiment"
OUTPUT = ROOT / "hardware_params" / "int8_mem"

OUTPUT.mkdir(parents=True, exist_ok=True)

manifest_path = SOURCE / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

summary = []

for path in sorted(SOURCE.glob("*.npy")):
    weights = np.load(path, allow_pickle=False)

    if weights.dtype != np.int8:
        raise TypeError(
            f"{path.name}: expected int8 weights, got {weights.dtype}"
        )

    flat = weights.reshape(-1)
    mem_path = OUTPUT / f"{path.stem}.mem"

    with mem_path.open("w", encoding="ascii", newline="\n") as f:
        for value in flat:
            f.write(f"{int(value) & 0xFF:02X}\n")

    summary.append({
        "source": path.name,
        "memory_file": mem_path.name,
        "shape": list(weights.shape),
        "elements": int(weights.size),
        "dtype": "int8",
        "encoding": "one signed two's-complement byte per line",
    })

(OUTPUT / "memory_manifest.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print(f"Weight memory files generated: {len(summary)}")
print(f"Output directory: {OUTPUT}")
for item in summary:
    print(f"{item['memory_file']}: {item['elements']} weights")
