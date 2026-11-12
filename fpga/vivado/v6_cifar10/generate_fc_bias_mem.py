
import numpy as np
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Input: floating-point FC biases
INPUT_FILE = BASE_DIR / "quantized_int8_weights" / "fc_bias.npy"

# Output: Verilog memory initialization file
OUTPUT_FILE = BASE_DIR / "fc_bias.mem"

# Load trained biases
bias = np.load(INPUT_FILE)

print("Original bias shape:", bias.shape)
print("Original biases:", bias)

# Expected 10 output classes
if bias.size != 10:
    raise ValueError(
        f"Expected 10 FC biases, found {bias.size}"
    )

# Convert floating-point biases to signed INT16
SCALE = 16.0

bias_q = np.clip(
    np.round(bias * SCALE),
    -32768,
    32767
).astype(np.int16)

# Write hexadecimal values
with open(OUTPUT_FILE, "w") as f:
    for val in bias_q:
        f.write(f"{int(val) & 0xFFFF:04X}\n")

print("\nQuantized biases:", bias_q)
print("Generated:", OUTPUT_FILE)
print("Number of entries:", bias_q.size)
print("File size:", OUTPUT_FILE.stat().st_size, "bytes")
print("\nfc_bias.mem generated successfully!")
