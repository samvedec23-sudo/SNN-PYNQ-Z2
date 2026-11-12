
import numpy as np
import os

# -------------------------------------------------
# 1. Load exported INT8 weights
# -------------------------------------------------

INPUT_FILE = os.path.join(
    "quantized_int8_weights",
    "fc_weights.bin"
)

OUTPUT_FILE = "fc_weights.mem"

# Expected FC weight shape: (10, 4096)
weights = np.fromfile(INPUT_FILE, dtype=np.int8)

# Verify the number of weights
if weights.size != 10 * 4096:
    raise ValueError(
        f"Expected 40960 weights, but found {weights.size}"
    )

weights = weights.reshape(10, 4096)

print("FC weights loaded successfully!")
print("Weight shape:", weights.shape)


# -------------------------------------------------
# 2. Generate Verilog $readmemh-compatible file
# -------------------------------------------------

with open(OUTPUT_FILE, "w") as f:

    for row in weights:
        for val in row:

            # Convert signed INT8 to 2-digit hexadecimal
            hex_value = int(val) & 0xFF

            f.write(f"{hex_value:02X}\n")


# -------------------------------------------------
# 3. Display results
# -------------------------------------------------

print("\nFC weight conversion completed!")
print("Output file:", os.path.abspath(OUTPUT_FILE))
print("Total weights exported:", weights.size)
print("File size:", os.path.getsize(OUTPUT_FILE), "bytes")

print("\nCreated fc_weights.mem!")
print("Ready for Vivado $readmemh initialization.")
