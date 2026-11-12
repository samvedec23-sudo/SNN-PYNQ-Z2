
import numpy as np
import os

# Paths relative to this Python script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "hardware_export")
OUTPUT_DIR = os.path.join(BASE_DIR, "quantized_int8_weights")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_weight(filename):
    path = os.path.join(EXPORT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    return np.load(path)


def fuse_bn_into_conv(w_conv, gamma, beta, mean, var, eps=1e-5):
    """
    Fuse convolution weights with Batch Normalization.

    W_fused = W * (gamma / sqrt(var + eps))
    Bias_fused = beta - gamma * mean / sqrt(var + eps)
    """

    std = np.sqrt(var + eps)
    scale = gamma / std

    scale_reshaped = scale.reshape(-1, 1, 1, 1)

    w_fused = w_conv * scale_reshaped
    b_fused = beta - (scale * mean)

    return w_fused, b_fused


def quantize_to_int8(tensor, scale=64.0):
    """
    Quantize floating-point weights to signed INT8.
    Range: -128 to +127
    """

    scaled = np.round(tensor * scale)

    return np.clip(scaled, -128, 127).astype(np.int8)


def process_conv_bn(conv_name, bn_name, output_name):
    print(f"\nProcessing {conv_name}...")

    conv_w = load_weight(f"{conv_name}__weight.npy")

    bn_w = load_weight(f"{bn_name}__weight.npy")
    bn_b = load_weight(f"{bn_name}__bias.npy")
    bn_m = load_weight(f"{bn_name}__running_mean.npy")
    bn_v = load_weight(f"{bn_name}__running_var.npy")

    w_fused, b_fused = fuse_bn_into_conv(
        conv_w, bn_w, bn_b, bn_m, bn_v
    )

    w_int8 = quantize_to_int8(w_fused)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    w_int8.tofile(output_path)

    bias_name = output_name.replace("_weights.bin", "_bias_fused.npy")
    np.save(os.path.join(OUTPUT_DIR, bias_name), b_fused)

    print(f"  Shape: {w_int8.shape}")
    print(f"  INT8 weight size: {w_int8.nbytes / 1024:.2f} KB")
    print(f"  Saved: {output_name}")
    print(f"  Saved: {bias_name}")


# ============================================================
# 1. CONV0 + BN0
# ============================================================

print("=" * 60)
print("SNN CIFAR-10: BN FUSION AND INT8 QUANTIZATION")
print("=" * 60)

process_conv_bn(
    "conv0",
    "bn0",
    "conv0_weights.bin"
)


# ============================================================
# 2. RESIDUAL BLOCKS: RES1, RES2, RES3
# ============================================================

stages = ["res1", "res2", "res3"]

for stage in stages:

    for sub in ["1", "2"]:

        process_conv_bn(
            f"{stage}__conv{sub}",
            f"{stage}__bn{sub}",
            f"{stage}_conv{sub}_weights.bin"
        )


# ============================================================
# 3. FULLY CONNECTED CLASSIFIER
# ============================================================

print("\nProcessing FC layer...")

fc_w = load_weight("fc__weight.npy")
fc_b = load_weight("fc__bias.npy")

fc_w_int8 = quantize_to_int8(fc_w)

fc_w_int8.tofile(
    os.path.join(OUTPUT_DIR, "fc_weights.bin")
)

np.save(
    os.path.join(OUTPUT_DIR, "fc_bias.npy"),
    fc_b
)

print(f"  Shape: {fc_w_int8.shape}")
print(f"  INT8 weight size: {fc_w_int8.nbytes / 1024:.2f} KB")
print("  Saved: fc_weights.bin")
print("  Saved: fc_bias.npy")


# ============================================================
# 4. FINAL MEMORY SUMMARY
# ============================================================

total_bytes = 0

for filename in os.listdir(OUTPUT_DIR):

    if filename.endswith("_weights.bin"):
        file_path = os.path.join(OUTPUT_DIR, filename)
        size = os.path.getsize(file_path)

        total_bytes += size

print("\n" + "=" * 60)
print("QUANTIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Total INT8 weight storage: {total_bytes / 1024:.2f} KB")
print(f"Total INT8 weight storage: {total_bytes / (1024*1024):.3f} MB")

print(f"\nOutput directory: {OUTPUT_DIR}")
print("=" * 60)
