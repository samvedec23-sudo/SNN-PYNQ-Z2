from pathlib import Path
import torch

PROJECT_DIR = Path(r"C:\SNN_project")
CHECKPOINT_PATH = PROJECT_DIR / "latest_v4_checkpoint.pth"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

state = checkpoint["model_state_dict"]

print("Saved epoch:", checkpoint["epoch"])
print("Training accuracy:", checkpoint.get("train_accuracy"))
print("\nModel parameters and shapes:\n")

for name, tensor in state.items():
    if torch.is_tensor(tensor):
        print(f"{name:45s} {tuple(tensor.shape)}")
    else:
        print(f"{name:45s} [non-tensor value]")

print("\nTotal entries:", len(state))