cat > sam_nir/smoke_test_sam_nir.py <<'PY'
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sam_nir.sam_encoder_model import SAMNIRModel

def main():
    sam_ckpt = os.path.join(ROOT_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    print("checkpoint exists:", os.path.exists(sam_ckpt))

    model = SAMNIRModel(
        sam_ckpt_path=sam_ckpt,
        lora_rank=4,
        freeze_encoder=True,
    ).to(device)

    x = torch.randn(1, 3, 1024, 1024, device=device)

    with torch.no_grad():
        y = model(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)

if __name__ == "__main__":
    main()
PY