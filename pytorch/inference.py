import json
from pathlib import Path
import torch

try:
    from .model import CTRNet
except ImportError:
    from model import CTRNet



def load_model(model_dir="pytorch/models", meta_name="metagkf.json", model_name="modelgkf.pt", device="cpu"):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / meta_name).read_text(encoding="utf-8"))
    device = torch.device(device)

    model = CTRNet(
        meta["cardinalities"],
        emb_dim=meta["arch"]["emb_dim"],
        hidden=tuple(meta["arch"]["hidden"]),
        dropout=meta["arch"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / model_name, map_location=device))
    model.eval()
    return model, meta, device