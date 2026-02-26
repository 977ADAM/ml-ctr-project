import json
from pathlib import Path
import torch
import numpy as np

try:
    from .model import CTRNet
    from .config import Config
except ImportError:
    from model import CTRNet
    from config import Config



def load_model(model_dir=Config.MODEL_DIR, meta_name=Config.META_NAME, model_name=Config.MODEL_NAME, device="cpu"):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / meta_name).read_text(encoding="utf-8"))
    device = torch.device(device)

    model = CTRNet(
        meta["cardinalities"],
        emb_dim=meta["arch"]["emb_dim"],
        hidden=tuple(meta["arch"]["hidden"]),
        dropout=meta["arch"]["dropout"],
    ).to(device)
    print(model_dir)
    print(model_dir / meta_name)
    print(model_dir / model_name)
    model.load_state_dict(torch.load(model_dir / model_name, map_location=device))
    model.eval()
    return model, meta, device

def encode_row(row: dict, meta: dict) -> np.ndarray:
    x = []
    for col in meta["cat_cols"]:
        classes = meta["mappings"][col]["classes"]
        v = str(row.get(col, ""))
        # unknown -> 0 (можно сделать отдельный UNK, но для простоты так)
        idx = classes.index(v) if v in classes else 0
        x.append(idx)
    return np.array(x, dtype=np.int64)