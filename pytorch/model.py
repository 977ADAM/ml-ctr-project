import math
import torch
import torch.nn as nn

# ---------- model ----------
class CTRNet(nn.Module):
    """
    Эмбеддинги для категориальных фич + MLP -> logit(p).
    """
    def __init__(self, cardinalities, emb_dim=16, hidden=(64, 32), dropout=0.1):
        super().__init__()

        self.embs = nn.ModuleList()
        emb_out = 0
        for c in cardinalities:
            d = min(emb_dim, int(math.ceil(c ** 0.25) * 4))  # норм эвристика
            self.embs.append(nn.Embedding(c, d))
            emb_out += d

        layers = []
        in_dim = emb_out
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        z = torch.cat(embs, dim=1)
        logit = self.mlp(z).squeeze(1)
        return logit