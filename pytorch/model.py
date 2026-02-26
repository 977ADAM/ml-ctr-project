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

# ---------- DeepFM model ----------
class DeepFM(nn.Module):

    def __init__(self, cardinalities, emb_dim=16, hidden=(64, 32), dropout=0.1):
        super().__init__()

        self.num_fields = len(cardinalities)

        # -------- Linear part --------
        self.linear_embs = nn.ModuleList([
            nn.Embedding(c, 1) for c in cardinalities
        ])

        # -------- FM + Deep embeddings --------
        self.embs = nn.ModuleList()
        for c in cardinalities:
            self.embs.append(nn.Embedding(c, emb_dim))

        # -------- Deep part (MLP) --------
        input_dim = self.num_fields * emb_dim
        layers = []
        in_dim = input_dim

        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h

        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat):
        # ===== Linear part =====
        linear_terms = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.linear_embs)
        ]
        linear_logit = torch.sum(torch.cat(linear_terms, dim=1), dim=1)

        # ===== Embeddings =====
        embs = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embs)
        ]
        embs = torch.stack(embs, dim=1)  # (B, F, D)

        # ===== FM part =====
        sum_square = torch.sum(embs, dim=1) ** 2
        square_sum = torch.sum(embs ** 2, dim=1)
        fm_logit = 0.5 * torch.sum(sum_square - square_sum, dim=1)

        # ===== Deep part =====
        deep_input = embs.view(embs.size(0), -1)
        deep_logit = self.mlp(deep_input).squeeze(1)

        # ===== Final logit =====
        logit = linear_logit + fm_logit + deep_logit

        return logit