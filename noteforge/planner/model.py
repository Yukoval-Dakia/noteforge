"""Density planner: audio cells -> 2 s density curve and song style. Generation is unchanged; this only fills in the request."""
import torch
from torch import nn

EPS = .25
IN = 128+16+16+3


class Planner(nn.Module):
    def __init__(self, width=192, layers=6, dropout=.1):
        super().__init__()
        self.input = nn.Linear(IN+3, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(width, width, 5, padding=2*2**(k % 4), dilation=2**(k % 4)), nn.GELU(), nn.Dropout(dropout)) for k in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.rnn = nn.GRU(width, width//2, batch_first=True, bidirectional=True)
        self.curve = nn.Linear(width, 1)
        self.song = nn.Linear(width, 5)  # level (log mean heads/s) + style logits

    def forward(self, cells, mask, is_phira, level, has_level):
        b, n, _ = cells.shape
        side = torch.stack([is_phira.float(), level*has_level.float(), has_level.float()], 1)[:, None].expand(b, n, 3)
        x = self.input(torch.cat([cells, side], -1))
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x+block(x.transpose(1, 2)).transpose(1, 2))
        x = x+self.rnn(x)[0]
        pooled = (x*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        song = self.song(pooled)
        return self.curve(x)[..., 0], song[:, 0], song[:, 1:]
