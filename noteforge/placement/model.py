"""Small autoregressive placement model definition.

Architecture only: rhythm + previous positions -> next x bin. No audio.
The trained weights ship as ``placement3.pt``; the training pipeline is
private and not part of this repository.
"""
import numpy as np
import torch
import torch.nn as nn

BINS = np.linspace(-675, 675, 28)
NB = len(BINS) - 1


def feats(a):
    """a: [n,4] start,end,kind,x -> features per note"""
    t, e, k = a[:, 0], a[:, 1], a[:, 2].astype(int)
    n = len(a)
    prev = np.diff(t, prepend=t[0] - 2.)
    nxt = np.diff(t, append=t[-1] + 2.)
    f = np.zeros((n, 11), np.float32)
    f[:, 0] = np.log(prev + .01)
    f[:, 1] = np.log(nxt + .01)
    f[np.arange(n), 2 + k] = 1
    f[:, 6] = np.log(np.maximum(e - t, 0) + .01) * (k == 1)
    f[:, 7] = prev < .005
    f[:, 8] = nxt < .005
    ends = np.maximum.accumulate(e)
    f[:, 9] = (np.r_[0, ends[:-1]] > t + .03)  # a hold is still down
    lo = np.searchsorted(t, t - 1.)
    hi = np.searchsorted(t, t + 1.)
    f[:, 10] = np.log((hi - lo) / 2 + .5)
    return f


class M(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.prev = nn.Embedding(NB + 1, 48)
        self.inp = nn.Linear(11 + 48, h)
        self.g = nn.GRU(h, h, 2, batch_first=True, dropout=.3)
        self.o = nn.Linear(h, NB)

    def forward(self, f, pb, st=None):
        o, st = self.g(torch.relu(self.inp(torch.cat([f, self.prev(pb)], -1))), st)
        return self.o(o), st
