import numpy as np
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, input_dim=8, hidden=16):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return x


def test_hook_capture_last_mean_max():
    model = TinyModel()
    captured = {}

    def hook(module, inp, out):
        # out is (batch, features)
        captured['out'] = out.detach().numpy()

    handle = model.linear2.register_forward_hook(hook)
    x = torch.randn(4, 8)
    out = model(x)
    handle.remove()

    assert 'out' in captured
    assert captured['out'].shape == (4, 16)
