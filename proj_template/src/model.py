import torch
from torch import nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self) -> None:
        super(NNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def _init_params(self):
        pass

    def __str__(self) -> str:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad == True)
        return f'\nModel\n\tnum params: {num_params}\n'