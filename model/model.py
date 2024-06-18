import torch
from torch import nn

class OurModel (nn.Module):
    def __init__ (self, config):
        super().__init__()

        self.config = config

    