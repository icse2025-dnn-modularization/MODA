import torch
from torch import nn
from torch.functional import F

from fcn_model import LeNet300


class ModularLeNet300(LeNet300):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super().__init__(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        self.reset_mask()

    def reset_mask(self):
        self.mask_fc1 = torch.ones(self.fc1.out_features, device="cuda")
        self.mask_fc2 = torch.ones(self.fc2.out_features, device="cuda")
        self.mask_fc3 = torch.ones(self.fc3.out_features, device="cuda")

    def forward(self, x):
        self.activations = {}

        out = torch.flatten(x, 1)
        out = self.act_fn(self.fc1(out))
        out = out * self.mask_fc1

        out = self.act_fn(self.fc2(out))
        out = out * self.mask_fc2

        out = self.act_fn(self.fc3(out))
        out = out * self.mask_fc3

        out = self.fc4(out)
        return out
