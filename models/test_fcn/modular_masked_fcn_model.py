import torch
from torch import nn
from torch.functional import F

from masked_fcn_model import MaskedLeNet300


class ModularMaskedLeNet300(MaskedLeNet300):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super().__init__(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        self.reset_mask()

    def load_pretrained_weights(self, checkpoint_path):
        super().load_pretrained_weights(checkpoint_path)
        self.fc2 = self.fc2.fc
        self.fc3 = self.fc3.fc

    def reset_mask(self):
        self.mask_fc2 = torch.ones(self.fc2.out_features, device="cuda")
        self.mask_fc3 = torch.ones(self.fc3.out_features, device="cuda")

    def forward(self, x):
        self.activations = {}

        out = torch.flatten(x, 1)
        out = self.act_fn(self.fc1(out))

        out = self.act_fn(self.fc2(out))
        out = out * self.mask_fc2

        out = self.act_fn(self.fc3(out))
        out = out * self.mask_fc3

        out = self.fc4(out)
        return out
