import torch
from torch import nn
from torch.functional import F

from mask_utils import MaskLinear
from fcn_model import LeNet300


class MaskedLeNet300(LeNet300):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super().__init__(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate)

        self.fc2 = MaskLinear(300, 200)
        self.fc3 = MaskLinear(200, 100)

    def forward(self, x):
        out = super().forward(x)
        self.activations = self.get_masks()
        return out

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")

    def get_masks(self):
        masks = {
            m_name: m_object.masks for m_name, m_object in self._modules.items() if hasattr(m_object, 'masks')
        }
        return masks
