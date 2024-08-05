import torch
from torch import nn


class MaskGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskGenerator, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(input_dim, output_dim if output_dim < 128 else output_dim // 2, bias=False),
            nn.ReLU(True),
            nn.Linear(output_dim if output_dim < 128 else output_dim // 2, output_dim)
        )

        # for m in self.mask_generator.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)  # kaiming_normal_ is good
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 1.0)  # 1.0 is good

    def forward(self, x):
        mask = self.mask_generator(x)
        mask = torch.tanh(mask)
        mask = torch.relu(mask)

        return mask


class MaskLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskLinear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.mask_generator = MaskGenerator(input_dim, output_dim)
        self.in_features = input_dim
        self.out_features = output_dim

    def forward(self, x):
        out = self.fc(x)
        self.masks = self.mask_generator(x)
        out = out * self.masks
        return out
