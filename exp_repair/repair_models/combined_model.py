import torch
from torch import nn
import torch.nn.functional as F

from models.model_utils import get_model_leaf_layers


class RepairedModel(nn.Module):
    def __init__(self, weak_model, strong_model, patch_output_index):
        super(RepairedModel, self).__init__()
        # putting inputting models into a dict to make sure their params are not included during training
        self.inputting_model_dict = {
            "weak_model": weak_model,
            "strong_model": strong_model,
        }
        self._freeze_inputting_models()

        # initialize a calibration (FC) layer to make two models' outputs compatible
        weak_model__last_layer = get_model_leaf_layers(weak_model, return_with_layer_name=False)[-1]
        assert isinstance(weak_model__last_layer, nn.Linear)
        weak_model__out_dim = weak_model__last_layer.out_features
        self.patch_output_mask = torch.zeros(weak_model__out_dim, dtype=torch.bool)
        self.patch_output_mask[patch_output_index] = True

        self.calibration_layer = nn.Linear(weak_model__out_dim, weak_model__out_dim)

        self._initialize_weights()

    def transform_input(self, x):
        with torch.no_grad():
            weak_model = self.inputting_model_dict["weak_model"]
            out_weak = weak_model(x).detach()

            strong_model = self.inputting_model_dict["strong_model"]
            out_strong = strong_model(x).detach()

            return out_weak, out_strong

    def forward(self, x):
        out_weak, out_strong = self.transform_input(x)
        final_out = torch.cat((out_weak[:, ~self.patch_output_mask], out_strong), dim=1)
        final_out = self.calibration_layer(final_out)
        return final_out

    def _freeze_inputting_models(self):
        weak_model = self.inputting_model_dict["weak_model"]
        strong_model = self.inputting_model_dict["strong_model"]

        weak_model.eval()
        for param in weak_model.parameters():
            param.requires_grad = False

        strong_model.eval()
        for param in strong_model.parameters():
            param.requires_grad = False

    def _initialize_weights(self):
        layers = get_model_leaf_layers(self, return_with_layer_name=False)
        assert len(layers) == 1
        calibration_layer = layers[0]
        nn.init.normal_(calibration_layer.weight, 0, 0.01)
        nn.init.constant_(calibration_layer.bias, 0)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        for _, inputting_model in self.inputting_model_dict.items():
            inputting_model.to(device)
        self.patch_output_mask = self.patch_output_mask.to(device)
