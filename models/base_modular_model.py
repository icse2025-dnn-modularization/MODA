import torch
from torch import nn

from models.model_utils import get_model_leaf_layers
from models.modular_utils import _hook_to_track_layer_outputs


class BaseModularModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._modular__tracking_layer_activations = False

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")

    def get_list_of_layers_need_to_track_act(self):
        return [(name, l) for name, l in get_model_leaf_layers(self, return_with_layer_name=True) if
                isinstance(l, nn.ReLU)]

    def enable_hook_to_track_layer_activation_values(self):
        self._modular__tracking_layer_activations = True
        act_tracking_layers = self.get_list_of_layers_need_to_track_act()
        for i, (name, layer) in enumerate(act_tracking_layers):
            # print(i, name, layer)
            layer.register_forward_hook(_hook_to_track_layer_outputs)

    def get_modular_activations_of_current_batch(self):
        act_tracking_layers = self.get_list_of_layers_need_to_track_act()
        assert self._modular__tracking_layer_activations, "layers' activations were not tracked"
        return [layer.modular_activations for name, layer in act_tracking_layers]
