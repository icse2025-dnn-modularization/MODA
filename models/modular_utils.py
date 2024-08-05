import inspect
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _hook_to_track_layer_outputs(module, input, output):
    module.modular_activations = output


@torch.no_grad()
def get_activation_rate_during_inference(model, data_loader, num_classes, device):
    model.to(device)
    model.eval()

    # Initialize a dictionary to hold activation counts for each class, for each layer
    all_activation_value_dict = {c: {} for c in range(num_classes)}
    all_labels = []

    pbar = tqdm(data_loader, ncols=80, desc="Inference")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        model(images)
        modular_activations = model.get_modular_activations_of_current_batch()

        # calculate number of times that a neuron activated during inference
        for layer_idx, act_val in enumerate(modular_activations):
            if len(act_val.shape[1:]) == 3:  # For Conv2d layers
                act_val = act_val.abs().mean(dim=(2, 3))

            act_val_bin = (act_val != 0).int()  # Activated or not

            # Sum activations for all classes at once
            for c in range(num_classes):
                label_mask = (labels == c).unsqueeze(1)
                act_counts = torch.where(label_mask, act_val_bin, 0).sum(dim=0)

                if layer_idx not in all_activation_value_dict[c]:
                    all_activation_value_dict[c][layer_idx] = act_counts
                else:
                    all_activation_value_dict[c][layer_idx] += act_counts

        all_labels.append(labels)

    # Post-process to calculate average activation rates
    # calculate average act rate in percentage
    flatten_labels = torch.cat(all_labels)
    unique_labels, label_counts = flatten_labels.unique(return_counts=True)
    for c, count in zip(unique_labels, label_counts):
        for layer_idx, layer_act_val in all_activation_value_dict[c.item()].items():
            all_activation_value_dict[c.item()][layer_idx] = layer_act_val.float() / count

    return all_activation_value_dict, flatten_labels


def calculate_module_losses(layer_activation_values, labels):
    """
    Inherit from https://github.com/qibinhang/MwT/blob/3d29e7eb2c1505c8d3b3e112b8dd5a06fd49fd9b/src/modular_trainer.py#L152
    """

    sum_dispersion = 0.0
    sum_affinity = 0.0
    sum_compactness = 0.0

    same_label_mask = (labels[:, None] == labels[None, :])
    valid_sample_pair_mask = torch.triu(torch.ones_like(same_label_mask,
                                                        device=labels.device,
                                                        dtype=torch.bool), diagonal=1)
    # flattening indices ".view(-1).nonzero().squeeze()"
    # is just for performance optimization of indexing
    # for understanding the rationale behind, just ignore the flattening part
    same_label_indices = (valid_sample_pair_mask & same_label_mask).view(-1).nonzero().squeeze()
    diff_label_indices = (valid_sample_pair_mask & ~same_label_mask).view(-1).nonzero().squeeze()

    for curr_layer_act in layer_activation_values:
        if len(curr_layer_act.shape[1:]) == 3:  # reshape Conv2d layer output
            # consider a feature map as a neuron (by averaging its act values)
            transformed_layer_act = curr_layer_act.abs().mean(dim=(2, 3))
        else:
            transformed_layer_act = curr_layer_act

        norm_acts = F.normalize(transformed_layer_act, p=2, dim=1)
        act_sim = torch.matmul(norm_acts, norm_acts.T).view(-1)

        sum_affinity += act_sim.index_select(0, same_label_indices).mean()
        sum_dispersion += act_sim.index_select(0, diff_label_indices).mean()
        sum_compactness += transformed_layer_act.norm(p=1) / transformed_layer_act.numel()

    num_layers = len(layer_activation_values)
    loss_affinity = 1 - sum_affinity / num_layers
    loss_dispersion = sum_dispersion / num_layers
    loss_compactness = sum_compactness / num_layers
    return loss_affinity, loss_dispersion, loss_compactness


def group_mean_list(samples, labels):
    if len(samples.shape) == 1:
        samples = samples.unsqueeze(-1)
    M = torch.zeros(labels.max() + 1, len(samples)).cuda()
    M[labels, torch.arange(len(samples))] = 1
    M = F.normalize(M, p=1, dim=1)
    return torch.mm(M, samples).mean()


def mean_list(input_list):
    if len(input_list) == 0:
        return torch.tensor(0)
    result = sum(input_list) / len(input_list)
    if isinstance(result, torch.Tensor):
        result = result.item()
    return result


def _get_function_params(method, include_self=False):
    signature = inspect.signature(method)
    params = signature.parameters
    param_names = list(params.keys())
    if not include_self:
        param_names.remove("self")
    return param_names


def clone_and_modify_layer_structure(original_layer, *args, **kwargs):
    layer_cls = type(original_layer)

    default_attrs = {key: value for key, value in original_layer.__dict__.items()}
    if hasattr(original_layer, 'bias'):
        default_attrs['bias'] = original_layer.bias is not None

    accepting_kwarg_names = _get_function_params(layer_cls.__init__, include_self=False)
    accepting_default_attrs = {k: v for k, v in default_attrs.items() if k in accepting_kwarg_names}

    for d_attr, d_attr_value in accepting_default_attrs.items():
        if d_attr not in kwargs:
            kwargs[d_attr] = d_attr_value
    new_layer = layer_cls(*args, **kwargs)
    return new_layer
