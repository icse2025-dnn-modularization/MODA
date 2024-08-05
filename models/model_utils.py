import time
from datetime import datetime
import os
import sys

import torch
import numpy as np
import random
from prettytable import PrettyTable


def get_runtime_device():
    return torch.device(torch.device('cuda') if torch.cuda.is_available() else "cpu")


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # slow & deterministic
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # # fast & non-deterministic
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms(True)


def redirect_stdout_to_file(file_dir, file_name):
    out_log_f = open(os.path.join(file_dir, file_name), "w")
    sys.stderr = out_log_f
    sys.stdout = out_log_f
    return out_log_f


def get_current_human_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_timestamp():
    return int(time.time())


def get_model_leaf_layers(model, return_with_layer_name=True):
    leaf_layers = []
    for name, layer in model.named_modules():
        if len(list(layer.children())) > 0:
            continue
        if return_with_layer_name:
            leaf_layers.append((name, layer))
        else:
            leaf_layers.append(layer)
    return leaf_layers


def print_model_summary(model):
    columns = ["Modules", "Parameters", "Param Shape"]
    table = PrettyTable(columns)
    for i, col in enumerate(columns):
        if i == 0:
            table.align[col] = "l"
        else:
            table.align[col] = "r"
    total_param_nums = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_nums = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, "{:,}".format(param_nums), "{}".format(param_shape)])
        total_param_nums += param_nums

    separator = ["-" * len(x) for x in table.field_names]
    table.add_row(separator)
    table.add_row(["Total", "{:,}".format(total_param_nums), "{}".format("_")])

    print(table, "\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item
