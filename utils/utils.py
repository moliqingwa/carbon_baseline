from typing import List, Any

import os
import math
import random

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = OrderedDict()
    d["median"] = np.median(array)
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d


def to_tensor(value: Any, raise_error=True) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    elif isinstance(value, (tuple, list)):
        if torch.is_tensor(value[0]):
            return torch.stack(value)
        elif isinstance(value[0], np.ndarray):
            return torch.tensor(value)
        else:
            try:
                return torch.tensor(value)
            except Exception as ex:
                pass
    else:
        pass
    if raise_error:
        raise TypeError("not support item type: {}".format(type(value)))
    return None


def init_(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.)
    return module


def flatten(data: list) -> list:
    """
    Flatten embedded list to list.

    :param data: the data to be flattened
    :return: The flattened list.

    :note: numpy.ndarray/torch.Tensor will NOT be flattened
    """
    new_data = []
    for value in data:
        if isinstance(value, list):
            new_data.extend(flatten(value))
        else:
            new_data.append(value)
    return new_data


def calculate_gard_norm(parameters: List) -> float:
    sum_grad = 0
    for x in parameters:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_folders_if_necessary(path):
    if not os.path.isdir(str(path)):
        os.makedirs(path)
