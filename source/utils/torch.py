"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import math
import random
from typing import Callable
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torch.backends.cudnn as cudnn

# Distributed Data Parallel Utilities
def all_reduce_tensor(tensor: torch.Tensor, world_size=1):
    r"""Average reduce a tensor across all_no_processing_of_pts workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_tensors(x: torch.Tensor, world_size=1):
    r"""Average reduce all_no_processing_of_pts tensors across all_no_processing_of_pts workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x


# Common Utilities
def initialize(seed=None, cudnn_deterministic=True, autograd_anomaly_detection=False):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)
    return 


def release_cuda(x):
    r"""Release all_no_processing_of_pts tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def to_cuda(X, device):
    r"""move tensors to device in-place"""
    if isinstance(X,dict):
        for k,v in X.items():
            X[k] = to_cuda(v,device)
    elif isinstance(X,list):
        for i,e in enumerate(X):
            X[i] = to_cuda(e,device)
    elif isinstance(X,tuple) and hasattr(X,"_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = to_cuda(dd,device)
        return type(X)(**dd)
    elif isinstance(X,torch.Tensor):
        return X.to(device=device)
    return X


# logging 
def get_print_format(value):
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.3f'


def get_format_strings(kv_pairs):
    r"""Get format string for a list of key-value pairs."""
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings


def get_log_string(result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, timer=None):
    log_strings = []
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())
    message = ', '.join(log_strings)
    return message


def load_weights(model, snapshot):
    r"""Load weights and check keys."""
    state_dict = torch.load(snapshot)
    model_dict = state_dict['model']
    model.load_state_dict(model_dict, strict=False)

    snapshot_keys = set(model_dict.keys())
    model_keys = set(model.model_dict().keys())
    missing_keys = model_keys - snapshot_keys
    unexpected_keys = snapshot_keys - model_keys

    return missing_keys, unexpected_keys


# Learning Rate Scheduler

def exponentiel_lr_starting_at_x(opt, optimizer):
    """Learning rate used in nerf pytorch. Exponentially decayed starting from opt.optim.start_decrease. 

    Args:
        opt (edict): settings
        optimizer (torch.optim): optimizer
    """
    
    max_iter = opt.optim.max_iter if hasattr(opt.optim, 'max_iter') else opt.max_iter
    gamma =  (opt.optim.lr_end/opt.optim.lr)**(1./max_iter)
    start = opt.optim.start_decrease
    lr_fn = lambda epoch: 1. if epoch <= start else gamma ** (epoch - opt.optim.start_decrease)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


class CosineAnnealingFunction(Callable):
    def __init__(self, max_epoch, eta_min=0.0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        next_epoch = last_epoch + 1
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * next_epoch / self.max_epoch))


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step):
        # last_step starts from -1, which means last_steps=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        else:
            if next_step > self.total_steps:
                return self.eta_min
            next_step -= self.warmup_steps
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * next_step / self.normal_steps))


def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler
