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

import torch
import os
from pathlib import Path
import importlib
import inspect
import glob

from typing import List, Dict, Any, Optional

import source.admin.settings as ws_settings


def load_pretrained(module: str, name: str, checkpoint: str = None, **kwargs):
    """Load a network trained using the LTR framework. This is useful when you want to initialize your new network with
    a previously trained model.
    args:
        module  -  Name of the train script module. I.e. the name of the folder in ltr/train_scripts.
        name  -  The name of the train_script.
        checkpoint  -  You can supply the checkpoint number or the full path to the checkpoint file (see load_network).
        **kwargs  -  These are passed to load_network (see that function).
    """

    settings = ws_settings.Settings()
    network_dir = os.path.join(settings.env.workspace_dir, 'checkpoints', module, name)
    return load_network(network_dir=network_dir, checkpoint=checkpoint, **kwargs)


def load_checkpoint(network_dir: str = None, checkpoint: str = None, latest: bool = True, **kwargs):
    """Loads a network checkpoint file.

    Can be called in two different ways:
        load_checkpoint(network_dir):
            Loads the checkpoint file given by the path. If checkpoint_dir is a directory,
            it tries to find the latest checkpoint in that directory.

        load_checkpoint(network_dir, checkpoint=epoch_num):
            Loads the network at the given epoch number (int).


        load_checkpoint(path_to_checkpoint):
            Loads the file from the given absolute path (str).

    The extra keyword arguments are supplied to the network constructor to replace saved ones.
    """
    if network_dir is not None:
        net_path = Path(network_dir)
    else:
        net_path = None

    if net_path is not None and net_path.is_file():
        checkpoint = str(net_path)

    if checkpoint is None:
        if latest:
            # Load most recent checkpoint
            checkpoint_list = glob.glob('{}/iter-*.pth.tar'.format(net_path))
            checkpoint_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                raise Exception('No matching checkpoint file found')
        else:
            checkpoint_path = os.path.join(net_path, 'model_best.pth.tar')
            if not os.path.exists(checkpoint_path): raise ValueError(f'the path {checkpoint_path} doesnt exist')    
    elif isinstance(checkpoint, int):
        # Checkpoint is the epoch number
        checkpoint_list = sorted(net_path.glob('*iter-{:04d}.pth.tar'.format(checkpoint)))
        if not checkpoint_list or len(checkpoint_list) == 0:
            raise Exception('No matching checkpoint file found')
        if len(checkpoint_list) > 1:
            raise Exception('Multiple matching checkpoint files found')
        else:
            checkpoint_path = checkpoint_list[0]
    elif isinstance(checkpoint, str):
        # Checkpoint is the path
        checkpoint_path = os.path.expanduser(checkpoint)
    else:
        raise TypeError

    # Load network
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint_path, checkpoint_dict


def load_network(network_dir: str = None, checkpoint: str = None, 
                 constructor_fun_name: str = None, constructor_module: str = None, 
                 net_: str = None, latest: bool = True, **kwargs):
    """Loads a network checkpoint file.

    Can be called in two different ways:
        load_checkpoint(network_dir):
            Loads the checkpoint file given by the path. If checkpoint_dir is a directory,
            it tries to find the latest checkpoint in that directory.

        load_checkpoint(network_dir, checkpoint=epoch_num):
            Loads the network at the given epoch number (int).


        load_checkpoint(path_to_checkpoint):
            Loads the file from the given absolute path (str).

    The extra keyword arguments are supplied to the network constructor to replace saved ones.
    """
    checkpoint_path, checkpoint_dict = load_checkpoint(network_dir, checkpoint, latest, **kwargs)

    # Construct network model
    if 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
        net_constr = checkpoint_dict['constructor']

        if constructor_fun_name is not None:
            net_constr.fun_name = constructor_fun_name
        if constructor_module is not None:
            net_constr.fun_module = constructor_module
        net_fun = getattr(importlib.import_module(net_constr.fun_module), net_constr.fun_name)
        net_fun_args = list(inspect.signature(net_fun).parameters.keys())  # all_no_processing_of_pts parameters, including default

        # net_constr.args, empty
        # net_constr.kwds, dictionary containing what was actually given as input to the network constructor function
        for arg, val in kwargs.items():
            if arg == 'cfg':
                for arg_, val_ in val.items():
                    if isinstance(val_, dict):
                        for args_2, val_2 in val_.items():
                            if arg_ in net_constr.args[0].__dict__.keys():
                                net_constr.args[0].__dict__[arg_][args_2] = val_2
                    else:
                        net_constr.args[0].__dict__[arg_] = val_
            else:
                print('WARNING: Keyword argument "{}" not found when loading network. It was ignored.'.format(arg))
        net = net_constr.get()
    elif net_ is not None:
        net = net_
    else:
        raise RuntimeError('No constructor for the given network.')

    net.load_state_dict(checkpoint_dict['state_dict'], strict=True)

    if 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
        net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

    if 'epoch' in checkpoint_dict:
        print('Epoch is {}'.format(checkpoint_dict['epoch']))
        if hasattr(net, 'set_epoch'):
            net.set_epoch(checkpoint_dict['epoch'])

    return net, checkpoint_dict


def load_weights(net: torch.nn.Module, path: str, strict: bool = True):
    checkpoint_dict = torch.load(path)
    weight_dict = checkpoint_dict['net'] if 'net' in checkpoint_dict.keys() else checkpoint_dict['state_dict']
    net.load_state_dict(weight_dict, strict=strict)

    if 'epoch' in checkpoint_dict:
        print('Epoch is {}'.format(checkpoint_dict['epoch']))
        if hasattr(net, 'set_epoch'):
            net.set_epoch(checkpoint_dict['epoch'])


def partial_load(pretrained_dict: Dict[str, Any], model: torch.nn.Module, skip_keys: List[str] = []):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if
                     k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('\nSkipped keys: ', skipped_keys)
    print('\nLoading keys: ', filtered_dict.keys())
