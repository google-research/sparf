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

import sys
import os.path as osp
from datetime import date
import abc
from collections import OrderedDict
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from typing import Any, Optional, Dict, Tuple

from source.utils.summary_board import SummaryBoard
from source.utils.timer import Timer
from source.utils.torch import all_reduce_tensors, release_cuda, initialize, to_cuda
from source.training.engine.logger import Logger


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


default_settings = edict()
default_settings.resume_snapshot = None
default_settings.seed = 0
default_settings.keep_last_checkpoints = 2
default_settings.local_rank = 0
default_settings.log_steps = 100  # log every x steps
default_settings.mean_nbr_steps = default_settings.log_steps


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        settings: Dict[str, Any],
        cudnn_deterministic: bool=True,
        autograd_anomaly_detection: bool=False,
        run_grad_check: bool=False,
        grad_acc_steps: int=1,
    ):
        # parser
        self.settings = settings
        self.debug = settings.debug   # to be able to easily debug
        self._set_default_settings()
        self._base_save_dir = self.settings.env.workspace_dir
        if not os.path.exists(self._base_save_dir):
            os.makedirs(self._base_save_dir, exist_ok=True)
        # the model weights will be saved at self._base_save_dir, self.settings.project_path

        # logger
        log_dir = osp.join(settings.env.log_dir, self.settings.module_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = osp.join(log_dir, 'train-{}.log'.format(settings.script_name))
        self.logger = Logger(log_file=log_file, local_rank=self.settings.local_rank)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        self.logger.info('Training:  {}  {}\nDate: {}'.format(settings.module_name, settings.script_name, d1))

        # tensorboard
        self.tensorboard_dir = os.path.join(settings.env.tensorboard_dir, settings.project_path)
        if not os.path.isdir(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.logger.info(f'Tensorboard is enabled. Write events to {self.tensorboard_dir}.')

        # cuda and distributed
        self.local_rank = self.settings.local_rank
        self.distributed = False  # not supported otherwise
        if torch.cuda.device_count() > 1:
            self.logger.warning('DataParallel is deprecated. Use DistributedDataParallel instead.')
        self.world_size = 1
        self.local_rank = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info('Using Single-GPU mode.')
        self.logger.info(f'Using device {self.device}')
        
        self.cudnn_deterministic = cudnn_deterministic
        self.autograd_anomaly_detection = autograd_anomaly_detection
        self.seed = settings.seed + self.local_rank
        initialize(
            seed=self.seed,
            cudnn_deterministic=self.cudnn_deterministic,
            autograd_anomaly_detection=self.autograd_anomaly_detection,
        )
        self.logger.info(f'Setting seed {self.seed}')

        # basic config
        self.log_steps = self.settings.log_steps
        self.mean_nbr_steps = self.settings.log_steps  
        self.run_grad_check = run_grad_check
        self.just_started = False  # for running validation first

        # state
        self.net = None
        self.optimizer = None
        self.optimizer_pose = None
        self.scheduler = None
        self.scheduler_pose = None
        self.scheduler_per_iteration = False  # per_epoch
        self.epoch = 0
        self.iteration = 0
        self.iteration_nerf = 0

        self.inner_iteration = 0
        self.best_val = float("Inf")  # absolute best val, to know when to save the checkpoint.
        self.epoch_of_best_val = 0
        self.current_best_val = 0  # will be updated at each epoch, if we do validation

        self.train_loader = None
        self.val_loader = None
        self.summary_board = SummaryBoard(last_n=self.mean_nbr_steps, adaptive=True)
        self.timer = Timer()
        self.saved_states = {}

        # training config
        self.training = True
        self.grad_acc_steps = grad_acc_steps

    def _set_default_settings(self):
        # Dict of all_no_processing_of_pts default values
        for param, default_value in default_settings.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
        return

    def to_cuda(self, x):
        return to_cuda(x, self.device)

    def delete_old_checkpoints(self):
        """Delete all_no_processing_of_pts but the num_keep last saved checkpoints."""
        if self.local_rank != 0:
            return
        ckpts = sorted(glob.glob('{}/{}/iter-*.pth.tar'.format(self._base_save_dir, self.settings.project_path)))
        ckpts.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        ckpts_to_remove = ckpts[:-self.settings.keep_last_checkpoints]
        if len(ckpts_to_remove) > 0:
            for ckpt in ckpts_to_remove:
                os.remove(ckpt)
        return

    def save_snapshot(self, filename: str, training_info: bool=True):
        """Saves a checkpoint of the network and other variables."""
        if self.local_rank != 0:
            return

        if hasattr(self, 'return_model_dict'):
            model_state_dict = self.return_model_dict()
        else:
            if hasattr(self.net, 'return_model_dict'):
                model_state_dict = self.net.return_model_dict()
            else:
                model_state_dict = self.net.state_dict()

        # save model
        directory = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        filename = osp.join(directory, filename)
        state_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'iteration_nerf': self.iteration_nerf, 
            'state_dict':  model_state_dict,
            'best_val': self.best_val,
            'epoch_of_best_val': self.epoch_of_best_val
        }

        if training_info:
            # for training, add other info
            state_dict['optimizer'] = self.optimizer.state_dict()
            if hasattr(self, 'optimizer_pose') and self.optimizer_pose is not None:
                state_dict['optimizer_pose'] = self.optimizer_pose.state_dict()
            if self.scheduler is not None:
                state_dict['scheduler'] = self.scheduler.state_dict()
            if hasattr(self, 'scheduler_pose') and self.scheduler_pose is not None:
                state_dict['scheduler_pose'] = self.scheduler_pose.state_dict()

        # save snapshot
        torch.save(state_dict, filename)
        self.logger.info('Snapshot saved to "{}"'.format(filename))
        return 

    def load_snapshot(self, checkpoint: str=None, fields=None, ignore_fields=None, additional_ignore_fields=None,
                      load_constructor=False, fix_prefix=True):
        """Loads a network checkpoint file.
        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.net
        # get the checkpoint path to load
        if checkpoint is None:
            # Load most recent checkpoint
            # if this is iter this is not correct
            checkpoint_list = glob.glob('{}/{}/iter-*.pth.tar'.format(self._base_save_dir,
                                                                       self.settings.project_path))
            checkpoint_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/iter-{:04d}.pth.tar'.format(self._base_save_dir, self.settings.project_path,
                                                                  checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        self.logger.info('Loading from "{}".'.format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Load model dict
        model_dict = checkpoint_dict['state_dict']

        if fix_prefix and self.distributed:
            model_dict = OrderedDict([('module.' + key, value) for key, value in model_dict.items()])
        if hasattr(self, 'load_state_dict'):
            self.load_state_dict(model_dict)
        else:
            net.load_state_dict(model_dict, strict=True)
        
        '''
        # log missing keys and unexpected keys in the model dict
        snapshot_keys = set(model_dict.keys())
        model_keys = set(net.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if self.distributed:
            missing_keys = set([missing_key[7:] for missing_key in missing_keys])
            unexpected_keys = set([unexpected_key[7:] for unexpected_key in unexpected_keys])
        if len(missing_keys) > 0:
            message = f'Missing keys: {missing_keys}'
            self.logger.warning(message)
        if len(unexpected_keys) > 0:
            message = f'Unexpected keys: {unexpected_keys}'
            self.logger.warning(message)
        self.logger.info('Model has been loaded.')
        '''

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

        # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['scheduler', 'scheduler_pose', 'constructor', 'net_type', 'actor_type', 'net_info'])
        if additional_ignore_fields is not None:
            ignore_fields.extend(additional_ignore_fields)

        # Load other attributes
        for key in fields:
            if key in ignore_fields:
                continue
            elif key == 'epoch':
                self.epoch = checkpoint_dict['epoch']
                self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
            elif key == 'iteration':
                self.iteration = checkpoint_dict['iteration']
                self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
            elif key == 'optimizer':
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                    self.logger.info('Optimizer has been loaded.')
            elif key == 'optimizer_pose':
                if self.optimizer_pose is not None:
                    self.optimizer_pose.load_state_dict(checkpoint_dict['optimizer_pose'])
                    self.logger.info('Optimizer has been loaded.')
                    '''
                    # to be more flexible, we do not load it 
                    elif key == 'scheduler' and self.scheduler is not None:
                        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
                        self.logger.info('Scheduler has been loaded.')
                    '''
            else:
                setattr(self, key, checkpoint_dict[key])
        
        if 'iteration_nerf' in fields:
            self.iteration_nerf = checkpoint_dict['iteration_nerf']
        else:
            self.iteration_nerf = self.iteration

        # TODO: adapt learning rate when distributed training
        # Update the epoch in lr scheduler
        if self.scheduler is not None:
            if self.scheduler_per_iteration and 'iteration' in fields:
                last_epoch = self.iteration_nerf
                # here we do not save and update the scheduler to make it easier to change the lr
                self.scheduler.last_epoch = last_epoch
                current_lr = self.scheduler.get_last_lr()#_get_closed_form_lr()  # as it should be according to schedule
                for ind, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = current_lr[ind]
            elif not self.scheduler_per_iteration and 'epoch' in fields:
                last_epoch = self.epoch
                # here we do not save and update the scheduler to make it easier to change the lr
                self.scheduler.last_epoch = last_epoch
                current_lr = self.scheduler._get_closed_form_lr()  # as it should be according to schedule
                for ind, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = current_lr[ind]
        
        if hasattr(self, 'scheduler_pose') and self.scheduler_pose is not None and 'iteration' in fields:
            last_epoch = self.iteration
            # here we do not save and update the scheduler to make it easier to change the lr
            self.scheduler_pose.last_epoch = last_epoch
            current_lr = self.scheduler_pose._get_closed_form_lr()  # as it should be according to schedule
            for ind, param_group in enumerate(self.optimizer_pose.param_groups):
                param_group['lr'] = current_lr[ind]
        return 

    def register_model(self, net: torch.nn.Module):
        r"""Register model. DDP is automatically used."""
        if self.distributed:
            local_rank = self.local_rank
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        self.net = net
        return net

    def register_optimizer(self, optimizer: torch.optim.Optimizer):
        r"""Register optimizer. DDP is automatically used."""
        if self.distributed:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.world_size
        self.optimizer = optimizer
        return

    def register_scheduler(self, scheduler: Any):
        r"""Register LR scheduler."""
        self.scheduler = scheduler
        return 

    def register_loader(self, train_loader: torch.utils.data.DataLoader, 
                        val_loader: torch.utils.data.DataLoader): 
        r"""Register data loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        return 

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def optimizer_step(self, iteration: int, do_backprop: bool=True):
        if iteration % self.grad_acc_steps == 0:
            # here could accumulate gradients for longer, this is why the zero_grad is put here
            if do_backprop:
                # could be skipped in case of large gradients or something, but we still want to put the optimizer
                # gradient to zero, so it is not further accumulated
                self.optimizer.step()
            self.optimizer.zero_grad()
        return 

    def save_state(self, key: str, value: Any):
        self.saved_states[key] = release_cuda(value)

    def read_state(self, key):
        return self.saved_states[key]

    def check_invalid_gradients(self, net: torch.nn.Module):

        if isinstance(net, (list, tuple)):
            parameters = []
            parameters += [net[i].parameters() for i in range(len(net))]
        else:
            parameters = net.parameters()

        for param in parameters:
            if getattr(param, 'grad', None) is not None and torch.isnan(param.grad).any():
                self.logger.error('Epoch: {}, iter: {}: NaN in gradients.'.format(self.epoch, self.inner_iteration))
                return False
            if getattr(param, 'grad', None) is not None and torch.isinf(param.grad).any():
                self.logger.error('Epoch: {}, iter: {}: Inf in gradients.'.format(self.epoch, self.inner_iteration))
                return False
        return True

    def release_tensors(self, result_dict: Dict[str, Any]):
        r"""All reduce and release tensors."""
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = release_cuda(result_dict)
        return result_dict

    def set_train_mode(self):
        self.training = True
        self.net.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.net.eval()
        torch.set_grad_enabled(False)

    def write_event(self, phase: str, event_dict: Dict[str, Any], index: int):
        r"""Write TensorBoard event."""
        if self.local_rank != 0:
            return
        for key, value in event_dict.items():
            self.writer.add_scalar(f'{phase}/{key}', value, index)
        return 
        
    def write_image(self, phase: str, plotting_dict: Dict[str, Any], index: int):
        r"""Write TensorBoard event."""
        if self.local_rank != 0:
            return
        for key, value in plotting_dict.items():
            if len(value.shape) == 4:
                self.writer.add_images(f'{phase}/{key}', value, index , dataformats='NCHW')
            else:
                self.writer.add_image(f'{phase}/{key}', value, index) 
        return
        
    @abc.abstractmethod
    def run(self):
        raise NotImplemented
