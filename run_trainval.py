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

import os
import argparse
import importlib
import cv2 as cv
import torch
import torch.backends.cudnn
from datetime import date
from easydict import EasyDict as edict
from typing import Any, Dict

import source.admin.settings as ws_settings
from source.training.define_trainer import define_trainer

def run_training(train_module: str, train_name: str, seed: int, cudnn_benchmark: bool=True, data_root: str='',
                 debug: int=False, args: Dict[str, Any]=None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
        data_root: in case you are in a server, and the data_root is not known beforehand, passed as
                   an argument.
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    settings = ws_settings.Settings(data_root, debug)
    settings.data_root = data_root

    train_module_for_launching = train_module

    # update with arguments
    # this is not very robust, assumes that it will be module/dataset, and want to add something here
    base_dir_train_module = train_module.split('/') 
    if args.train_sub is not None and args.train_sub != 0:
        base_dir_train_module[1] += '/subset_' + str(args.train_sub)
    else:
        args.train_sub = None
    
    if args.scene is not None:
        base_dir_train_module[1] += '/' + args.scene
    train_module = '/'.join(base_dir_train_module)

    
    settings.module_name_for_eval = train_module_for_launching
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = '{}/{}'.format(train_module, train_name)
    settings.seed = seed
    settings.use_wandb = False

    # update arguments
    args_to_update = {}
    for k, v in vars(args).items():
        if v is not None:
            args_to_update[k] = v
    for k, v in settings.__dict__.items():
        if k == 'env':
            continue
        if v is not None:
            args_to_update[k] = v
    settings.args_to_update = args_to_update


    # dd/mm/YY
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    print('Training:  {}  {}\nDate: {}'.format(train_module, train_name, d1))

    # will save the checkpoints there
    if not os.path.exists(os.path.join(settings.env.workspace_dir, settings.project_path)):
        os.makedirs(os.path.join(settings.env.workspace_dir, settings.project_path))
 
    
    # get the config file
    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module_for_launching.replace('/', '.'),
                                                                        train_name.replace('/', '.')))
    expr_func = getattr(expr_module, 'get_config')

    # does not support multi-gpu for now
    settings.distributed = False
    settings.local_rank = 0
    settings = edict(settings.__dict__)
    
    # get the config and define the trainer
    model_config = expr_func()
    trainer = define_trainer(args=settings, settings_model=model_config, 
                             debug=args.debug)

    # actual run
    if trainer.debug:
        trainer.run_debug(load_latest=True, make_validation_first=False)
    else:
        trainer.run(load_latest=True)
        

def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--data_root', type=str, default='/home/jupyter/shared',
                        help='Name of the train settings file.')

    # arguments
    parser.add_argument('--scene', type=str, default=None,
                        help='scene')
    parser.add_argument('--train_sub', type=int, default=None,
                        help='train subset: how many input views to consider?')

    # for eval
    parser.add_argument('--render_video_only', type=bool, default=False,
                        help='render_video_only?')
    parser.add_argument('--render_video_pose_only', type=bool, default=False,
                        help='render_video_pose_only?')
    parser.add_argument('--test_metrics_only', type=bool, default=False,
                        help='test_metrics_only?')
    parser.add_argument('--plot', type=bool, default=False,
                        help='plot?')
    parser.add_argument('--save_ind_files', type=bool, default=False,
                        help='Save the rgb and depth renderings, gt images and depth as separate png?')
    # debugging
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug?')
    
    # others
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--seed', type=int, default=0, help='Pseudo-RNG seed')
    

    args = parser.parse_args()

    run_training(args.train_module, args.train_name, cudnn_benchmark=args.cudnn_benchmark, seed=args.seed,
                 data_root=args.data_root, debug=args.debug, args=args)


if __name__ == '__main__':
    main()