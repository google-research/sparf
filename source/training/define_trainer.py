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
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.training.nerf_trainer import NerfTrainerPerScene
from source.training.nerf_trainer_w_fixed_colmap_poses import NerfTrainerPerSceneWColmapFixedPoses
from source.training.joint_pose_nerf_trainer import PoseAndNerfTrainerPerScene

from source.utils.config_utils import save_options_file


def define_trainer(args: Dict[str, Any], settings_model: Dict[str, Any], 
                   debug: bool=False, save_option: bool=True):
    """Defines the trainer (NeRF with fixed ground-truth poses, NeRF with fixed
    colmap poses, joint pose-NeRF training)

    Args:
        args (edict): arguments from the command line. Importantly, contains
                      args.env
        settings_model (edict): config of the model
        debug (bool, optional): Defaults to False.
    """
    settings_model.update(args.args_to_update)

    if settings_model.model != 'joint_pose_nerf_training':
        # number of iterations when poses are fixed
        if ('dtu' in settings_model.dataset or 'replica' in settings_model.dataset):
            if settings_model.train_sub == 3:
                settings_model.max_iter = 50000
            elif settings_model.train_sub == 6:
                settings_model.max_iter = 100000
            elif settings_model.train_sub == 9:
                settings_model.max_iter = 150000
        elif 'llff' in settings_model.dataset:
            if settings_model.train_sub == 3:
                settings_model.max_iter = 70000
            elif settings_model.train_sub == 6:
                settings_model.max_iter = 140000
            elif settings_model.train_sub == 9:
                settings_model.max_iter = 200000
    elif settings_model.model == 'joint_pose_nerf_training':
        if ('dtu' in settings_model.dataset or 'replica' in settings_model.dataset):
            if settings_model.train_sub == 2:
                settings_model.max_iter = 60000
            elif settings_model.train_sub == 3:
                settings_model.max_iter = 100000
            elif settings_model.train_sub == 6:
                settings_model.max_iter = 150000
            else:
                settings_model.max_iter = 200000
        elif 'llff' in settings_model.dataset:  
            if settings_model.train_sub == 2:
                settings_model.max_iter = 60000           
            elif settings_model.train_sub == 3:
                settings_model.max_iter = 100000
            elif settings_model.train_sub == 6:
                settings_model.max_iter = 170000
            else:
                settings_model.max_iter = 220000

    if settings_model.dataset == 'dtu':
        settings_model.seed = int(settings_model.scene.split('scan')[-1])
        
    if debug:
        settings_model.vis_steps = 2    # visualize results (every N iterations)
        settings_model.log_steps = 2    # log losses and scalar states (every N iterations)
        settings_model.snapshot_steps = 5 
        settings_model.val_steps = 5 

    if save_option:
        save_options_file(settings_model, os.path.join(args.env.workspace_dir, 
                                                       args.project_path), override='y')
    
    args.debug = debug
    args.update(settings_model)

    if args.model == 'nerf_gt_poses':
        trainer = NerfTrainerPerScene(args)
    elif args.model == 'nerf_fixed_noisy_poses':
        trainer = NerfTrainerPerSceneWColmapFixedPoses(args)
    elif args.model == 'joint_pose_nerf_training':
        trainer = PoseAndNerfTrainerPerScene(args)
    else:
        raise ValueError
    return trainer
