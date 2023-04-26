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
 
import time
from pathlib import Path
from easydict import EasyDict as edict

from source.utils.config_utils import  override_options
from train_settings.default_config import get_joint_pose_nerf_default_config_360_data


def get_config():
    default_config = get_joint_pose_nerf_default_config_360_data()
    
    settings_model = edict()

    # camera options    
    settings_model.camera = edict()
    settings_model.camera.initial_pose = 'noisy_gt'                                           
    settings_model.camera.noise = 0.15

    settings_model.barf_c2f =  [0.4, 0.7]  # coarse-to-fine pe

    # dataset
    settings_model.dataset = 'dtu'
    settings_model.resize = None

    settings_model.nerf = edict()                                                       # NeRF-specific options
    settings_model.nerf.depth = edict()                                                  # depth-related options
    settings_model.nerf.depth.param = 'metric'                                       # depth parametrization (for sampling along the ray)
    settings_model.nerf.fine_sampling = True                               # hierarchical sampling with another NeRF

    # loss type
    settings_model.loss_type = 'photometric'

    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0 
    return override_options(default_config, settings_model)

