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
import os

from source.utils.config_utils import override_options
from train_settings.default_config import get_nerf_default_config_llff


def get_config():
    default_config = get_nerf_default_config_llff()
    
    settings_model = edict()

    # dataset
    settings_model.dataset = 'llff'
    settings_model.resize = None
    settings_model.llff_img_factor = 8

    # scheduling with coarse to fine 
    settings_model.barf_c2f =  [0.1, 0.5]  # coarse-to-fine pe

    # flow stuff
    settings_model.use_flow = True
    settings_model.flow_backbone='PDCNet'

    # loss type
    settings_model.loss_type = 'photometric_and_corres_and_depth_cons' 
    settings_model.matching_pair_generation = 'all_to_all'

    settings_model.gradually_decrease_corres_weight = True 


    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0.    
    settings_model.loss_weight.corres = -3  # for 10^
    settings_model.loss_weight.depth_cons = -3
    return override_options(default_config, settings_model)

