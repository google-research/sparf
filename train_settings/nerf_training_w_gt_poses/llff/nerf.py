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
    settings_model.barf_c2f = None

    settings_model.nerf = edict()                                                       # NeRF-specific options
    settings_model.nerf.depth = edict()                                                  # depth-related options
    settings_model.nerf.fine_sampling = True                                  # hierarchical sampling with another NeRF


    # dataset
    settings_model.dataset = 'llff'
    settings_model.resize = None
    settings_model.llff_img_factor = 8

    # loss type
    settings_model.loss_type = 'photometric'

    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0    
    return override_options(default_config, settings_model)

