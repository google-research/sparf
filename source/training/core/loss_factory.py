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
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.training.core.base_losses import (Loss, SparseCOLMAPDepthLoss, BasePhotoandReguLoss)
from source.training.core.corres_loss import CorrespondencesPairRenderDepthAndGet3DPtsAndReproject
from source.training.core.depth_cons_loss import DepthConsistencyLoss


def define_loss(loss_type: str, opt: Dict[str, Any], nerf_net: torch.nn.Module, 
                train_data: Dict[str, Any], device: torch.device, flow_net: torch.nn.Module=None):
    loss_module = []

    if 'photometric' in loss_type:
        loss_module.append(BasePhotoandReguLoss(opt, nerf_net, train_data=train_data, device=device)) 
    
    if 'SparseCOLMAPDepthLoss' in loss_type:
        loss_module.append(SparseCOLMAPDepthLoss(opt, nerf_net, device=device))
        
    if 'corres' in loss_type:
        loss_module.append(CorrespondencesPairRenderDepthAndGet3DPtsAndReproject\
            (opt, nerf_net, flow_net=flow_net, train_data=train_data, device=device))   

    if 'depth_cons' in loss_type:
        loss_module.append(DepthConsistencyLoss(opt, nerf_net, device=device))
    loss_module = Loss(loss_module)
    return loss_module