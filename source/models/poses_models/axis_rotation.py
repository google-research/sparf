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
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import source.utils.camera as camera


class AxisRotationPoseParameters(nn.Module):
    def __init__(self, opt: Dict[str, Any], nbr_poses: int, initial_poses_w2c: torch.Tensor,  
                 device: torch.device):
        super().__init__()

        self.opt = opt
        self.nbr_poses = nbr_poses  # corresponds to total number of poses
        self.device = device
        self.initial_poses_w2c = initial_poses_w2c
        # corresponds to initialization of all the poses, including the ones that are fixed
        self.init_poses_embed()

    def init_poses_embed(self):
        if self.opt.camera.optimize_relative_poses:
            n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
            # the first n_first_fixed_poses are fixed, so they do not have any embedding
            # the others are to be optimized
            n_to_optimize = self.nbr_poses - n_first_fixed_poses
            self.pose_embedding = torch.nn.Parameter(torch.zeros(n_to_optimize, 6, device=self.device))
        else:
            # the trainable part of the poses
            self.pose_embedding = torch.nn.Parameter(torch.zeros(self.nbr_poses,6,device=self.device))

    def get_w2c_poses(self) -> torch.Tensor:
        # add learned pose correction to all training data initial poses
        pose_refine = camera.lie.se3_to_SE3(self.pose_embedding)  # when it is an embedding

        if self.opt.camera.optimize_relative_poses:
            # the first one is identity
            n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
            first_pose_fixed = self.initial_poses_w2c[:n_first_fixed_poses]

            pose_to_optimize = camera.pose.compose([pose_refine, self.initial_poses_w2c[n_first_fixed_poses:]])
            
            pose = torch.cat((first_pose_fixed, pose_to_optimize), dim=0)
            assert pose.shape[0] == self.nbr_poses
            # debug to check if relative poses would give correct results
            # pose_GT_ = torch.eye(4).to(self.device).unsqueeze(0).repeat(pose_GT.shape[0], 1, 1)
            # pose_GT_[:, :3] = pose_GT
            # relative_poses = pose_GT_[1:] @ pose_inverse_4x4(pose_GT_[0:1])
            # pose = torch.cat((first_pose_fixed, relative_poses[:, :3]), dim=0)
        else:
            pose = camera.pose.compose([pose_refine, self.initial_poses_w2c])
        return pose
    
    def get_c2w_poses(self) -> torch.Tensor:
        w2c = self.get_w2c_poses()
        return camera.pose.invert(w2c)
