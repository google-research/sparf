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

def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrix to 9D representation. 

    We take the two first ROWS of the rotation matrix, 
    along with the translation vector.
    ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
    """
    nbatch = pose.shape[0]
    R = pose[:, :3, :3]  # [N, 3, 3]
    t = pose[:, :3, -1]  # [N, 3]

    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    d9 = torch.cat((t, r6), -1)  # [N, 9]
    # first is the translation vector, then two first ROWS of rotation matrix

    return d9


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
            first two rows of the rotation matrix. 
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row


class FirstTwoColunmnsPoseParameters(nn.Module):
    def __init__(self, opt: Dict[str, Any], nbr_poses: int, initial_poses_w2c: torch.Tensor,  
                 device: torch.device):
        super().__init__()

        self.opt = opt
        self.optimize_c2w = self.opt.camera.optimize_c2w
        self.optimize_trans = self.opt.camera.optimize_trans
        self.optimize_rot = self.opt.camera.optimize_rot

        self.nbr_poses = nbr_poses  # corresponds to the total number of poses!
        self.device = device


        self.initial_poses_w2c = initial_poses_w2c  # corresponds to initialization of all the poses
        # including the ones that are fixed
        self.initial_poses_c2w = camera.pose.invert(self.initial_poses_w2c)

        # [N, 9]: (x, y, z, r1, r2) or [N, 3]: (x, y, z)
        self.init_poses_embed()
        # here pose_embedding is camera to world!!

    def init_poses_embed(self):
        if self.optimize_c2w:
            if self.opt.camera.optimize_relative_poses:
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
                poses_c2w_to_optimize = self.initial_poses_c2w[n_first_fixed_poses:]
                poses_c2w_to_optimize_embed = pose_to_d9(poses_c2w_to_optimize)
            else:
                poses_c2w_to_optimize = self.initial_poses_c2w
                poses_c2w_to_optimize_embed = pose_to_d9(poses_c2w_to_optimize)

            if self.optimize_rot and self.optimize_trans:
                # to stay consistent with my previosu trainings
                self.pose_embedding = nn.Parameter(poses_c2w_to_optimize_embed) 
            elif self.optimize_rot:
                self.trans_embedding = poses_c2w_to_optimize_embed[:, :3]
                self.rot_embedding = nn.Parameter(poses_c2w_to_optimize_embed[:, 3:])
            elif self.optimize_trans:
                self.trans_embedding = nn.Parameter(poses_c2w_to_optimize_embed[:, :3])
                self.rot_embedding = poses_c2w_to_optimize_embed[:, 3:]
            else:
                raise ValueError('Either the trans or the rot must be optimized')
        
        else:
            if self.opt.camera.optimize_relative_poses:
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
                # the first n_first_fixed_poses are fixed, so they do not have any embedding
                # the others are to be optimized
                poses_w2c_to_optimize = self.initial_poses_w2c[n_first_fixed_poses:]
                poses_w2c_to_optimize_embed = pose_to_d9(poses_w2c_to_optimize)

            else:
                poses_w2c_to_optimize = self.initial_poses_w2c
                poses_w2c_to_optimize_embed = pose_to_d9(poses_w2c_to_optimize) 

            if self.optimize_rot and self.optimize_trans:
                # to stay consistent with my previosu trainings
                self.pose_embedding = nn.Parameter(poses_w2c_to_optimize_embed) 
            elif self.optimize_rot:
                self.trans_embedding = poses_w2c_to_optimize_embed[:, :3]
                self.rot_embedding = nn.Parameter(poses_w2c_to_optimize_embed[:, 3:])
            elif self.optimize_trans:
                self.trans_embedding = nn.Parameter(poses_w2c_to_optimize_embed[:, :3])
                self.rot_embedding = poses_w2c_to_optimize_embed[:, 3:]
            else:
                raise ValueError('Either the trans or the rot must be optimized')

    def get_initial_w2c(self):
        return self.initial_poses_w2c

    def get_c2w_poses(self) -> torch.Tensor:
        if self.optimize_c2w:
            if self.opt.camera.optimize_relative_poses:
                poses_c2w_fixed = self.initial_poses_c2w[: self.opt.camera.n_first_fixed_poses]
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses
                if self.optimize_rot and self.optimize_trans:
                    t = self.pose_embedding[:, :3]  # [n_to_optimize, 3]
                    r = self.pose_embedding[:, 3:]  # [n_to_optimize, 6]
                else:
                    t = self.trans_embedding
                    r = self.rot_embedding
                R = r6d2mat(r)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_c2w_to_optimize = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
                poses_c2w = torch.cat((poses_c2w_fixed, poses_c2w_to_optimize), dim=0)
                assert poses_c2w.shape[0] == self.nbr_poses
            else:
                if self.optimize_rot and self.optimize_trans:
                    t = self.pose_embedding[:, :3]  # [n_to_optimize, 3]
                    r = self.pose_embedding[:, 3:]  # [n_to_optimize, 6]
                else:
                    t = self.trans_embedding
                    r = self.rot_embedding

                R = r6d2mat(r)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_c2w = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
        else:
            w2c = self.get_w2c_poses()
            poses_c2w = camera.pose.invert(w2c)
        return poses_c2w

    def get_w2c_poses(self) -> torch.Tensor:
        if not self.optimize_c2w:
            if self.opt.camera.optimize_relative_poses:
                poses_w2c_fixed = self.initial_poses_w2c[: self.opt.camera.n_first_fixed_poses]
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses
                if self.optimize_rot and self.optimize_trans:
                    t = self.pose_embedding[:, :3]  # [n_to_optimize, 3]
                    r = self.pose_embedding[:, 3:]  # [n_to_optimize, 6]
                else:
                    t = self.trans_embedding
                    r = self.rot_embedding
                R = r6d2mat(r)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_w2c_to_optimize = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
                poses_w2c = torch.cat((poses_w2c_fixed, poses_w2c_to_optimize), dim=0)
                assert poses_w2c.shape[0] == self.nbr_poses
            else:
                if self.optimize_rot and self.optimize_trans:
                    t = self.pose_embedding[:, :3]  # [n_to_optimize, 3]
                    r = self.pose_embedding[:, 3:]  # [n_to_optimize, 6]
                else:
                    t = self.trans_embedding
                    r = self.rot_embedding
                R = r6d2mat(r)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_w2c = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
        else:
            c2w = self.get_c2w_poses()
            poses_w2c = camera.pose.invert(c2w)
        return poses_w2c

