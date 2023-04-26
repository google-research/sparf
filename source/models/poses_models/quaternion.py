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
from typing import Dict, Any, Optional
import source.utils.camera as camera
    
# ------- Important tips ---------
# If you represent a rotation as a quaternion, it is important to normalize it
# to unit norm after each gradient update, e.g.
#...
#loss.backward()
#optimizer.step()
#q.data = torch.nn.functional.normalize(q.data, dim=-1)


class QuaternionsPoseParameters(nn.Module):
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

        # quaternions will be [N, 4] with (w, x, y,z)
        self.init_poses_embed()
        # here pose_embedding is camera to world!!

    def init_poses_embed(self):
        if self.optimize_c2w:
            if self.opt.camera.optimize_relative_poses:
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
                # the first n_first_fixed_poses are fixed, so they do not have any embedding
                # the others are to be optimized
                poses_c2w_to_optimize = self.initial_poses_c2w[n_first_fixed_poses:]
                R_c2w_to_optimize = poses_c2w_to_optimize[:, :3, :3]
                t_c2w_to_optimize = poses_c2w_to_optimize[:, :3, -1]
                R_c2w_to_optimize_embed = camera.quaternion.R_to_q(R_c2w_to_optimize)  # (N, 4)
            else:
                poses_c2w_to_optimize = self.initial_poses_c2w
                R_c2w_to_optimize = poses_c2w_to_optimize[:, :3, :3]
                t_c2w_to_optimize = poses_c2w_to_optimize[:, :3, -1]
                R_c2w_to_optimize_embed = camera.quaternion.R_to_q(R_c2w_to_optimize)  # (N, 4)

            if self.optimize_rot:
                self.rot_embedding = nn.Parameter(R_c2w_to_optimize_embed) 
            else:
                self.rot_embedding = R_c2w_to_optimize_embed

            if self.optimize_trans:
                self.trans_embedding = nn.Parameter(t_c2w_to_optimize)
            else:
                self.trans_embedding = t_c2w_to_optimize
        else:
            if self.opt.camera.optimize_relative_poses:
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses 
                # the first n_first_fixed_poses are fixed, so they do not have any embedding
                # the others are to be optimized
                poses_w2c_to_optimize = self.initial_poses_w2c[n_first_fixed_poses:]
                R_w2c_to_optimize = poses_w2c_to_optimize[:, :3, :3]
                t_w2c_to_optimize = poses_w2c_to_optimize[:, :3, -1]
                R_w2c_to_optimize_embed = camera.quaternion.R_to_q(R_w2c_to_optimize)  # (N, 4)
            else:
                poses_w2c_to_optimize = self.initial_poses_w2c
                R_w2c_to_optimize = poses_w2c_to_optimize[:, :3, :3]
                t_w2c_to_optimize = poses_w2c_to_optimize[:, :3, -1]
                R_w2c_to_optimize_embed = camera.quaternion.R_to_q(R_w2c_to_optimize)  # (N, 4)

            if self.optimize_rot:
                self.rot_embedding = nn.Parameter(R_w2c_to_optimize_embed) 
            else:
                self.rot_embedding = R_w2c_to_optimize_embed

            if self.optimize_trans:
                self.trans_embedding = nn.Parameter(t_w2c_to_optimize)
            else:
                self.trans_embedding = t_w2c_to_optimize

    def get_c2w_poses(self) -> torch.Tensor:
        if self.optimize_c2w:
            if self.opt.camera.optimize_relative_poses:
                poses_c2w_fixed = self.initial_poses_c2w[: self.opt.camera.n_first_fixed_poses]
                n_first_fixed_poses = self.opt.camera.n_first_fixed_poses
                t = self.trans_embedding  # [n_to_optimize, 3]
                quat = self.rot_embedding  # [n_to_optimize, 4]
                quat = torch.nn.functional.normalize(quat, dim=-1)
                R = camera.quaternion.q_to_R(quat)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_c2w_to_optimize = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
                poses_c2w = torch.cat((poses_c2w_fixed, poses_c2w_to_optimize), dim=0)
                assert poses_c2w.shape[0] == self.nbr_poses
            else:
                t = self.trans_embedding  # [n_to_optimize, 3]
                quat = self.rot_embedding  # [n_to_optimize, 4]
                quat = torch.nn.functional.normalize(quat, dim=-1)
                R = camera.quaternion.q_to_R(quat)[:, :3, :3]  # [n_to_optimize, 3, 3]
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
                t = self.trans_embedding  # [n_to_optimize, 3]
                quat = self.rot_embedding  # [n_to_optimize, 4]
                quat = torch.nn.functional.normalize(quat, dim=-1)
                R = camera.quaternion.q_to_R(quat)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_w2c_to_optimize = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
                poses_w2c = torch.cat((poses_w2c_fixed, poses_w2c_to_optimize), dim=0)
                assert poses_w2c.shape[0] == self.nbr_poses
            else:
                t = self.trans_embedding  # [n_to_optimize, 3]
                quat = self.rot_embedding  # [n_to_optimize, 4]
                quat = torch.nn.functional.normalize(quat, dim=-1)
                R = camera.quaternion.q_to_R(quat)[:, :3, :3]  # [n_to_optimize, 3, 3]
                poses_w2c = torch.cat((R, t[..., None]), -1)  # [n_to_optimize, 3, 4]
        else:
            c2w = self.get_c2w_poses()
            poses_w2c = camera.pose.invert(c2w)
        return poses_w2c

