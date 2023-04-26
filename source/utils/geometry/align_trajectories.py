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

import numpy as np
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as RotLib
from typing import Union, Any, List, Dict, Tuple

from third_party.ATE.align_utils import alignTrajectory
from source.utils.camera import pose_inverse_4x4
import source.utils.camera as camera




def SO3_to_quat(R: np.ndarray):
    """
    Args:
        R:  (N, 3, 3) or (3, 3) np
    Returns:   
        (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat: np.ndarray):
    """
    Args:
        quat:    (N, 4, ) or (4, ) np
    Returns:  
        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input: torch.Tensor) -> torch.Tensor:
    """
    Args:
        input:  (N, 3, 4) or (3, 4) torch or np
    Returns: 
        (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def backtrack_from_aligning_and_scaling_to_first_cam(pose_GT_w2c: torch.Tensor, 
                                                     ssim_est_gt_c2w: Dict[str, Any]):
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    pose_GT_c2w[:, :3, -1] -= ssim_est_gt_c2w.final_t.reshape(-1, 3)  # (-1, 3)
    pose_GT_c2w[:, :3, -1] /= ssim_est_gt_c2w.trans_scaling_after

    R_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) @ pose_GT_c2w[:, :3, :3]  # (N, 3, 3)
    t_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) / ssim_est_gt_c2w.s @ (pose_GT_c2w[:, :3, 3:4] - ssim_est_gt_c2w.t)  # (N, 3, 1)

    pose_GT_c2w_aligned = camera.pose(R=R_GT_c2w_aligned,t=t_GT_c2w_aligned.reshape(-1, 3))

    # pose_GT_c2w_aligned /= ssim_est_gt_c2w.trans_scaling_before

    pose_w2d_recovered = camera.pose.invert(pose_GT_c2w_aligned)
    return pose_w2d_recovered


def backtrack_from_aligning_the_trajectory(pose_GT_w2c: torch.Tensor, 
                                           ssim_est_gt_c2w: Dict[str, Any]):
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
    R_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) @ pose_GT_c2w[:, :3, :3]
    t_GT_c2w_aligned = ssim_est_gt_c2w.R.transpose(-2, -1) / ssim_est_gt_c2w.s @ (pose_GT_c2w[:, :3, 3:4] - ssim_est_gt_c2w.t)
    pose_GT_c2w_aligned = camera.pose(R=R_GT_c2w_aligned,t=t_GT_c2w_aligned.reshape(-1, 3))
    pose_w2d_recovered = camera.pose.invert(pose_GT_c2w_aligned)
    return pose_w2d_recovered

    
def align_translations(pose_GT_w2c: torch.Tensor, initial_poses_w2c: torch.Tensor):
    pose_GT_w2c_ = torch.eye(4).unsqueeze(0).repeat(pose_GT_w2c.shape[0], 1, 1).to(pose_GT_w2c.device)
    pose_GT_w2c_[:, :3] = pose_GT_w2c
    pose_GT_c2w = pose_inverse_4x4(pose_GT_w2c_)

    initial_poses_c2w = pose_inverse_4x4(initial_poses_w2c)  # (B, 4, 4)

    # initial_poses_c2w = pose_inverse_4x4(initial_poses_w2c)
    # position of camera in world coordinate basically
    trans_error = pose_GT_c2w[:, :3, -1].mean(0).reshape(1, -1) - \
        initial_poses_c2w[:, :3, -1].mean(0).reshape(1, -1)
    
    initial_poses_c2w[:, :3, -1] += trans_error

    translation_scaling = 1.
    initial_poses_w2c = pose_inverse_4x4(initial_poses_c2w)
    return initial_poses_w2c, translation_scaling


def align_to_first_camera(pose_GT_w2c: torch.Tensor, initial_poses_w2c: torch.Tensor):
    pose_GT_w2c_ = torch.eye(4).unsqueeze(0).repeat(pose_GT_w2c.shape[0], 1, 1).to(pose_GT_w2c.device)
    pose_GT_w2c_[:, :3] = pose_GT_w2c

    # the new world becomes camera 0 
    # the others are all expressed 
    # initial_poses_w2c[0] is not basically identity 
    initial_poses_w2c[1:] = initial_poses_w2c[1:] @ pose_inverse_4x4(initial_poses_w2c[0]).unsqueeze(0)

    # now we want the world to correspond to the gt one 
    initial_poses_w2c[0] = pose_GT_w2c_[0]  # basically identity @ initial_pose_w2c[0]
    initial_poses_w2c[1:] = initial_poses_w2c[1:] @ initial_poses_w2c[0].unsqueeze(0)

    # we invert the poses to apply scaling and translation
    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)  # (B, 3, 4)
    initial_poses_c2w = pose_inverse_4x4(initial_poses_w2c)  # (B, 4, 4)

    # compute translation scaling 
    initial_poses_relative = initial_poses_w2c[0] @ pose_inverse_4x4(initial_poses_w2c[1])
    # if one of the translation is zero, then the translation factor will become infinity
    # example when all the poses are initialized to identity
    if (initial_poses_relative[:3, -1].eq(0.)).sum() > 0:
        translation_scaling = 1.
    else:
        GT_relative = pose_GT_w2c_[0] @ pose_inverse_4x4(pose_GT_w2c_[1])
        translation_scaling = torch.abs(GT_relative[:3, -1] / initial_poses_relative[:3, -1]).mean()  # so they roughtly correspond
    initial_poses_c2w[:, :3, -1] *= translation_scaling
    
    # position of camera in world coordinate basically
    trans_error = pose_GT_c2w[:, :3, -1].mean(0).reshape(1, -1) - \
        initial_poses_c2w[:, :3, -1].mean(0).reshape(1, -1)
    
    initial_poses_c2w[:, :3, -1] += trans_error

    initial_poses_w2c = pose_inverse_4x4(initial_poses_c2w)
    return initial_poses_w2c, translation_scaling


def recenter_and_rescale_poses(pose_GT_w2c: torch.Tensor, initial_poses_w2c: torch.Tensor):
    pose_GT_w2c_ = torch.eye(4).unsqueeze(0).repeat(pose_GT_w2c.shape[0], 1, 1).to(pose_GT_w2c.device)
    pose_GT_w2c_[:, :3] = pose_GT_w2c

    pose_GT_c2w = camera.pose.invert(pose_GT_w2c)  # (B, 3, 4)
    initial_poses_c2w = pose_inverse_4x4(initial_poses_w2c)  # (B, 4, 4)

    # compute translation scaling 
    initial_poses_relative = initial_poses_w2c[0] @ pose_inverse_4x4(initial_poses_w2c[1])
    # if one of the translation is zero, then the translation factor will become infinity
    # example when all the poses are initialized to identity

    if (initial_poses_relative[:3, -1].eq(0.)).sum() > 0:
        translation_scaling = 1.
    else:
        GT_relative = pose_GT_w2c_[0] @ pose_inverse_4x4(pose_GT_w2c_[1])
        translation_scaling = torch.abs(GT_relative[:3, -1] / initial_poses_relative[:3, -1]).mean()  # so they roughtly correspond
    initial_poses_c2w[:, :3, -1] *= translation_scaling

    # recenters at the origin
    avg_trans = initial_poses_c2w[:, :3, -1].mean(0)
    initial_poses_c2w[:, :3, -1] -= avg_trans.reshape(1, -1)

    initial_poses_w2c = pose_inverse_4x4(initial_poses_c2w)

    # test
    '''
    initial_poses_relative = initial_poses_w2c[0] @ pose_inverse_4x4(initial_poses_w2c[1])
    GT_relative = pose_GT_w2c_[0] @ pose_inverse_4x4(pose_GT_w2c_[1])
    print(initial_poses_relative[:3, -1], GT_relative[:3, -1])
    '''
    return initial_poses_w2c, translation_scaling


def pts_dist_max(pts: Union[torch.Tensor, np.ndarray]):
    """
    Args: 
        pts: (N, 3) torch or np
    Returns: scalar
    """
    if torch.is_tensor(pts):
        dist = pts.unsqueeze(0) - pts.unsqueeze(1)  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = dist.norm(dim=1)  # (N, )
        max_dist = dist.max()
    else:
        dist = pts[None, :, :] - pts[:, None, :]  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = np.linalg.norm(dist, axis=1)  # (N, )
        max_dist = dist.max()
    return max_dist


def align_ate_c2b_use_a2b(traj_a_c2w: torch.Tensor, traj_b_c2w: torch.Tensor, 
                          traj_c: torch.Tensor=None, method='sim3', 
                          pose_id_to_align=0) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Align c to b using the sim3 from a to b.
    Args:
        traj_a:  (N0, 3/4, 4) torch tensor
        traj_b:  (N0, 3/4, 4) torch tensor
        traj_c:  None or (N1, 3/4, 4) torch tensor
    Returns:
        (N1, 4, 4) torch tensor
    """
    device = traj_a_c2w.device
    if traj_c is None:
        traj_c = traj_a_c2w.clone()

    traj_a = traj_a_c2w.float().cpu().numpy()
    traj_b = traj_b_c2w.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method, pose_id_to_align=pose_id_to_align)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)


    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    ssim_est_gt_c2w = edict(R=torch.from_numpy(R).to(device), t=torch.from_numpy(t).to(device), s=s)
    return traj_c_aligned, ssim_est_gt_c2w  # (N1, 4, 4)


def align_scale_c2b_use_a2b(traj_a: torch.Tensor, traj_b: torch.Tensor, 
                            traj_c: torch.Tensor=None):
    '''Scale c to b using the scale from a to b.
    Args:
        traj_a:      (N0, 3/4, 4) torch tensor
        traj_b:      (N0, 3/4, 4) torch tensor
        traj_c:      None or (N1, 3/4, 4) torch tensor
    Returns:
        scaled_traj_c   (N1, 4, 4)   torch tensor
        scale           scalar
    '''
    if traj_c is None:
        traj_c = traj_a.clone()

    t_a = traj_a[:, :3, 3]  # (N, 3)
    t_b = traj_b[:, :3, 3]  # (N, 3)

    # scale estimated poses to colmap scale
    # s_a2b: a*s ~ b
    scale_a2b = pts_dist_max(t_b) / pts_dist_max(t_a)

    traj_c[:, :3, 3] *= scale_a2b

    if traj_c.shape[1] == 3:
        traj_c = convert3x4_4x4(traj_c)  # (N, 4, 4)

    return traj_c, scale_a2b  # (N, 4, 4)