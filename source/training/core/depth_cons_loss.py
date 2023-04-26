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
import os,sys,time
import torch
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.datasets.data_utils import get_nearest_pose_ids
from source.training.core.sampling_strategies import sample_rays
from source.utils.camera import pose_inverse_4x4
from source.utils.config_utils import override_options
from source.utils.geometry.batched_geometry_utils import (batch_backproject_to_3d, batch_project)
from source.training.core.base_losses import BaseLoss


class DepthConsistencyLoss(BaseLoss):
    """This is the loss module for the depth-consistency loss. The main idea is to use the
    depth maps rendered from the training viewpoints to create pseudo-depth supervision 
    for novel, unseen, viewpoints. Check Sec. 4.2 of SPARF for more details. """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, device: torch.device):
        super().__init__(device)
        default_cfg = edict({'gradually_decrease_geo_sampling_loss': False, 
                             'geo_sampling_loss_reduct_at_x_iter': 10000, 
                             'diff_loss_type': 'huber'})
        self.opt = override_options(default_cfg, opt)
        self.net = nerf_net
        self.device = device

    def sample_pose(self, poses_c2w: torch.Tensor, id_self: int, pose_w2c_self: torch.Tensor):
        """Sample a new pose, as an interpolation between two training poses. 

        Args:
            poses_c2w (torch.Tensor): c2w poses, shape (B, 4, 4)
            id_self (int): index corresponding to pose_w2c_self
            pose_w2c_self (torch.Tensor): sampled pose within the training poses. 
        """
        # sample a pose between the two others
        poses_c2w_numpy = poses_c2w.detach().cpu().numpy() 
        id_other = get_nearest_pose_ids(tar_pose_c2w=poses_c2w_numpy[id_self], ref_poses_c2w=poses_c2w_numpy, 
                                        num_select=1, tar_id=id_self, angular_dist_method='vector')[0]
        w = np.random.rand()
        pose_c2w_self = pose_inverse_4x4(pose_w2c_self).detach()
        pose_c2w_other = poses_c2w[id_other].detach()
        
        pose_c2w_at_unseen =  w * pose_c2w_self + (1 - w) * pose_c2w_other
        pose_w2c_at_unseen = pose_inverse_4x4(pose_c2w_at_unseen)
        return pose_w2c_at_unseen

    def compute_loss_from_existing_pixels(self, opt: Dict[str, Any], data_dict: Dict[str, Any], iteration: int, 
                                          pixels_in_ref: torch.Tensor, id_ref: int, depth_ref: torch.Tensor=None
                                          ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Computes the depth consistency loss. Instead of sampling pixels into a reference 
        training images, we use already computed pseudo-gt 3D point and the corresponding pixels. 
        This can correspond to pixels and rendering computed for the correspondence loss. 

        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            iteration (int): 
            pixels_in_ref (torch.Tensor): Pixels in the reference training image, corresponding to 
                                          the pseudo gt. Shape is (N, 2)
            id_self (int): index of this reference training images
            depth_ref (torch.Tensor, optional): Depth rendered from the renference image for teh pixels_in_ref. 
                                          Shape is (N, 2). Defaults to None.
        """

        loss_dict, stats_dict, plotting_dict = {}, {}, {}
        
        start_iter_depth_cons = opt.start_ratio.depth_cons * opt.max_iter \
            if opt.start_ratio.depth_cons is not None else opt.start_iter.depth_cons
        if iteration < start_iter_depth_cons:
            return loss_dict, stats_dict, plotting_dict

        images = data_dict.image  # (B, 3, H, W)
        B, _, H, W = images.shape
        
        
        # get current pose predictions and detach
        bottom = torch.from_numpy(np.array([0, 0, 0, 1])).to(self.device).reshape(1, 1, -1).repeat(B, 1, 1)
        poses_w2c = data_dict.poses_w2c.detach()
        poses_w2c = torch.cat((poses_w2c, bottom), axis=1)
        poses_c2w = pose_inverse_4x4(poses_w2c)
        
        intr = data_dict.intr 
        
        intr_ref = intr[id_ref]
        pose_w2c_ref = poses_w2c[id_ref]
        pose_c2w_ref = poses_c2w[id_ref]

        # project ref points to 3d ==> to create pseudo gt
        pts3d_in_w_from_ref = batch_backproject_to_3d(kpi=pixels_in_ref, di=depth_ref, Ki=intr_ref, 
                                                      T_itoj=pose_c2w_ref)

        # sample unobserved pose
        pose_w2c_at_unseen = self.sample_pose(poses_c2w, id_ref, pose_w2c_ref)
        intr_at_sampled = intr_ref.clone()

        return self.compute_loss_at_sampled_pose(data_dict, pose_w2c_at_unseen=pose_w2c_at_unseen, 
                                                 intr_at_unseen=intr_at_sampled, 
                                                 pseudo_gt_3dpts_in_w=pts3d_in_w_from_ref, H=H, W=W, 
                                                 pixels_in_ref=pixels_in_ref)


    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], 
                     output_dict: Dict[str, Any], iteration: int, 
                     mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Computes the depth consistency loss. Here, we sampled  a new set of pixels for which the 
        pseudo-gt depth is created. 

        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Will not be used here, because rendering must be where 
                                 a match is available. 
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """
        

        if mode != 'train':
            # only during training
            return {}, {}, {}

        loss_dict, stats_dict, plotting_dict = {}, {}, {}

        # condition if using this loss
        start_iter_depth_cons = opt.start_ratio.depth_cons * opt.max_iter \
            if opt.start_ratio.depth_cons is not None else opt.start_iter.depth_cons
        if iteration < start_iter_depth_cons:
            return loss_dict, stats_dict, plotting_dict
        
        images = data_dict.image  # (B, 3, H, W)
        B, _, H, W = images.shape
        bottom = torch.from_numpy(np.array([0, 0, 0, 1])).to(self.device).reshape(1, 1, -1).repeat(B, 1, 1)
        
        # get current estimates of the pose and detach them ==> not backpropagated through
        poses_w2c = data_dict.poses_w2c 
        poses_w2c = poses_w2c.detach()
        poses_w2c = torch.cat((poses_w2c, bottom), axis=1)
        poses_c2w = pose_inverse_4x4(poses_w2c)
        
        intr = data_dict.intr 


        # select one image as id_self and sample pixels within this image ==> will be used to project to unseen view
        # and create pseudo gt. 
        id_self = np.random.randint(B)
        pixels_ref, _ = sample_rays(H, W, nbr=max(1024, self.opt.nerf.rand_rays), 
                                    fraction_in_center=self.opt.sampled_fraction_in_center)

        pixels_ref = pixels_ref.reshape(-1, 2).to(self.device)
        intr_ref = intr[id_self]
        pose_w2c_ref = poses_w2c[id_self]
        pose_c2w_ref = poses_c2w[id_self]

        # create pseudo-gt 3D points. 
        # render points in ref image
        ret_ref = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose=pose_w2c_ref[:3], 
                                                                  intr=intr_ref, H=H, W=W, 
                                                                  pixels=pixels_ref, mode='train', iter=iteration)
        
        # we use the fine depth as the supervision
        # the only issue is if using it just after activating start fine ==> then it hasnt converged at all yet, it would be better
        # to use the coarse 
        compute_fine_sampling = 'depth_fine' in ret_ref.keys()
        if compute_fine_sampling and hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
            and iteration < opt.max_iter * (opt.nerf.ratio_start_fine_sampling_at_x + 0.05):
            compute_fine_sampling = False
            
        if compute_fine_sampling:
            depth_ref = ret_ref.depth_fine.squeeze(0).squeeze(-1)
        else:
            depth_ref = ret_ref.depth.squeeze(0).squeeze(-1)

        pts3d_in_w_from_ref = batch_backproject_to_3d(kpi=pixels_ref, di=depth_ref, Ki=intr_ref, T_itoj=pose_c2w_ref)

        # sample an unseen pose, on which to project the pseudo gt 3d points. 
        pose_w2c_at_sampled = self.sample_pose(poses_c2w, id_self, pose_w2c_ref)
        intr_at_sampled = intr_ref.clone()

        return self.compute_loss_at_sampled_pose(data_dict, pose_w2c_at_unseen=pose_w2c_at_sampled, intr_at_unseen=intr_at_sampled, 
                                                 pseudo_gt_3dpts_in_w=pts3d_in_w_from_ref, H=H, W=W, 
                                                 pixels_in_ref=pixels_ref)

    def compute_loss_at_sampled_pose(self, data_dict: Dict[str, Any], pose_w2c_at_unseen: torch.Tensor, 
                                     intr_at_unseen: torch.Tensor, 
                                     pseudo_gt_3dpts_in_w: torch.Tensor, H: int, W: int, 
                                     pixels_in_ref: torch.Tensor
                                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """The core loss. 

        Args:
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            pose_w2c_at_unseen (torch.Tensor) : w2c pose of the unseen viewing direction, shape is (4,4)
            intr_at_unseen (torch.Tensor): intrinsics for the unseen viewing direction, shape is (3, 3)
            pseudo_gt_3dpts_in_w (torch.Tensor): pseudo gt 3D points in world coordinate frame, shape is (N, 3)
            H (int), W (int): size of image
            pixels_in_ref (torch.Tensor): Pixels in the reference training image, corresponding to 
                                          the pseudo gt. Shape is (N, 2)
        """
        iteration = data_dict['iter']
        stats_dict, plotting_dict = {'nbr_px_sampling': pseudo_gt_3dpts_in_w.shape[0]}, {}
        
        # project pseudo gt 3D points to unseen view ==> creates pseudo-gt depth. 
        pts_in_sampled_img, pseudo_gt_depth_in_sampled_img = batch_project(pseudo_gt_3dpts_in_w, 
                                                                           T_itoj=pose_w2c_at_unseen, Kj=intr_at_unseen, 
                                                                           return_depth=True)
        #(N, 2)
        
        valid_pts = pts_in_sampled_img[:, 0].ge(0.) & pts_in_sampled_img[:, 1].ge(0.) & \
            pts_in_sampled_img[:, 0].le(W-1) & pts_in_sampled_img[:, 1].le(H-1) & \
                pseudo_gt_depth_in_sampled_img.ge(data_dict.depth_range[0][0])

        pts_in_sampled_img = pts_in_sampled_img[valid_pts]
        pixels_in_ref = pixels_in_ref[valid_pts]
        pseudo_gt_3dpts_in_w = pseudo_gt_3dpts_in_w[valid_pts]
        pseudo_gt_depth_in_sampled_img = pseudo_gt_depth_in_sampled_img[valid_pts]

        if pseudo_gt_depth_in_sampled_img.shape[0] == 0:
            return {'depth_cons': torch.tensor(0., requires_grad=True).to(self.device)}, {}, {}


        # compute visibility mask
        with torch.no_grad():
            ret_max_depth_at_sampled = self.net.render_up_to_maxdepth_at_specific_pose_and_rays\
                (self.opt, data_dict, pose_w2c_at_unseen[:3], intr_at_unseen, H, W, 
                 depth_max=pseudo_gt_depth_in_sampled_img, pixels=pts_in_sampled_img, 
                 mode='train', iter=iteration)
            visibility_weight_ = ret_max_depth_at_sampled['all_cumulated_fine'].squeeze(0).unsqueeze(-1) if \
                'all_cumulated_fine' in ret_max_depth_at_sampled.keys() else \
                    ret_max_depth_at_sampled['all_cumulated'].squeeze(0).unsqueeze(-1) # (N, 1)
            assert visibility_weight_.le(1.).all()
        
        # threshold visibility and keep only valid points
        valid_mask = visibility_weight_.ge(0.2).reshape(-1)

        pts_in_sampled_img = pts_in_sampled_img[valid_mask]
        pixels_in_ref = pixels_in_ref[valid_mask]
        pseudo_gt_3dpts_in_w = pseudo_gt_3dpts_in_w[valid_mask]
        pseudo_gt_depth_in_sampled_img = pseudo_gt_depth_in_sampled_img[valid_mask]
        visibility_weight_ = visibility_weight_[valid_mask]


        if pseudo_gt_depth_in_sampled_img.shape[0] == 0:
            return {'depth_cons': torch.tensor(0., requires_grad=True).to(self.device)}, {}, {}
        
        # coarse rendering
        # compute nerf predictions at the projected pixel locations, with valid visibility mask. 
        ret_self = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose_w2c_at_unseen[:3], intr_at_unseen, H, W, 
                                                                   pixels=pts_in_sampled_img, mode='train', iter=iteration) 
        depth_rendered_at_sampled = ret_self.depth.squeeze().reshape(-1)  # (N)
        accuracy = ret_self.opacity.squeeze(0).detach() 
        visibility_weight = visibility_weight_ * accuracy
        
        loss_sampling = self.compute_diff_loss\
            (self.opt.diff_loss_type, 
             diff=pseudo_gt_depth_in_sampled_img.view(-1) - depth_rendered_at_sampled.view(-1), 
             weights=visibility_weight.view(-1)) 
        
        if 'rgb_fine' in ret_self.keys():
            depth_rendered_at_sampled = ret_self.depth_fine.squeeze().reshape(-1)  # (N)
            accuracy = ret_self.opacity_fine.squeeze(0).detach() 
            visibility_weight = visibility_weight_ * accuracy

            loss_sampling += self.compute_diff_loss\
                    (self.opt.diff_loss_type, 
                    diff=pseudo_gt_depth_in_sampled_img.view(-1) - depth_rendered_at_sampled.view(-1), 
                    weights=visibility_weight.view(-1)) 
        
        stats_dict['avg_vis_weight'] = visibility_weight.sum() / (visibility_weight.nelement() + 1e-6)
        loss = {'depth_cons': loss_sampling}

        if self.opt.gradually_decrease_depth_cons_loss:
            # gamma = 0.1**(max(iteration - self.opt.start_iter_photometric, 0)/self.opt.max_iter)
            # reduce the corres weight by 2 every x iterations 
            gamma = 2 ** (iteration // self.opt.depth_cons_loss_reduct_at_x_iter)
            loss['depth_cons'] = loss['depth_cons'] / gamma

        return loss, stats_dict, plotting_dict
