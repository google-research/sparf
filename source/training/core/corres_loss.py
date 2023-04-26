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
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.utils.camera import pose_inverse_4x4
from source.utils.config_utils import override_options
from source.training.core.base_corres_loss import CorrespondenceBasedLoss
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img


class CorrespondencesPairRenderDepthAndGet3DPtsAndReproject(CorrespondenceBasedLoss):
    """The main class for the correspondence loss of SPARF. It computes the re-projection error
    between previously extracted correspondences relating the input views. The projection
    is computed with the rendered depth from the NeRF and the current camera pose estimates. 
    """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(opt, nerf_net, flow_net, train_data, device)
        default_cfg = edict({'diff_loss_type': 'huber', 
                             'compute_photo_on_matches': False, 
                             'renderrepro_do_pixel_reprojection_check': False, 
                             'renderrepro_do_depth_reprojection_check': False, 
                             'renderrepro_pixel_reprojection_thresh': 10., 
                             'renderrepro_depth_reprojection_thresh': 0.1, 
                             'use_gt_depth': False,  # debugging
                             'use_gt_correspondences': False,  # debugging
                             'use_dummy_all_one_confidence': False # debugging
                             })
        self.opt = override_options(self.opt, default_cfg)
        self.opt = override_options(self.opt, opt)

    def compute_render_and_repro_loss_w_repro_thres(self, opt: Dict[str, Any], pixels_in_self_int: torch.Tensor, 
                                                    depth_rendered_self: torch.Tensor, intr_self: torch.Tensor, 
                                                    pixels_in_other: torch.Tensor, depth_rendered_other: torch.Tensor, 
                                                    intr_other: torch.Tensor, T_self2other: torch.Tensor, 
                                                    conf_values: torch.Tensor, stats_dict: Dict[str, Any], 
                                                    return_valid_mask: bool=False
                                                    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the actual re-projection error loss between 'self' and 'other' images, 
        along with possible filterings. 
        
        Args:
            opt (edict): settings
            pixels_in_self_int (torch.Tensor): (N, 2)
            depth_rendered_self (torch.Tensor): (N)
            intr_self (torch.Tensor): (3, 3)
            pixels_in_other (torch.Tensor): (N, 2)
            depth_rendered_other (torch.Tensor): (N)
            intr_other (torch.Tensor): (3, 3)
            T_self2other (torch.Tensor): (4, 4)
            conf_values (torch.Tensor): (N, 1)
            stats_dict (dict): dict to keep track of statistics to be logged
            return_valid_mask (bool, optional): Defaults to False.
        """
        pts_self_repr_in_other, depth_self_repr_in_other = batch_project_to_other_img(
            pixels_in_self_int.float(), di=depth_rendered_self, 
            Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)

        loss = torch.norm(pts_self_repr_in_other - pixels_in_other, dim=-1, keepdim=True) # [N_rays, 1]
        valid = torch.ones_like(loss).bool()
        if opt.renderrepro_do_pixel_reprojection_check:
            valid_pixel = loss.detach().le(opt.renderrepro_pixel_reprojection_thresh)
            valid = valid & valid_pixel
            stats_dict['perc_val_pix_rep'] = valid_pixel.sum().float() / (valid_pixel.nelement() + 1e-6)

        if opt.renderrepro_do_depth_reprojection_check:
            valid_depth = torch.abs(depth_rendered_other - depth_self_repr_in_other) / (depth_rendered_other + 1e-6)
            valid_depth = valid_depth.detach().le(opt.renderrepro_depth_reprojection_thresh)
            valid = valid & valid_depth.unsqueeze(-1)
            stats_dict['perc_val_depth_rep'] = valid_depth.sum().float() / (valid_depth.nelement() + 1e-6)

        loss_corres = self.compute_diff_loss(loss_type=opt.diff_loss_type, diff=pts_self_repr_in_other - pixels_in_other, 
                                             weights=conf_values, mask=valid, dim=-1)

        if return_valid_mask:
            return loss_corres, stats_dict, valid
        return loss_corres, stats_dict

    def compute_loss_on_image_pair(self, data_dict: Dict[str, Any], images: torch.Tensor, poses_w2c: torch.Tensor, 
                                   intrs: torch.Tensor, id_self: int, id_matching_view: int, 
                                   corres_map_self_to_other: torch.Tensor, corres_map_self_to_other_rounded_flat: torch.Tensor, 
                                   conf_map_self_to_other: torch.Tensor, mask_correct_corr: torch.Tensor, 
                                   pose_w2c_self: torch.Tensor, pose_w2c_other: torch.Tensor, intr_self: torch.Tensor, intr_other: torch.Tensor, 
                                   loss_dict: Dict[str, Any], stats_dict: Dict[str, Any], plotting_dict: Dict[str, Any]
                                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """ 
        Core of the loss function. 
        Args:
            data_dict (edict): Input data dict. Contains important fields:
                                - Image: GT images, (B, 3, H, W)
                                - intr: intrinsics (B, 3, 3)
                                - pose: GT w2c pose (B, 3, 4)
                                - poses_w2c: current estimate of w2c poses (being optimized)
                                - idx: idx of the images (B)
                                - depth_gt (optional): gt depth, (B, 1, H, W)
                                - valid_depth_gt (optional): (B, 1, H, W)
            images: Input images (B, H, W, 3) 
            poses_w2c: (B, 3, 4), current estimates of the poses
            intrs: image intrinsics (B, 3, 3)
            id_self
            id_matching_view
            corres_map_self_to_other (H, W, 2) float Tensor
            corres_map_self_to_other_rounded_flat: Contain idx of match in flattened (HW) tensor. 
                                                   Shape is (H, W, 1) Long Tensor
            conf_map_self_to_other (H, W)
            mask_correct_corr: (H, W, 1), bool Tensor
            pose_w2c_self: (4, 4)
            pose_w2c_other (4, 4)
            intr_self: (3, 3)
            intr_other: (3, 3)
            loss_dict: dictionary containing the losses
            stats_dict: dictionary containing the stats
            plotting_dict: dictionary containing plots
            plot: bool
            compute_sampling_loss: bool
        """
        # the actual render and reproject code
        iteration = data_dict['iter']
        B, H, W  = images.shape[:3]
        
        pixels_in_self = self.grid[mask_correct_corr]  # [N_ray, 2], absolute pixel locations
        ray_in_self_int = self.grid_flat[mask_correct_corr]  # [N_ray]

        pixels_in_other = corres_map_self_to_other[mask_correct_corr] # [N_ray, 2], absolute pixel locations
        ray_in_other_int = corres_map_self_to_other_rounded_flat[mask_correct_corr]  # [N_ray]
        conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1] 


        # in case there are too many values, subsamples
        if ray_in_self_int.shape[0] > self.opt.nerf.rand_rays // 2:
            random_values = torch.randperm(ray_in_self_int.shape[0],device=self.device)[:self.opt.nerf.rand_rays//2]
            ray_in_self_int = ray_in_self_int[random_values]
            pixels_in_self = pixels_in_self[random_values]

            pixels_in_other = pixels_in_other[random_values]
            ray_in_other_int = ray_in_other_int[random_values]
            conf_values = conf_values[random_values]
        
        
        ret_self = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose_w2c_self[:3], 
                                                                   intr_self, H, W, 
                                                                   pixels=pixels_in_self, mode='train', iter=iteration)
        # edict with rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine, shape [1, N_rays, K]
        

        ret_other = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, pose_w2c_other[:3], 
                                                                    intr_other, H, W, 
                                                                    pixels=pixels_in_other, mode='train', iter=iteration)                                                        

        if self.opt.compute_photo_on_matches:
            # compute the rgb loss for the two images
            image_self = images[id_self].view(-1, 3)[ray_in_self_int]
            image_other = images[id_matching_view].view(-1, 3)[ray_in_other_int]
            loss_photo_self = self.MSE_loss(ret_self.rgb.view(-1, 3), image_self)
            loss_photo_other = self.MSE_loss(ret_other.rgb.view(-1, 3), image_other)
            
            if 'rgb_fine' in ret_self.keys():
                loss_photo_self += self.MSE_loss(ret_self.rgb_fine.view(-1, 3), image_self)
                loss_photo_other += self.MSE_loss(ret_other.rgb_fine.view(-1, 3), image_other)
            loss_photo = (loss_photo_other + loss_photo_self) / 2.
            loss_dict['render_matches'] += loss_photo

        # compute the correspondence loss
        # for each image, project the pixel to the other image according to current estimate of pose 
        depth_rendered_self = ret_self.depth.squeeze(0).squeeze(-1)
        depth_rendered_other = ret_other.depth.squeeze(0).squeeze(-1)
        # ret_self.depth is [1, N_ray, 1], it needs to be [N_ray]
        stats_dict['depth_in_corr_loss'] = depth_rendered_self.detach().mean()

        T_self2other = pose_w2c_other @ pose_inverse_4x4(pose_w2c_self)

        # [N_ray, 2] and [N_ray]
        loss_corres, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
            (self.opt, pixels_in_self, depth_rendered_self, intr_self, 
             pixels_in_other, depth_rendered_other, intr_other, T_self2other, 
             conf_values, stats_dict)


        loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
            (self.opt, pixels_in_other, depth_rendered_other, intr_other, 
             pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other), 
             conf_values, stats_dict)  
        loss_corres += loss_corres_

        if 'depth_fine' in ret_other.keys():
            depth_rendered_self = ret_self.depth_fine.squeeze(0).squeeze(-1)
            depth_rendered_other = ret_other.depth_fine.squeeze(0).squeeze(-1)

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
                (self.opt, pixels_in_self, depth_rendered_self, intr_self, 
                pixels_in_other, depth_rendered_other, intr_other, T_self2other, 
                conf_values, stats_dict)
            loss_corres += loss_corres_

            loss_corres_, stats_dict = self.compute_render_and_repro_loss_w_repro_thres\
                (self.opt, pixels_in_other, depth_rendered_other, intr_other, 
                pixels_in_self, depth_rendered_self, intr_self, pose_inverse_4x4(T_self2other), 
                conf_values, stats_dict)  
            loss_corres += loss_corres_
        loss_corres = loss_corres / 4. if 'depth_fine' in ret_other.keys() else loss_corres / 2.
        loss_dict['corres'] = loss_corres
        return loss_dict, stats_dict, plotting_dict



