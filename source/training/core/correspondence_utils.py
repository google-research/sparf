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
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.utils.camera import pose_inverse_4x4
import source.utils.camera as camera
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img_and_check_depth
from source.utils.geometry.geometric_utils_numpy import get_absolute_coordinates
from third_party.DenseMatching.utils_flow.pixel_wise_mapping import warp_with_mapping
from source.utils.vis_rendering import colorize


class CorrrespondenceUtils:
    """Basic correspondence operations. """
    @torch.no_grad()
    def get_mask_valid_from_conf_map(self, p_r: torch.device, corres_map: torch.device, 
                                     min_confidence: float, max_confidence: float=None) -> torch.Tensor:
        """
        Get the valid mask from the predicted confidence and correspondence maps. 
        Args:
            p_r: (H, W, 1)
            corres_map: (H, W, 2)
        returns:
            valid mask (torch.Bool), with shape (H, W, 1)
        """
        return get_mask_valid_from_conf_map(p_r, corres_map, min_confidence, max_confidence=max_confidence)

    def get_gt_correspondence_maps_all_to_all(self, n_views: int) -> torch.Tensor:
        """For debugging, creates the ground-truth correspondence maps relating the images, using the
        ground-truth depths and ground-truth poses. 
        outpit is (n_views, n_views, 3, H, W). First 2 channels are the correspondence map, last channel is the 
        valid mask. Exhaustive matching of all views to all views (including itself). """
        all_corres_map_and_mask = []
        for id_self in range(n_views):
            corres_map_and_mask_from_id_self = []
            for id_matching_view in range(n_views):
                if id_self == id_matching_view:
                    corres_map = self.grid.unsqueeze(0)  # (1, H, W, 2)
                    mask = torch.ones_like(corres_map[:, :, :, :1])  # (1, H, W, 1)
                    corres_map_and_mask_from_id_self.append(torch.cat((corres_map, mask), dim=-1))
                else:
                    corres_map, mask = get_correspondences_gt(self.train_data.all, idx_target=id_self, idx_source=id_matching_view)
                    # corres_map is (H, W, 2) and mask is (H, W)
                    corres_map_and_mask_from_id_self.append(torch.cat((corres_map, mask.unsqueeze(-1).float()), dim=-1).unsqueeze(0))
            corres_map_and_mask_from_id_self = torch.cat(corres_map_and_mask_from_id_self, dim=0)  # (N, H, W, 3)
            all_corres_map_and_mask.append(corres_map_and_mask_from_id_self.unsqueeze(0))
        corres_map_and_mask = torch.cat(all_corres_map_and_mask, dim=0).permute(0, 1, 4, 2, 3)  # (N, N, H, W, 3) and then (N, N, 3, H, W)
        return corres_map_and_mask

    def get_flow_metrics_confident(self, pred_corres_map: torch.Tensor=None, 
                                   pred_valid_mask: torch.Tensor=None) -> Dict[str, Any]:
        """Get the flow/correspondence error metrics in predicted confident regions. """
        stats_dict = {}

        if pred_corres_map is None:
            pred_corres_map = self.corres_maps  # (M, 2, H, W)
        
        if pred_valid_mask is None:
            pred_valid_mask = self.mask_valid_corr  # (M, 1, H, W)
        
        flow_pairs = np.array(self.flow_pairs)  # (len(pairs), 2), first column is target and second is source
        flow_pairs_target = flow_pairs[:, 0]
        flow_pairs_source = flow_pairs[:, 1]

        gt_corres_map_all_to_all = self.gt_corres_map_and_mask_all_to_all[:, :, :2]  # (N, N, 2, H, W)
        gt_valid_mask_all_to_all = self.gt_corres_map_and_mask_all_to_all[:, :, 2:3].bool()  # (N, N, 1, H, W) torch.BoolTensor
        gt_corres_map = gt_corres_map_all_to_all[flow_pairs_target, flow_pairs_source]  # (M, 2, H, W)
        gt_valid_mask = gt_valid_mask_all_to_all[flow_pairs_target, flow_pairs_source]  # (M, 1, H, W)

        error = torch.norm(pred_corres_map.float()-gt_corres_map.float(), dim=1, keepdim=True)

        error_conf = error[gt_valid_mask & pred_valid_mask]
        stats_dict['avg_epe_in_conf'] = error_conf.mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_1_in_conf'] = error_conf.le(1.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_3_in_conf'] = error_conf.le(3.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        return stats_dict
        
    def get_flow_metrics(self, pred_corres_map: torch.Tensor=None, 
                         pred_valid_mask: torch.Tensor=None) -> Dict[str, Any]:
        """Get flow/correspondence error metrics. """
        stats_dict = {}
        if self.gt_corres_map_and_mask_all_to_all is None:
            return stats_dict

        if pred_corres_map is None:
            pred_corres_map = self.corres_maps  # (M, 2, H, W)
        
        if pred_valid_mask is None:
            pred_valid_mask = self.mask_valid_corr  # (M, 1, H, W)
        
        flow_pairs = np.array(self.flow_pairs)  # (len(pairs), 2), first column is target and second is source
        flow_pairs_target = flow_pairs[:, 0]
        flow_pairs_source = flow_pairs[:, 1]

        gt_corres_map_all_to_all = self.gt_corres_map_and_mask_all_to_all[:, :, :2]  # (N, N, 2, H, W)
        gt_valid_mask_all_to_all = self.gt_corres_map_and_mask_all_to_all[:, :, 2:3].bool()  # (N, N, 1, H, W) torch.BoolTensor
        gt_corres_map = gt_corres_map_all_to_all[flow_pairs_target, flow_pairs_source]  # (M, 2, H, W)
        gt_valid_mask = gt_valid_mask_all_to_all[flow_pairs_target, flow_pairs_source]  # (M, 1, H, W)

        error = torch.norm(pred_corres_map.float()-gt_corres_map.float(), dim=1, keepdim=True)
        error_all = error[gt_valid_mask]
        stats_dict['avg_epe_all'] = error_all.mean() if len(error_all) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_1_all'] = error_all.le(1.).float().mean() if len(error_all) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_3_all'] = error_all.le(3.).float().mean() if len(error_all) > 0 else torch.as_tensor(0.)

        error_conf = error[gt_valid_mask & pred_valid_mask]
        stats_dict['avg_epe_in_conf'] = error_conf.mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_1_in_conf'] = error_conf.le(1.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        stats_dict['avg_pck_3_in_conf'] = error_conf.le(3.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)
        return stats_dict

    def plot_something(self):
        """Plotting of the predicted correspondences. """
        plotting = {}
        if hasattr(self, 'flow_plot') and self.flow_plot is not None:
            plotting['predicted_corr'] = self.flow_plot
        if hasattr(self, 'flow_plot_masked') and self.flow_plot_masked is not None:
            plotting['predicted_corr_masked'] = self.flow_plot_masked 
        return plotting

    @torch.no_grad()
    def plot_flow_and_conf_map(self, corres_map_target_to_source: torch.Tensor, 
                               conf_map_target_to_source: torch.Tensor, target: torch.Tensor, 
                               source: torch.Tensor, valid_mask: torch.Tensor=None):
        """ 
        Args:
            corres_map_target_to_source: torch.Tensor (H, W, 2) or (2, H, W)
            conf_map_target_to_source: torch.Tensor (H, W, 1)
            target (H, W, 3) or (3, H, W)
            valid_mask: (H, W)
        """
        return plot_flow_and_conf_map(corres_map_target_to_source, conf_map_target_to_source, target, 
                                      source, valid_mask=valid_mask)

    def compute_relative_pose_diff(self, pose_aligned_w2c, pose_GT_w2c) -> Tuple[torch.Tensor, torch.Tensor]:
        # just invert both poses to camera to world
        # so that the translation corresponds to the position of the camera in world coordinate frame. 
        pose_aligned_c2w = camera.pose.invert(pose_aligned_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
        R_aligned_c2w,t_aligned_c2w = pose_aligned_c2w.split([3,1],dim=-1)
        # R_aligned is (B, 3, 3)
        t_aligned_c2w = t_aligned_c2w.reshape(-1, 3)  # (B, 3)

        R_GT_c2w,t_GT_c2w = pose_GT_c2w.split([3,1],dim=-1)
        t_GT_c2w = t_GT_c2w.reshape(-1, 3)
        R_error = camera.rotation_distance(R_aligned_c2w,R_GT_c2w)
        t_error = (t_aligned_c2w - t_GT_c2w).norm(dim=-1)
        return R_error, t_error
    
    
# ----------------- Utils pair selection -------------------------------

def batched_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def image_pair_candidates_with_angular_distance(extrinsics_: torch.Tensor, 
                                                pairing_angle_threshold: float=60) -> torch.Tensor:
    """Compute image pair list that have an initial angular distance below #pairing_angle_threshold

    Args:
        extrinsics_ (torch.Tensor): w2c poses of shape (B, 3, 4)
        pairing_angle_threshold (int, optional): Acceptable angle between valid pairs. Defaults to 60.

    Returns:
        torch array containing indexes of possible pairs, (Nx2)
    """

    # i_map is used when provided extrinsics are not having sequentiall 
    # index. i_map is a list of ints where each element corresponds to 
    # image index. 
    
    pairs = []
    eps = 1e-7
    num_images = extrinsics_.shape[0]
    extrinsics = torch.eye(4).unsqueeze(0).repeat(num_images, 1, 1).to(extrinsics_.device)
    extrinsics[:, :3] = extrinsics_
    
    for i in range(num_images):
        
        rot_mat_i = extrinsics[i, :3, :3]
        
        for j in range(i + 1, num_images):
            
            rot_mat_j = extrinsics[j, :3, :3]
            rot_mat_ij = rot_mat_i @ torch.inverse(rot_mat_j)
            angle_rad = torch.acos(((torch.trace(rot_mat_ij) - 1) / 2).clamp(-1+eps,1-eps))
            angle_deg = angle_rad / np.pi * 180
            
            if torch.abs(angle_deg) < pairing_angle_threshold:    
                pairs.append([i, j])
    
    pairs = np.array(pairs)  # N x 2
    if len(pairs) == 0:
        pairs = np.zeros((0, 2))
    return torch.from_numpy(pairs.T)

def generate_pair_list(n_views: int):
    """Generate list of possible exhaustive pairs, (Nx2). """
    pairs = []
    for i in range(n_views):
        for j in range(i+1, n_views):
            pairs.append([i, j])
    # pairs is N
    pairs = np.array(pairs)  # N x 2
    return torch.from_numpy(pairs.T)

# ---------------------- Flow/correspondence processing utils -----------------------------

def get_mask_valid_from_conf_map(p_r: torch.Tensor, corres_map: torch.Tensor, 
                                 min_confidence: float, max_confidence: float=None) -> torch.Tensor:
    channel_first = False
    if len(corres_map.shape) == 4:
        # (B, 2, H, W) or (B, H, W, 2)
        if corres_map.shape[1] == 2:
            corres_map = corres_map.permute(0, 2, 3, 1)
            channel_first = True
        if len(p_r.shape) == 3:
            p_r = p_r.unsqueeze(-1)
        if p_r.shape[1] == 1:
            p_r = p_r.permute(0, 2, 3, 1)
        h, w = corres_map.shape[1:3]
        valid_matches = corres_map[:, :, :, 0].ge(0) & corres_map[:, :, :, 0].le(w-1) & corres_map[:, :, :, 1].ge(0) & corres_map[:, :, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (B, H, W, 1)
        if channel_first:
            mask = mask.permute(0, 3, 1, 2)
    else:
        if corres_map.shape[0] == 2:
            corres_map = corres_map.permute(1, 2, 0)
        if len(p_r.shape) == 2:
            p_r = p_r.unsqueeze(-1)
            channel_first = True
        if p_r.shape[0] == 1:
            p_r = p_r.unsqueeze(1, 2, 0)
        h, w = corres_map.shape[:2]
        valid_matches = corres_map[:, :, 0].ge(0) & corres_map[:, :, 0].le(w-1) & corres_map[:, :, 1].ge(0) & corres_map[:, :, 1].le(h-1)
        mask = p_r.ge(min_confidence)
        if max_confidence is not None:
            mask = mask & p_r.le(max_confidence)
        mask = mask & valid_matches.unsqueeze(-1)  # (H, W, 1)
        if channel_first:
            mask = mask.permute(2, 0, 1)
    return mask 


# ---------------------------- Plotting utils ----------------------------------

def plot_flow_and_conf_map(corres_map_target_to_source: torch.Tensor, conf_map_target_to_source: torch.Tensor, 
                           target: torch.Tensor, source: torch.Tensor, valid_mask: torch.Tensor=None):
    """ 
    Args:
        corres_map_target_to_source: torch.Tensor (H, W, 2) or (2, H, W)
        conf_map_target_to_source: torch.Tensor (H, W, 1)
        target (H, W, 3) or (3, H, W)
        valid_mask: (H, W)
    """

    if source.shape[0] != 3:
        source = source.permute(2, 0, 1)
    if target.shape[0] != 3:
        target = target.permute(2, 0, 1)
    if corres_map_target_to_source.shape[0] != 2:
        corres_map_target_to_source = corres_map_target_to_source.permute(2, 0, 1)
    H, W = target.shape[-2:]
    if valid_mask is None:
        valid_mask = conf_map_target_to_source.ge(0.95).squeeze()
        # valid_mask = torch.ones(H, W).to(corres_map_target_to_source.device).bool()

    warped_source = warp_with_mapping(source.unsqueeze(0), corres_map_target_to_source.unsqueeze(0).float())
    warped_source = warped_source.squeeze(0)
    masked_warped_source = warped_source * valid_mask.squeeze()[None].float()

    conf_map = colorize(conf_map_target_to_source.squeeze(), range=[0., 1.], channel_first=True)

    flow_plot = torch.from_numpy(flow_to_image(corres_map_target_to_source.permute(1, 2, 0)\
        .detach().cpu().numpy())).permute(2, 0, 1).to(conf_map.device)  # (3, H, W)
    alpha = 0.7
    overlaid = target.clone()
    overlaid = overlaid * alpha + (1-alpha) * valid_mask.squeeze()[None].float()
    overlaid.permute(1, 2, 0)[valid_mask] = warped_source.permute(1, 2, 0)[valid_mask]
    return torch.cat((source, target, warped_source, flow_plot, conf_map, masked_warped_source, overlaid), dim=-1)



# ------------------------------- debugging with ground-truth ---------------------

def get_correspondences_gt(data_dict: Dict[str, Any], idx_target: int, 
                           idx_source: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Computes ground-truth correspondence map using gt depth  map and poses. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
    """
    H, W = data_dict['image'].shape[-2:]
    pixels_target = get_absolute_coordinates(H, W, True).reshape(-1, 2).cuda()  # (H, W, 2)
    depth_source = data_dict['depth_gt'][idx_source].reshape(H, W)
    
    depth_target = data_dict['depth_gt'][idx_target].reshape(-1)
    valid_depth_target = data_dict['valid_depth_gt'][idx_target].reshape(H, W)

    K_target = data_dict['intr'][idx_target]
    K_source = data_dict['intr'][idx_source]
    w2c_target_ = data_dict['pose'][idx_target]
    w2c_target = torch.eye(4).cuda()
    w2c_target[:3] = w2c_target_

    w2c_source_ = data_dict['pose'][idx_source]
    w2c_source = torch.eye(4).cuda()
    w2c_source[:3] = w2c_source_

    target2source = w2c_source @ pose_inverse_4x4(w2c_target)
    repr_in_source, visible = batch_project_to_other_img_and_check_depth(
        pixels_target, di=depth_target, depthj=depth_source, Ki=K_target, Kj=K_source, 
        T_itoj=target2source, validi=valid_depth_target.reshape(-1), rth=0.05)
    corres_target_to_source = repr_in_source.reshape(H, W, 2)
    valid = corres_target_to_source[:, :, 0].ge(0) & corres_target_to_source[:, :, 1].ge(0) & \
        corres_target_to_source[:, :, 0].le(W-1) & corres_target_to_source[:, :, 1].le(H-1)
    valid = valid & valid_depth_target & visible.reshape(H, W)

    '''
    target = data_dict['image'][idx_target].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    source = data_dict['image'][idx_source].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    corres_map = corres_target_to_source.cpu().numpy()
    warped = remap_using_correspondence_map(source, corres_map.astype(np.float32)[:, :, 0], corres_map.astype(np.float32)[:, :, 1])
    image = np.concatenate((source, target, warped), axis=1)
    import cv2
    cv2.imwrite('test_img_{}_to_{}.png'.format(idx_target, idx_source), image * 255)
    '''
    return corres_target_to_source, valid




