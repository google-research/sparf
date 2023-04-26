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
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

from third_party.pytorch_ssim.ssim import ssim as ssim_loss



def MSE_loss(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor=None):
    loss = (pred.contiguous()-label)**2
    if mask is not None:
        assert len(loss.shape) == len(mask.shape)
        loss = loss[mask.repeat(1, loss.shape[1], 1, 1)]
    return loss.mean()

def compute_mse_on_rays(data_dict: Dict[str, Any], output_dict: Dict[str, Any]):
    """
    Computes MSE between rendered colors at rays and gt colors
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
        output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
    """
    # get all the images
    batch_size = len(data_dict.idx)
    image = data_dict.image.reshape(batch_size,3,-1).permute(0,2,1)  # (B, 3, H*W) and then (B, H*W, 3)

    idx_img_rendered = output_dict.idx_img_rendered
    nbr_rendered_images = len(output_dict.idx_img_rendered)
    image = image[idx_img_rendered].reshape(nbr_rendered_images, -1, 3)

    if 'ray_idx' in output_dict.keys():
        if len(output_dict.ray_idx.shape) == 2 and output_dict.ray_idx.shape[0] == nbr_rendered_images:
            n_rays = output_dict.ray_idx.shape[-1]
            #ray_idx is B, N-rays, ie different ray indices for each image in batch
            batch_idx_flattened = torch.arange(start=0, end=nbr_rendered_images)\
                .unsqueeze(-1).repeat(1, n_rays).long().view(-1)
            ray_idx_flattened = output_dict.ray_idx.reshape(-1).long()
            image = image[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 3)
            image = image.reshape(nbr_rendered_images, n_rays, -1)  # (nbr_rendered_images, n_rays, 3)
        else:
            image = image[:,output_dict.ray_idx] # (nbr_rendered_images, n_rays, 3)
    
    # compute image losses
    mse_coarse = MSE_loss(output_dict.rgb.reshape(nbr_rendered_images, -1, 3), image)
    mse_fine = None
    if 'rgb_fine' in output_dict.keys():
        mse_fine = MSE_loss(output_dict.rgb_fine.reshape(nbr_rendered_images, -1, 3), image)
    return mse_coarse, mse_fine

    
def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())

def compute_depth_error_on_rays(data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                                pred_depth: torch.Tensor, scaling_factor_for_pred_depth: float=1.):
    """
    Computes depth error between rendered depth at rays and gt depth. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
        output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
        pred_depth: rendered depth (corresponds to depth or depth_fine keys of output_dict), (B, N, 1)
        scaling_factor_for_pred_depth (float): to scale the rendered depth to be similar to gt depth. 
    """
    idx_img_rendered = output_dict.idx_img_rendered
    B = len(idx_img_rendered)

    depth_gt = data_dict.depth_gt[idx_img_rendered].view(B, -1, 1)
    valid_depth = data_dict.valid_depth_gt[idx_img_rendered].view(B, -1)
    if 'ray_idx' in output_dict.keys():
        ray_idx = output_dict.ray_idx
        if len(ray_idx.shape) == 2 and ray_idx.shape[0] == B:
            # different ray idx for each image in batch
            n_rays = ray_idx.shape[-1]
            #ray_idx is B, N-rays, ie different ray indices for each image in batch
            batch_idx_flattened = torch.arange(start=0, end=B)\
                .unsqueeze(-1).repeat(1, n_rays).long().view(-1)
            ray_idx_flattened = ray_idx.reshape(-1).long()
            depth_gt = depth_gt[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 1)
            depth_gt = depth_gt.reshape(B, n_rays, -1)  # (nbr_rendered_images, n_rays, 1)

            valid_depth = valid_depth[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 1)
            valid_depth = valid_depth.reshape(B, n_rays, -1)  # (nbr_rendered_images, n_rays, 1)
        else:
            # when random ray, only rendered some rays
            depth_gt = depth_gt[:, ray_idx]
            valid_depth = valid_depth[:, ray_idx]
    
    depth_gt = depth_gt[valid_depth]
    pred_depth = pred_depth[valid_depth] * scaling_factor_for_pred_depth

    abs_e = torch.abs(depth_gt - pred_depth)
    abs_e = abs_e.sum() / (abs_e.nelement() + 1e-6) # same than rmse

    rmse = compute_rmse(depth_gt, pred_depth)
    return abs_e, rmse

def compute_depth_error(data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                        pred_depth: torch.Tensor, scaling_factor_for_pred_depth: float=1.):
    """
    Computes depth error between rendered depth and gt depth, for a full image. Here N is HW. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
        output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
        pred_depth: rendered depth (corresponds to depth or depth_fine keys of output_dict), (B, N, 1)
        scaling_factor_for_pred_depth (float): to scale the rendered depth to be similar to gt depth. 
    """
    def compute_metric(depth_gt, depth):
        # mse = torch.sqrt((depth_gt - depth)**2)
        abs_e = torch.abs(depth_gt - depth)
        abs_e = abs_e.sum() / (abs_e.nelement() + 1e-6)
        abs_e = abs_e.item()

        rmse = compute_rmse(depth_gt, depth).item()
        return abs_e, rmse

    idx_img_rendered = output_dict.idx_img_rendered
    B = len(idx_img_rendered)

    pred_depth = pred_depth.view(B, -1, 1)
    depth_gt = data_dict.depth_gt[idx_img_rendered].view(B, -1, 1)
    valid_depth = data_dict.valid_depth_gt[idx_img_rendered].view(B, -1)

    depth_gt = depth_gt[valid_depth]
    pred_depth = pred_depth[valid_depth] 
    if scaling_factor_for_pred_depth != 1.:
        abs_e_no_s, rmse_no_s = compute_metric(depth_gt, pred_depth)
        abs_e_s, rmse_s = compute_metric(depth_gt, pred_depth.clone() * scaling_factor_for_pred_depth)
        abs_e = min(abs_e_no_s, abs_e_s)
        rmse = min(rmse_no_s, rmse_s)
    else:
        abs_e, rmse = compute_metric(depth_gt, pred_depth)

    return abs_e, rmse

def compute_metrics_masked(data_dict: Dict[str, Any], pred_rgb_map: torch.Tensor, 
                           gt_rgb_map: torch.Tensor, 
                           lpips_loss: Callable[[torch.Tensor], torch.Tensor], suffix:str=''):
    """Compute metrics by masking the rendered image and ground-truth image 
    with the ground-truth foreground mask. 
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
                           - fg_mask: (B, 1, H, W)
        pred_rgb_map: rendered color (B, 3, H, W)
        gt_rgb_map: gt color map (B, 3, H, W)
    """
    assert 'fg_mask' in data_dict.keys()
    mask_float = data_dict.fg_mask.float()  # (B, 3, H, W)
    assert mask_float.shape[1] == 3 or mask_float.shape[1] == 1

    mask = (mask_float == 1.)

    rgb_map_fg = pred_rgb_map * mask_float + (1. - mask_float)
    gt_rgb_map_fg = gt_rgb_map * mask_float + (1. - mask_float)

    psnr_masked = -10*MSE_loss(rgb_map_fg, gt_rgb_map_fg, mask).log10().item()
    ssim_masked = ssim_loss(rgb_map_fg, gt_rgb_map_fg).item()
    lpips_masked = lpips_loss(rgb_map_fg*2-1, gt_rgb_map_fg*2-1).item()
    results = {'psnr_masked'+suffix: psnr_masked, 'ssim_masked'+suffix: ssim_masked, 'lpips_masked'+suffix: lpips_masked}
    return results

def compute_metrics(data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                    pred_rgb_map: torch.Tensor, pred_depth: torch.Tensor, gt_rgb_map: torch.Tensor, 
                    lpips_loss: Callable[[torch.Tensor], torch.Tensor], 
                    scaling_factor_for_pred_depth: float=1., suffix: str=''):
    """
    Computes RGB and depth metrics between rendered and gt values, on full images. 
    If the foreground mask is also available, computes metrics by masking the background as well. 
    Here N is HW
     
    Args:
        data_dict (edict): Input data dict. Contains important fields:
                           - Image: GT images, (B, 3, H, W)
                           - intr: intrinsics (B, 3, 3)
                           - idx: idx of the images (B)
                           - depth_gt (optional): gt depth, (B, 1, H, W)
                           - valid_depth_gt (optional): (B, 1, H, W)
        output_dict (edict): Output dict from the renderer. Contains important fields
                             - idx_img_rendered: idx of the images rendered (B), useful 
                             in case you only did rendering of a subset
                             - ray_idx: idx of the rays rendered, either (B, N) or (N)
                             - rgb: rendered rgb at rays, shape (B, N, 3)
                             - depth: rendered depth at rays, shape (B, N, 1)
                             - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                             - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
        pred_rgb_map: rendered color (corresponds to rgb or rgb_fine keys of output dict, reshaped), (B, 3, ...)
        pred_depth: rendered depth (corresponds to depth or depth_fine keys of output_dict), (B, N, 1)
        gt_rgb_map: gt color map (B, 3, ...)
        scaling_factor_for_pred_depth (float): to scale the rendered depth to be similar to gt depth. 
    """
    psnr = -10*MSE_loss(pred_rgb_map, gt_rgb_map).log10().item()
    ssim = ssim_loss(pred_rgb_map, gt_rgb_map).item()
    lpips = lpips_loss(pred_rgb_map*2-1,gt_rgb_map *2-1).item()

    abs_e, rmse = float('nan'), float('nan')
    # if depth is available, compute depth error
    if 'depth_gt' in data_dict.keys():
        if scaling_factor_for_pred_depth != 1.:
            abs_e_no_s, rmse_no_s = compute_depth_error(data_dict, output_dict, pred_depth, 1.)
            abs_e_s, rmse_s = compute_depth_error(data_dict, output_dict, pred_depth, scaling_factor_for_pred_depth)
            abs_e = min(abs_e_no_s, abs_e_s)
            rmse = min(rmse_no_s, rmse_s)
        else:
            abs_e, rmse = compute_depth_error(data_dict, output_dict, pred_depth, scaling_factor_for_pred_depth)

    # store results
    results = edict({'psnr' + suffix: psnr, 'ssim' + suffix: ssim, 'lpips' + suffix: lpips, 
                     'abse_depth' + suffix: abs_e, 'rmse_depth'+suffix: rmse}) 

    # compute metrics in fg mask
    if 'fg_mask' in data_dict.keys():
        results.update(compute_metrics_masked(data_dict, pred_rgb_map, gt_rgb_map, lpips_loss, suffix))
    return results