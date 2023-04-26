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
import lpips
import torch.nn as nn
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.training.core.regularization_losses import depth_patch_loss, lossfun_distortion


class Loss:
    """Overall loss module. Will compute each loss from a predefined list, 
        as well as combine their values. 
        
        Args:
            loss_modules (List): list of loss functions
    """
    def __init__(self, loss_modules: List[Callable[[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]], Any]]):

        self.loss_modules = loss_modules
    
    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Loops through all losses and computes them. At the end, computes final loss, as combination of the 
        different loss components. """
        loss = edict()
        stats_dict, plotting_dict = {}, {}
        for loss_module in self.loss_modules:
            loss_dict, stats_dict_, plotting_dict_ = \
                loss_module.compute_loss(opt, data_dict, output_dict, iteration=iteration,  
                                         mode=mode, plot=plot, **kwargs)
            loss.update(loss_dict)
            stats_dict.update(stats_dict_)
            plotting_dict.update(plotting_dict_)

        if opt.loss_weight.equalize_losses:
            loss = self.summarize_loss_w_equal_weights(opt, loss)
        else:
            loss = self.summarize_loss_w_predefined_weights(opt, loss)
        return loss, stats_dict, plotting_dict

    def compute_flow(self, train_data: Dict[str, Any], opt: Dict[str, Any]):
        """Compute the correspondences, if needed in each loss sub-module. """
        for loss_module in self.loss_modules:
            if hasattr(loss_module, 'compute_flow'):
                loss_module.compute_flow(train_data, opt)
        return 
        
    def plot_something(self) -> Dict[str, Any]:
        to_plot = {}
        for loss_module in self.loss_modules:
            if hasattr(loss_module, 'plot_something'):
                to_plot_ = loss_module.plot_something()
                to_plot.update(to_plot_)
        return to_plot
    
    def get_flow_metrics(self) -> Dict[str, Any]:
        """Compute flow/correspondences metrics, if correspondences are used in any of the sub-modules. """
        stats = {}
        for loss_module in self.loss_modules:
            if hasattr(loss_module, 'get_flow_metrics'):
                stats_ = loss_module.get_flow_metrics()
                stats.update(stats_)
        return stats

    def summarize_loss_w_equal_weights(self, opt: Dict[str, Any], loss_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Equalize the contribution of each loss term. """
        loss_all = 0.
        updated_loss = {}
        assert("all" not in loss_dict)
        # weigh losses

        assert 'render' in loss_dict
        render_loss = loss_dict.render.detach()
        for key in loss_dict:
            assert(key in opt.loss_weight)
            assert(loss_dict[key].shape==())

            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss_dict[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss_dict[key]),"loss {} is NaN".format(key)

                if loss_dict[key] != 0.:
                    w = render_loss / (loss_dict[key].detach() + 1e-6)
                else:
                    w = 1.

                weighted_loss = w * loss_dict[key]
                loss_all += weighted_loss
                updated_loss[key + '_after_w'] = weighted_loss
        loss_dict.update(all=loss_all)
        loss_dict.update(updated_loss)
        return loss_dict

    def summarize_loss_w_predefined_weights(self,opt: Dict[str, Any],loss_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Computes overall loss by summing the different loss terms, with predefined weights. """
        loss_all = 0.
        updated_loss = {}
        assert("all" not in loss_dict)
        # weigh losses

        for key in loss_dict:
            assert(key in opt.loss_weight)
            assert(loss_dict[key].shape==())

            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss_dict[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss_dict[key]),"loss {} is NaN".format(key)
                if opt.loss_weight.parametrization == 'exp':
                    # weights are 10^w
                    w = 10**float(opt.loss_weight[key]) 
                else:
                    w = float(opt.loss_weight[key])
                weighted_loss = w* loss_dict[key]
                loss_all += weighted_loss
                updated_loss[key + '_after_w'] = weighted_loss
        loss_dict.update(all=loss_all)
        loss_dict.update(updated_loss)
        return loss_dict



vgg = lpips.LPIPS(net="vgg")

class BaseLoss:
    def __init__(self, device: torch.device):
        """ Base class for all losses. """
        self.device = device
        self.vgg = vgg.to(self.device)

    def L1_loss(self, pred: torch.Tensor, label: torch.Tensor):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()
        
    def MSE_loss(self, pred: torch.Tensor, label: torch.Tensor):
        loss = (pred.contiguous()-label)**2
        return loss.sum() / (loss.nelement() + 1e-6)

    def huber_loss(self, pred: torch.Tensor, label: torch.Tensor, reduction: str='mean'):
        return torch.nn.functional.huber_loss(pred, label, reduction=reduction, delta=0.5) * 2.

    @staticmethod
    def length_sq(x: torch.Tensor):
        return torch.sum(x ** 2, dim=1, keepdims=True)
    
    def compute_regularization_losses(self, opt: Dict[str, Any], 
                                      output_dict: Dict[str, Any], loss: Dict[str, Any]
                                      ) -> Dict[str, Any]:
        """Regularization losses from the literature, ie the distortion or depth regularization losses """
        if opt.loss_weight.distortion is not None:
            distortion_loss_strength = 1e-3 * 2  # to compensate for the 2 in huber loss 
            loss_distortion = distortion_loss_strength * \
                lossfun_distortion(output_dict['t'], output_dict['weights'], normalize=False)
            if 'weights_fine' in output_dict:
               loss_distortion += distortion_loss_strength * \
                   lossfun_distortion(output_dict['t_fine'], output_dict['weights_fine'], normalize=False) 
            
            if 'distortion' in loss.keys():
                loss['distortion'] += loss_distortion
                loss['distortion'] /= 2.
            else:
                loss['distortion'] = loss_distortion
        
        # depth patch loss 
        if opt.loss_weight.depth_patch is not None:
            patch_loss_strength = 0.01 * 2  # to compensate for the 2 in huber loss 

            loss_depth_patch = patch_loss_strength * \
                depth_patch_loss(output_dict['depth'], patch_size=self.opt.depth_regu_patch_size)
            if 'depth_fine' in output_dict.keys():
                loss_depth_patch += patch_loss_strength * \
                    depth_patch_loss(output_dict['depth_fine'], patch_size=self.opt.depth_regu_patch_size)
            
            if 'depth_patch' in loss.keys():
                loss['depth_patch'] += loss_depth_patch
                loss['depth_patch'] /= 2. 
            else:
                loss['depth_patch'] = loss_depth_patch
        return loss

    def compute_diff_loss(self, loss_type: str, diff: torch.Tensor, weights: torch.Tensor=None, 
                          var: torch.Tensor=None, mask: torch.Tensor=None, dim=-1):
        if loss_type.lower() == 'epe':
            loss = torch.norm(diff, 2, dim, keepdim=True)
        elif loss_type.lower() == 'l1': 
            loss = torch.abs(diff)
        elif loss_type.lower() == 'mse':
            loss = diff**2
        elif loss_type.lower() == 'huber':
            delta = 1.
            loss = nn.functional.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=delta)
        else:
            raise ValueError('Wrong loss type: {}'.format(loss_type))
        
        if weights is not None:
            assert len(weights.shape) == len(loss.shape)

            loss = loss * weights

        if var is not None:
            eps = torch.tensor(1e-3)
            loss = loss / (torch.maximum(var, eps)) + torch.log(torch.maximum(var, eps))
            
        if mask is not None:
            assert len(mask.shape) == len(loss.shape)
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() + 1e-6)
        return loss.sum() / (loss.nelement() + 1e-6)

    def compute_flow_loss(self, flow_1, flow_2, weights=None, mask=None):
        assert flow_1.shape[-1] == 2 and flow_2.shape[-1] == 2
        diff = flow_1 - flow_2
        return self.compute_diff_loss(self.opt.flow_loss, diff, weights=weights, mask=mask)


class BasePhotoandReguLoss(BaseLoss):
    """Class responsable for computing the photometric loss (Huber or MSE), the mask loss 
    along with typical regularization losses (depth patch, distortion..). """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(device)
        self.opt = opt
        self.device = device
        self.net = nerf_net
        self.train_data = train_data

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
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
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """
        loss_dict = edict(render=torch.tensor(0., requires_grad=True).to(self.device))

        if iteration < self.opt.start_iter.photometric:
            return loss_dict, {}, {}
            
        # get all the images
        batch_size = len(data_dict.idx)
        image = data_dict.image.reshape(batch_size,3,-1).permute(0,2,1)  # (B, 3, H*W) and then (B, H*W, 3)

        fg_mask = None
        if opt.loss_weight.fg_mask is not None:
            fg_mask = data_dict.fg_mask
            fg_mask = fg_mask.float().view(batch_size, -1, 1)

        assert len(output_dict.idx_img_rendered) == batch_size
        # should be in same order than images
        nbr_rendered_images = len(output_dict.idx_img_rendered)

        if (hasattr(opt, 'nerf') and opt.nerf.rand_rays) and mode in ["train","test-optim"]:
            if len(output_dict.ray_idx.shape) == 2 and output_dict.ray_idx.shape[0] == nbr_rendered_images:
                n_rays = output_dict.ray_idx.shape[-1]
                #ray_idx is B, N-rays, ie different ray indices for each image in batch
                batch_idx_flattened = torch.arange(start=0, end=nbr_rendered_images)\
                    .unsqueeze(-1).repeat(1, n_rays).long().view(-1)
                ray_idx_flattened = output_dict.ray_idx.reshape(-1).long()
                image = image[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 3)
                image = image.reshape(nbr_rendered_images, n_rays, -1)  # (nbr_rendered_images, n_rays, 3)

                if fg_mask is not None:
                    fg_mask = fg_mask[batch_idx_flattened, ray_idx_flattened]  # (nbr_rendered_images*n_rays, 1)
                    fg_mask = fg_mask.reshape(nbr_rendered_images, n_rays, -1)
            else:
                image = image[:,output_dict.ray_idx] # (nbr_rendered_images, n_rays, 3)
                if fg_mask is not None:
                    fg_mask = fg_mask[:,output_dict.ray_idx] # (nbr_rendered_images, n_rays, 1)

        # compute image losses
        if self.opt.huber_loss_for_photometric:
            loss_dict.render = self.huber_loss(output_dict.rgb.reshape(nbr_rendered_images, -1, 3), image)
            if 'rgb_fine' in output_dict.keys():
                loss_dict.render += self.huber_loss(output_dict.rgb_fine.reshape(nbr_rendered_images, -1, 3), image)
        else:
            loss_dict.render = self.MSE_loss(output_dict.rgb.reshape(nbr_rendered_images, -1, 3), image)
            if 'rgb_fine' in output_dict.keys():
                loss_dict.render += self.MSE_loss(output_dict.rgb_fine.reshape(nbr_rendered_images, -1, 3), image)

        # mask loss
        if opt.loss_weight.fg_mask is not None:
            mask_loss_strength = 0.5
            mask_loss = mask_loss_strength * torch.abs(fg_mask - output_dict.opacity.reshape(nbr_rendered_images, -1, 1)).mean()
            if 'opacity_fine' in output_dict.keys():
                mask_loss += mask_loss_strength * torch.abs(fg_mask - output_dict.opacity_fine.reshape(nbr_rendered_images, -1, 1)).mean()
            loss_dict.fg_mask = mask_loss

        # REGULARIZATION LOSSES
        loss_dict = self.compute_regularization_losses(opt, output_dict, loss_dict)
        return loss_dict, {}, {}


class SparseCOLMAPDepthLoss(BaseLoss):
    """ DS-NeRF loss, i.e. L1 loss between rendered depth and depth obtained through COLMAP. 
        In DS-NeRF, they only triangulate points, considering fixed ground-truth poses. """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, device: torch.device):
        super().__init__(device)
        self.opt = opt
        self.net = nerf_net
        self.device = device

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): unused here, because need to render at the pixels for which a
                                 depth from COLMAP is available. 
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """
        if mode != 'train':
            return {}, {}, {}

        loss_dict = edict(colmap_depth=torch.tensor(0., requires_grad=True).to(self.device))
        stats_dict, plotting_dict = {}, {}

        # ground-truth data
        H, W = data_dict.image.shape[-2:]
        pose = self.net.get_w2c_pose(opt, data_dict, mode=mode)
        intr = data_dict.intr
        colmap_depth = data_dict.colmap_depth.view(-1, H, W)
        colmap_weight = data_dict.colmap_conf.view(-1, H, W)

        batch_size = pose.shape[0]
        stats_dict['perc_col_depth'] = len(torch.where(colmap_depth > 0)[0]) / colmap_depth.nelement()
        loss = torch.tensor(0., requires_grad=True).to(self.device)
        for ind_img in range(batch_size):
            colmap_depth_i = colmap_depth[ind_img]
            colmap_weight_i = colmap_weight[ind_img]
            valid_depth_ind_h, valid_depth_ind_w = torch.where(colmap_depth_i > 1e-6)

            if len(valid_depth_ind_h) == 0:
                continue 
            
            if len(valid_depth_ind_h) > opt.nerf.rand_rays//batch_size:
                random = torch.randperm(len(valid_depth_ind_h),device=self.device)[:opt.nerf.rand_rays//batch_size] 
                valid_depth_ind_h =  valid_depth_ind_h[random]
                valid_depth_ind_w = valid_depth_ind_w[random]

            colmap_depth_at_ray = colmap_depth_i[valid_depth_ind_h, valid_depth_ind_w].reshape(-1)
            colmap_weight_at_ray = colmap_weight_i[valid_depth_ind_h, valid_depth_ind_w].reshape(-1)

            # render 
            ray_idx = valid_depth_ind_h * W + valid_depth_ind_w
            ret_image = self.net.render_image_at_specific_pose_and_rays\
                (self.opt, data_dict, pose[ind_img], intr[ind_img], H, W, ray_idx=ray_idx, mode='train', iter=iteration)
            
            # compute the loss
            pred_depth = ret_image.depth.reshape(-1)
            loss_depth = torch.mean(((colmap_depth_at_ray - pred_depth) ** 2) * colmap_weight_at_ray)
            loss += loss_depth

            if 'depth_fine' in ret_image.keys():
                pred_depth = ret_image.depth_fine.reshape(-1)
                loss_depth = torch.mean(((colmap_depth_at_ray - pred_depth) ** 2) * colmap_weight_at_ray)
                loss += loss_depth

        loss_dict['colmap_depth'] = 0.1 * loss / batch_size
        # same weighting than in DS-NeRF
        return loss_dict, stats_dict, plotting_dict

