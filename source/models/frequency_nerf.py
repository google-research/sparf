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
import torch.nn as nn
import torch.nn.functional as torch_F
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.utils import camera


def get_layer_dims(layers: torch.nn.Module):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1],layers[1:]))


def l2_normalize(x: torch.Tensor, eps: torch.Tensor = torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""

    return x / torch.sqrt(
        torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
    )

    
class FrequencyEmbedder:
    """Positional embedding. """
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt

    def __call__(self, opt: Dict[str, Any], input: torch.Tensor, L: int) -> torch.Tensor:
        shape = input.shape
        if opt.arch.posenc.log_sampling:
            if opt.arch.posenc.include_pi_in_posenc:
                freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)*np.pi # [L]
            else:
                """
                Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
                in_channels: number of input channels (3 for both xyz and direction)
                https://github.com/bmild/nerf/issues/12
                # in GNerf
                https://github.com/quan-meng/gnerf/blob/a008c63dba3a0f7165e912987942c47972759879/model/rendering.py#L5
                """
                freq = 2**torch.arange(L,dtype=torch.float32,device=input.device)
            # similar to 
            # freq = 2.**torch.linspace(0., L-1, steps=L)*np.pi
        else:
            freq = torch.linspace(2.**0., 2.**(L-1), steps=L)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc


class NeRF(torch.nn.Module):
    """MLP network corresponding to NeRF. """
    def __init__(self, opt: Dict[str, Any], is_fine_network: bool = False):
        super().__init__()
        self.opt = opt
        self.define_network(opt, is_fine_network=is_fine_network)

        if opt.barf_c2f is None:
            # will be useless but just to be on the safe side. 
            # that corresponds to high frequency positional encoding
            self.progress = torch.nn.Parameter(torch.tensor(1.)) 
        else:
            self.progress = torch.nn.Parameter(torch.tensor(0.)) 
        # use Parameter so it could be checkpointed

    def define_network(self, opt: Dict[str, Any], is_fine_network: bool = False):
        input_3D_dim = 0
        if opt.arch.posenc.add_raw_3D_points:
            input_3D_dim += 3
        input_3D_dim += 6*opt.arch.posenc.L_3D if opt.arch.posenc.L_3D>0 else 0
        assert input_3D_dim > 0

        if opt.nerf.view_dep:
            input_view_dim = 0
            if opt.arch.posenc.add_raw_rays:
                input_view_dim += 3
            input_view_dim += 6*opt.arch.posenc.L_view if opt.arch.posenc.L_view>0 else 0
            assert input_view_dim > 0

        # point-wise feature
        self.mlp_feat = torch.nn.ModuleList()
        layers_feat = opt.arch.layers_feat_fine if is_fine_network and opt.arch.layers_feat_fine is not None else opt.arch.layers_feat
        L = get_layer_dims(layers_feat)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(L)-1: k_out += 1 
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="first" if li==len(L)-1 else None)
            self.mlp_feat.append(linear)

        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        L = get_layer_dims(opt.arch.layers_rgb)
        feat_dim = layers_feat[-1]
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = feat_dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="all" if li==len(L)-1 else None)
            self.mlp_rgb.append(linear)
        return

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.tensorflow_init_weights(self.opt, m)
        return

    def choose_activation(self, opt: Dict[str, Any]):
        act_ = nn.ReLU(True)
        return act_

    def tensorflow_init_weights(self, opt: Dict[str, Any], linear: torch.nn.Module, out: str = None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)
        return

    def compute_raw_density(self, opt: Dict[str, Any], points_3D_samples: torch.Tensor, 
                            embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor]
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if opt.arch.posenc.L_3D > 0:
            assert embedder_pts is not None
            points_3D_enc = self.positional_encoding(opt,points_3D_samples,embedder_fn=embedder_pts,
                                                     L=opt.arch.posenc.L_3D)
            if opt.arch.posenc.add_raw_3D_points:
                points_3D_enc = torch.cat([points_3D_samples,points_3D_enc],dim=-1) # [B,...,6L+3]
        else: points_3D_enc = points_3D_samples        
    
        feat = points_3D_enc
        # extract coordinate-based features
        for li,layer in enumerate(self.mlp_feat):
            # now there is an intermediate activation layer, so need to skip this one int the skip count
            if li in opt.arch.skip: feat = torch.cat([feat,points_3D_enc],dim=-1)
            feat = layer(feat)
            if li==len(self.mlp_feat)-1:
                raw_density = feat[...,0]
                feat = feat[...,1:]
            feat = torch_F.relu(feat)
        return raw_density, feat

    def forward(self, opt: Dict[str, Any], points_3D_samples: torch.Tensor, ray: torch.Tensor, 
                embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor],
                embedder_view: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                mode: str = None) -> Dict[str, Any]: # [B,...,3]
        """
        MLP predictions at given 3D points (with corresponding given ray directions)
        Args:
            opt (edict): settings
            points_3D_samples (torch.Tensor): shape (B, N, N_samples, 3)
            ray (torch.Tensor): shape (B, N, 3)
            embedder_pts (function): positional encoding function for 3D points
            embedder_view (function): positional encoding function for viewing directions
            mode (str, optional): Defaults to None.
        """
        
        raw_density, feat = self.compute_raw_density\
            (opt, points_3D_samples, embedder_pts)

        # predict density from raw density (apply activation)
        if opt.nerf.density_noise_reg and mode=="train":
            raw_density += torch.randn_like(raw_density)*opt.nerf.density_noise_reg

        density_activ = getattr(torch_F,opt.arch.density_activ) # relu_,abs_,sigmoid_,exp_....
        density = density_activ(raw_density)

        # compute colors
        ray_enc = None
        if opt.nerf.view_dep:
            assert ray is not None
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]

            if opt.arch.posenc.L_view > 0:
                assert embedder_view is not None
                # add high frequency positional encoding for viewing direction
                ray_enc = self.positional_encoding(opt,ray_unit_samples,embedder_fn=embedder_view,
                                                   L=opt.arch.posenc.L_view)
                if opt.arch.posenc.add_raw_rays:
                    ray_enc = torch.cat([ray_unit_samples,ray_enc],dim=-1) # [B,...,6L+3]
            else: ray_enc = ray_unit_samples

            feat = torch.cat([feat,ray_enc],dim=-1) # [B, N, 6L+3]

        for li,layer in enumerate(self.mlp_rgb):
            feat = layer(feat)
            # the last layer should not have the activation
            if li!=len(self.mlp_rgb)-1:
                feat = torch_F.relu(feat)
        rgb = feat.sigmoid_() # [B,...,3]

        pred = dict(
            rgb_samples=rgb,  # [B, n_rays, n_samples, 3]
            density_samples=density, # [B, n_rays, n_samples, 1]
        )
        return pred


    def positional_encoding(self, opt: Dict[str, Any], input: torch.Tensor, 
                            embedder_fn: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                            L: int) -> torch.Tensor: # [B,...,N]
        """Apply the coarse-to-fine positional encoding strategy of BARF. 

        Args:
            opt (edict): settings
            input (torch.Tensor): shaps is (B, ..., C) where C is channel dimension
            embedder_fn (function): positional encoding function
            L (int): Number of frequency basis
        returns:
            positional encoding
        """
        shape = input.shape
        input_enc = embedder_fn(opt, input, L) # [B,...,2NL]

        # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        # the progress could also be fixed but then we finetune without cf_pe, then it is just not updated 
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=input.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2

            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc

    def forward_samples(self, opt: Dict[str, Any], center: torch.Tensor, ray: torch.Tensor,  
                        depth_samples: torch.Tensor, 
                        embedder_pts: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                        embedder_view: Callable[[Dict[str, Any], torch.Tensor, int], torch.Tensor], 
                        mode: str=None) -> Dict[str, Any]:
        """MLP prediction given camera centers, ray directions and depth samples. 
        Args:
            opt (edict): settings
            center (torch.Tensor): shape (B, N, 3)
            ray (torch.Tensor): shape (B, N, 3)
            depth_samples (torch.Tensor): shape (B, N, N_samples, 1) Depth sampled along the ray
            embedder_pts (function): positional encoding function for 3D points
            embedder_view (function): positional encoding function for viewing directions
            mode (str, optional): Defaults to None.
        """
        points_3D_samples = camera.get_3D_points_from_depth(center,ray,depth_samples,
                                                            multi_samples=True) # [B,HW,N,3]
        
        pred_dict = self.forward(opt,points_3D_samples=points_3D_samples,
                                 ray=ray,embedder_pts=embedder_pts,
                                 embedder_view=embedder_view, mode=mode) # [B,HW,N],[B,HW,N,3]
        return pred_dict

    def composite(self,opt: Dict[str, Any],ray: torch.Tensor,
                  pred_dict: Dict[str, Any], depth_samples: torch.Tensor) -> Dict[str, Any]:
        """Transforms model's predictions to semantically meaningful values.
        Args:
            here N corresponds to num_samples along ray
            ray: [B, num_rays, 3] 
            pred_dict: Dict predicted by MLP. Must contain:
                rgb_samples [B, num_rays, N, 3] already passed through sigmoid
                density_samples: [N, num_rays, num_samples along ray]. predicted density for each sample (after through softplus)
            
            depth_samples: [N, num_rays, N, 1]. Depth sampled along the ray
        Returns:
            rgb: [B, num_rays, 3]. Estimated RGB color of a ray.
            depth: [B, num_rays, 1]. Depth map. Estimated distance to object.
            opacity: [B, num_rays, 1]. Sum of weights along each ray.
            weights: [N, num_rays, num_samples]. Weights assigned to each sampled color.
        """

        rgb_samples,density_samples = pred_dict['rgb_samples'], pred_dict['density_samples']
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)

        # computes the interval between the samples
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]

        sigma_delta = density_samples*dist_samples # [B,HW,N]

        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]  # density between 0 and 1 in this sample
        T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]

        # probability that the ray travels from origin to that point without hitting any particule
        # can only stay the same or decrease
        all_cumulated = T[:, :, -2].clone()

        weights = (T*alpha)[...,None] # [B,HW,N,1]

        # integrate RGB and depth weighted by weights

        # pdf = weights / (weights.sum(dim=2, keepdims=True) + 1e-6)
        # depth = (depth_samples*pdf).sum(dim=2) # [B,HW,1]
        # depth_var = (pdf*(depth_samples - depth.unsqueeze(-1))**2).sum(dim=2) # [B,HW,1] 
        
        depth = (depth_samples*weights).sum(dim=2) # [B,HW,1]
        depth_var = (weights*(depth_samples - depth.unsqueeze(-1))**2).sum(dim=2) # [B,HW,1] 

        # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth), depth / (torch.sum(weights, 2) +1e-6))

        rgb = (rgb_samples*weights).sum(dim=2) # [B,HW,3]
        rgb_var = ((rgb_samples-rgb.unsqueeze(-2)).sum(dim=-1, keepdims=True)*weights).sum(dim=2) # [B,HW,1]

        opacity = weights.sum(dim=2) # [B,HW,1]

        if opt.nerf.setbg_opaque or opt.mask_img:
            rgb = rgb + (1.-opacity)

        pred_dict.update(rgb=rgb,rgb_var=rgb_var,depth=depth,depth_var=depth_var,opacity=opacity,weights=weights,
                         all_cumulated=all_cumulated)
    
        return pred_dict
