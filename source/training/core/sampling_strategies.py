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
import numpy as np 
import cv2
from typing import List, Tuple, Union, Dict, Any, Optional


class RaySamplingStrategy:
    """Computes which rays we should sample.

    There are multiple ways of sampling: 
        - either randomly everywhere (except in border of size patch size)  
        - the ones that hit the dilated foreground mask.
        - Within a center bounding box

    Args:
        opt: settings
        data_dict: the data dict.
        device
    """
    def __init__(self, opt: Dict[str, Any], data_dict: Dict[str, Any], device: torch.device):
        self.opt = opt
        self.device = device


        self.nbr_images, _, self.H, self.W = data_dict.image.shape

        self.all_possible_pixels = self.get_all_samples(data_dict)  # [HW, 2]
        # rays = self.all_possible_pixels[:, 1] * self.W + self.all_possible_pixels[:, 0]

        self.all_center_pixels = self.get_all_center_pixels(data_dict)  # [N, 2]

        if self.opt.sample_fraction_in_fg_mask > 0.:
            self.in_mask_pixels, self.min_nbr_in_mask = self.samples_in_mask(data_dict)

        # pixels in patch
        y_range = torch.arange(self.opt.depth_regu_patch_size, dtype=torch.long,device=self.device)
        x_range = torch.arange(self.opt.depth_regu_patch_size, dtype=torch.long,device=self.device)
        Y,X = torch.meshgrid(y_range,x_range) # [patch_size,patch_size]
        self.dxdy = torch.stack([X,Y],dim=-1).view(-1,2) # [patch_size**2,2]

    def samples_in_mask(self, data_dict: Dict[str, Any]):
        """Sample pixels/rays within dilated foregroud masks"""
        assert 'fg_mask' in data_dict.keys()
        B, _, H, W = data_dict.image.shape
        mask_fg = data_dict.fg_mask  # (B, 1, H, W)
        assert mask_fg.shape == (B, 1, H, W)

        dilated_masks = []
        dilation = 10
        for ind in range(B):
            mask_fg_numpy = mask_fg[ind].squeeze(0).cpu().numpy().astype(np.float32)
            dilated_mask_ = cv2.dilate(mask_fg_numpy, np.ones((3, 3)), iterations=dilation) > 0
            dilated_mask_ = torch.from_numpy(dilated_mask_)
            dilated_masks.append(dilated_mask_)
        dilated_masks = torch.stack(dilated_masks, dim=0).to(self.device)  # (B, H, W)

        mask_for_patch = torch.zeros_like(dilated_masks).bool()
        mask_for_patch[:, :H-self.opt.depth_regu_patch_size-1, :W-self.opt.depth_regu_patch_size-1] = True

        dilated_masks = dilated_masks & mask_for_patch

        ind_b, ind_h, ind_w = torch.where(dilated_masks)  # ()
        pixels_per_el = []
        min_el = float('inf')
        for ind_el in range(B):
            mask_el = ind_b == ind_el
            pixels_el = torch.stack((ind_w[mask_el], ind_h[mask_el]), dim=-1)  # (n_ray_el, 2)
            # ray_el = ind_h[mask_el] * W + ind_w[mask_el] # (n_ray_el)

            min_el = min(min_el, len(pixels_el))
            pixels_per_el.append(pixels_el)
        return pixels_per_el, min_el
    
    def get_all_samples(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Samples all pixels/rays """
        H, W = data_dict.image.shape[-2:]
        if self.opt.loss_weight.depth_patch is not None:
            # exclude the patch size 
            y_range = torch.arange(H-self.opt.depth_regu_patch_size-1,dtype=torch.long,device=self.device)
            x_range = torch.arange(W-self.opt.depth_regu_patch_size-1,dtype=torch.long,device=self.device)
        else:
            y_range = torch.arange(H,dtype=torch.long,device=self.device)
            x_range = torch.arange(W,dtype=torch.long,device=self.device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
        # ray_idx = Y*W + X  # [HW]
        return xy_grid.long()  # ray_idx.view(-1).long()

    def get_all_center_pixels(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Sample all pixels/rays within center bounding box"""
        H, W = data_dict.image.shape[-2:]
        dH = int(H//2 * self.opt.precrop_frac)
        dW = int(W//2 * self.opt.precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        coords = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        return coords.long().to(self.device)


    def compute_pixel_coords_for_patch(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """Compute pixel coords for a patch."""
        patch_size = self.opt.depth_regu_patch_size
        shape_ = pixel_coords.shape[:-1]
        x_ind = pixel_coords.view(-1, 2)[..., 0]
        y_ind = pixel_coords.view(-1, 2)[..., 1]


        x_ind = (x_ind[:, None].repeat(1, patch_size**2) + self.dxdy[:, 0]).reshape(-1)
        y_ind = (y_ind[:, None].repeat(1, patch_size**2) + self.dxdy[:, 1]).reshape(-1)

        pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(shape_ + (patch_size**2, -1))
        return pixel_coords

    def __call__(self, nbr_pixels: int, sample_in_center: bool=False, 
                 idx_imgs: List[int]=None) -> torch.Tensor:
        """Samples #nbr_pixels randomly, within the pool allowed by the different options. """
        nbr_images = self.nbr_images
        if idx_imgs is not None:
            nbr_images = len(idx_imgs)

        nbr_pixels_per_img = nbr_pixels // nbr_images
        nbr_rand_pixels_per_img = nbr_pixels // nbr_images

        if self.opt.loss_weight.depth_patch is not None:
            nbr_pixels_per_img = nbr_pixels_per_img // self.opt.depth_regu_patch_size ** 2
            nbr_rand_pixels_per_img = nbr_rand_pixels_per_img // self.opt.depth_regu_patch_size ** 2

        pixels_in_mask, pixels_in_center = None, None
        if self.opt.sample_fraction_in_fg_mask > 0.:
            nbr_mask_pixels_per_img = min(self.min_nbr_in_mask, \
                int(nbr_rand_pixels_per_img*self.opt.sample_fraction_in_fg_mask))
            nbr_rand_pixels_per_img = nbr_rand_pixels_per_img - nbr_mask_pixels_per_img

            pixels_in_mask = []
            idx_imgs = np.arange(nbr_images) if idx_imgs is None else idx_imgs
            for id in idx_imgs:
                pixels_el = self.in_mask_pixels[id]  # (N_rays, 2)
                pixels_el = pixels_el[torch.randperm(len(pixels_el), device=self.device)[:nbr_mask_pixels_per_img]]
                pixels_in_mask.append(pixels_el)
            pixels_in_mask = torch.stack(pixels_in_mask, dim=0)  # (B, n_m, 2)
        elif self.opt.sampled_fraction_in_center > 0:
            nbr_mask_pixels_per_img = int(nbr_rand_pixels_per_img*self.opt.sampled_fraction_in_center)
            nbr_rand_pixels_per_img = nbr_rand_pixels_per_img - nbr_mask_pixels_per_img
            random_rays_idx = torch.randperm(len(self.all_center_pixels), device=self.device)[:nbr_mask_pixels_per_img]
            pixels_in_center = self.all_center_pixels[random_rays_idx]  # (n_r, 2)

        if sample_in_center:
            random_rays_idx = torch.randperm(len(self.all_center_pixels), device=self.device)[:nbr_rand_pixels_per_img]
            random_pixels = self.all_center_pixels[random_rays_idx]  # (n_r, 2)
        else:
            random_rays_idx = torch.randperm(len(self.all_possible_pixels), device=self.device)[:nbr_rand_pixels_per_img]
            random_pixels = self.all_possible_pixels[random_rays_idx]  # (n_r, 2)

        if pixels_in_mask is not None:
            random_pixels = random_pixels.unsqueeze(0).repeat(nbr_images, 1, 1)
            random_pixels = torch.cat((random_pixels, pixels_in_mask), dim=1)  # (B, n_r+n_m, 2)
        if pixels_in_center is not None:
            random_pixels = torch.cat((random_pixels, pixels_in_center), dim=0)  # (n_r + n_m, 2)

        if self.opt.loss_weight.depth_patch is not None:
            # put the points as depth patch loss
            random_pixels = self.compute_pixel_coords_for_patch(random_pixels)  
            # (B, (n_r+n_m), patchsize**2, 2) or (n_r, patchsize**2, 2)
            if len(random_pixels.shape) == 4:
                random_pixels = random_pixels.reshape(nbr_images, -1, 2)  # (B*(n_r+n_m), patchsize**2, 2)
            else:
                random_pixels = random_pixels.reshape(-1, 2)  # (n_r*patchsize**2, 2)
        
        random_rays = random_pixels[..., 1] * self.W + random_pixels[..., 0]
        return random_rays 


def sample_rays_for_patch(H: int, W: int, patch_size: int, precrop_frac: float=0.5, 
                          fraction_in_center: float=0., nbr: int=None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples pixels/rays with patch formatting, ie the output shape 
    is (N, patch_size**2, 2)/(N, patch_size**2)"""

    # patch
    y_range = torch.arange(patch_size, dtype=torch.long)
    x_range = torch.arange(patch_size, dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [patch_size,patch_size]
    dxdy = torch.stack([X,Y],dim=-1).view(-1,2) # [patch_size**2,2]

    # all pixels
    y_range = torch.arange(H-(patch_size - 1),dtype=torch.long)
    x_range = torch.arange(W-(patch_size - 1),dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2).long() # [HW,2]
    n = xy_grid.shape[0]
    x_ind = xy_grid[..., 0]
    y_ind = xy_grid[..., 1]

    if fraction_in_center > 0.:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        pixels_in_center = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        if nbr is not None:
            nbr_center = int(nbr*fraction_in_center)
            nbr_all = nbr - nbr_center
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr_all]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]

            idx = torch.randperm(len(pixels_in_center), device=x_ind.device)[:nbr_center]
            pixels_in_center = pixels_in_center[idx] # [N, 2]
            x_ind = torch.cat((x_ind, pixels_in_center[..., 0]))
            y_ind = torch.cat((y_ind, pixels_in_center[..., 1]))  # 
    else:
        if nbr is not None:
            # select a subset of those
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]
            
    n = len(x_ind)

    x_ind = (x_ind[:, None].repeat(1, patch_size**2) + dxdy[:, 0]).reshape(-1)
    y_ind = (y_ind[:, None].repeat(1, patch_size**2) + dxdy[:, 1]).reshape(-1)

    pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(n, patch_size**2, -1)

    rays = pixel_coords[..., 1] * W + pixel_coords[..., 0]

    return pixel_coords.float(), rays  # (N, patch_size**2, 2), (N, patch_size**2)


def sample_rays(H: int, W: int, precrop_frac: float=0.5, 
                fraction_in_center: float=0., nbr: int=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample pixels/rays within the image, the output formatting is (N, 2)/(N)"""

    # all pixels
    y_range = torch.arange(H - 1,dtype=torch.long)
    x_range = torch.arange(W - 1,dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2).long() # [HW,2]
    n = xy_grid.shape[0]
    x_ind = xy_grid[..., 0]
    y_ind = xy_grid[..., 1]

    if fraction_in_center > 0.:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        pixels_in_center = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        if nbr is not None:
            nbr_center = int(nbr*fraction_in_center)
            nbr_all = nbr - nbr_center
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr_all]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]

            idx = torch.randperm(len(pixels_in_center), device=x_ind.device)[:nbr_center]
            pixels_in_center = pixels_in_center[idx] # [N, 2]
            x_ind = torch.cat((x_ind, pixels_in_center[..., 0]))
            y_ind = torch.cat((y_ind, pixels_in_center[..., 1]))  # 
            n = len(x_ind)
    else:
        if nbr is not None:
            # select a subset of those
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]
            
    n = len(x_ind)
    pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(n, -1)

    rays = pixel_coords[..., 1] * W + pixel_coords[..., 0]

    return pixel_coords.float(), rays  # (N, 2), (N)