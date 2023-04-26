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
from typing import List, Union, Any, Optional, Dict, Tuple, Dict


from source.utils import camera
from source.models.frequency_nerf import FrequencyEmbedder, NeRF


class Graph(torch.nn.Module):
    """NeRF model (mlp prediction + rendering). """
    def __init__(self, opt: Dict[str, Any], device: torch.device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.define_renderer(opt)
    
    def define_renderer(self, opt: Dict[str, Any]):
        self.nerf = NeRF(opt).to(self.device)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt, is_fine_network=True).to(self.device)
        self.embedder_pts = FrequencyEmbedder(self.opt)
        self.embedder_view = FrequencyEmbedder(self.opt)
        return 

    def re_initialize(self):
        self.nerf.initialize()
        if self.opt.nerf.fine_sampling:
            self.nerf_fine.initialize()

    def get_network_components(self):
        ret = [self.nerf]
        if self.opt.nerf.fine_sampling:
            ret.append(self.nerf_fine)
        return ret

    def L1_loss(self, pred: torch.Tensor,label: torch.Tensor):
        loss = (pred.contiguous()-label).abs()
        return loss.mean()
        
    def MSE_loss(self, pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor = None):
        loss = (pred.contiguous()-label)**2
        if mask is not None:
            loss = loss[mask]
        return loss.mean()

    def get_w2c_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any], mode: str = None):
        # pose is fixed and known here 
        # this is ground-truth here
        # this function is overwritten in joint pose_nerf trainer.
        return data_dict.pose  
    
    def get_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any], mode: str = None):
        return self.get_w2c_pose(opt, data_dict, mode)
    
    def get_c2w_pose(self, opt: Dict[str, Any], data_dict: Dict[str, Any], mode: str = None):
        return camera.pose.invert(self.get_w2c_pose(opt, data_dict, mode))
    
    def forward(self,opt: Dict[str, Any], data_dict: Dict[str, Any], iter: int, 
                img_idx: Union[List[int], int] = None, mode: str = None) -> Dict[str, Any]:
        """Rendering of a random subset of pixels or all pixels for all images of the data_dict. 

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
            iter (int): iteration
            img_idx (list, optional): In case we want to render from only a subset of the
                                      images contained in data_dict.image. Defaults to None.
            mode (str, optional): Defaults to None.
        """
        batch_size = len(data_dict.idx)
        pose = self.get_w2c_pose(opt,data_dict,mode=mode) # current w2c pose estimates
        # same than data_dict.poses_w2c
        
        # render images
        H, W = data_dict.image.shape[-2:]

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if img_idx is not None:
            # we only render some images of the batch, not all of them
            ray_idx = None 
            nbr_img = len(img_idx) if isinstance(img_idx, list) else 1
            if opt.nerf.rand_rays and mode in ["train","test-optim"]:
                # sample random rays for optimization
                ray_idx = torch.randperm(H*W,device=self.device)[:opt.nerf.rand_rays//nbr_img]  # the same rays are rendered for all the images here 
            ret = self.render_image_at_specific_rays(opt, data_dict, img_idx, ray_idx=ray_idx)
            if ray_idx is not None:
                ret.ray_idx = ray_idx
            ret.idx_img_rendered = img_idx
            return ret

        # we render all images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            ray_idx = torch.randperm(H*W,device=self.device)[:opt.nerf.rand_rays//batch_size]  
            # the same rays are rendered for all the images here 
            ret = self.render(opt,pose,intr=data_dict.intr,ray_idx=ray_idx,mode=mode, H=H, 
                              W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]
            ret.ray_idx = ray_idx
        else:
            # render full image (process in slices)
            ret = self.render_by_slices(opt,pose,intr=data_dict.intr,mode=mode, 
                                        H=H, W=W, depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=data_dict.intr,mode=mode, H=H, W=W, 
                              depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]

        ret.idx_img_rendered = torch.arange(start=0, end=batch_size).to(self.device)  
        # all the images were rendered, corresponds to their index in the tensor
        return ret
    
    def render_image_at_specific_pose_and_rays(self, opt: Dict[str, Any], data_dict: Dict[str, Any], pose: torch.Tensor, 
                                               intr: torch.Tensor, H: int, W: int, iter: int, 
                                               pixels: torch.Tensor = None, ray_idx: torch.Tensor = None, 
                                               mode: str = 'train', per_slice: str = False) -> Dict[str, Any]:
        """Rendering of a specified set of pixels (or all) at a predefined pose. 

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
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations. 
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
            per_slice: render per slice? in case the number of pixels is too large to be processed all at once. 
        """
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0)

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if ray_idx is None and pixels is None:
            # rendere the full image
            ret = self.render_by_slices(opt,pose,intr=intr,mode=mode, H=H, W=W, \
                depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=intr,mode=mode, H=H, W=W, \
                      depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]
        else:
            # render only the ray_idx or pixels specified. If too large, use per_slice here. 
            ret = self.render_by_slices_w_custom_rays(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                                                      H=H, W=W, depth_range=depth_range, iter=iter) if per_slice else \
                  self.render(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                              H=H, W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]   
            ret.ray_idx = ray_idx  
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        return ret

    def render_image_at_specific_rays(self, opt: Dict[str, Any], data_dict: Dict[str, Any], iter: int, 
                                      img_idx: Union[List[int], int] = None, pixels: torch.Tensor = None, 
                                      ray_idx: torch.Tensor = None, mode: str = 'train', per_slice: str = False) -> Dict[str, Any]:
        """Rendering of a specified set of pixels for all images of data_dict. 

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
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations. 
                             (L, N, 2) or (N, 2)  / (L, N) or (N). 
                             where L is len(img_idx) is img_idx is not None else the number of images in the batch, ie L=B
            img_idx (list, optional): In case we want to render from only a subset of the
                                      images contained in data_dict.image. Defaults to None.
            mode (str, optional): Defaults to None.
            per_slice: render per slice? in case the number of pixels is too large to be processed all at once. 
        """
        pose = self.get_w2c_pose(opt,data_dict,mode=mode)
        intr = data_dict.intr
        batch_size = pose.shape[0]

        if img_idx is not None:
            if isinstance(img_idx, (tuple, list)):
                N = len(img_idx)
                pose = pose[img_idx].view(-1, 3, 4)  # (N, 3, 4)
                intr = intr[img_idx].view(-1, 3, 3)  # (N, 3, 3)
            else:
                pose = pose[img_idx].unsqueeze(0)  # (1, 3, 4)
                intr = intr[img_idx].unsqueeze(0)  # (1, 3, 3)
                img_idx = [img_idx]  # make it a list of a single element
        H, W = data_dict.image.shape[-2:]

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        if ray_idx is None and pixels is None:
            # rendere the full image
            ret = self.render_by_slices(opt,pose,intr=intr,mode=mode, H=H, W=W, depth_range=depth_range, iter=iter) if opt.nerf.rand_rays else \
                  self.render(opt,pose,intr=intr,mode=mode, H=H, W=W, depth_range=depth_range, iter=iter) # [B,HW,3],[B,HW,1]
        else:
            ret = self.render_by_slices_w_custom_rays(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                                                      H=H, W=W, depth_range=depth_range, iter=iter) if per_slice else \
                  self.render(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                              H=H, W=W, depth_range=depth_range, iter=iter) # [B,N,3],[B,N,1]  
            ret.ray_idx = ray_idx
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        ret.idx_img_rendered = torch.from_numpy(np.array(img_idx)).to(self.device) if img_idx is not None else \
            torch.arange(start=0, end=batch_size).to(self.device) 
        return ret

    def render(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int, intr: torch.Tensor, pixels: torch.Tensor = None, 
               ray_idx: torch.Tensor = None, depth_range: List[float] = None, iter: int = None, mode: int = None) -> Dict[str, Any]:
        """Main rendering function

        Args:
            opt (edict): settings
            pose (torch.Tensor): w2c poses at which to render (L, 3, 4) or (3, 4)
            intr (torch.Tensor): corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W (int, int): size of rendered image
            depth_range (list): Min and max value for the depth sampling along the ray. 
                                The same for all rays here. 
            iter (int): iteration
            pixels, ray_idx  (torch.Tensor): if any of the two is specified, will render only at these locations. 
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        batch_size = len(pose)

        if pixels is not None:
            center, ray = camera.get_center_and_ray_at_pixels(pose, pixels, intr=intr)
            # [B, N_rays, 3] and [B, N_rays, 3]
        else:
            # from ray_idx
            center,ray = camera.get_center_and_ray(pose, H, W, intr=intr) # [B,HW,3]
            while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
                center,ray = camera.get_center_and_ray(pose, H, W,intr=intr) # [B,HW,3]
        
            if ray_idx is not None:
                if len(ray_idx.shape) == 2 and ray_idx.shape[0] == batch_size:
                    n_rays = ray_idx.shape[-1]
                    #ray_idx is B, N-rays, ie different ray indices for each image in batch
                    batch_idx_flattened = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat(1, n_rays).long().view(-1)
                    ray_idx_flattened = ray_idx.reshape(-1).long()

                    # consider only subset of rays
                    center,ray = center[batch_idx_flattened, ray_idx_flattened],ray[batch_idx_flattened, ray_idx_flattened]  # [B*N_rays, 3]
                    center = center.reshape(batch_size, n_rays, 3)
                    ray = ray.reshape(batch_size, n_rays, 3)
                else:
                    # consider only subset of rays
                    # ray_idx is (-1)
                    center,ray = center[:,ray_idx],ray[:,ray_idx]  # [B, N_rays, 3]

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP

        pred = edict(origins=center, viewdirs=ray)

        # opt.nerf.sample_stratified means randomness in depth vlaues sampled
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1], 
                                            n_samples=opt.nerf.sample_intvs, H=H, W=W, 
                                            depth_range=depth_range, mode=mode) # [B,HW,N,1] or [B, N_rays, N, 1]
        pred_coarse = self.nerf.forward_samples(opt,center,ray,depth_samples,
                                                embedder_pts=self.embedder_pts, 
                                                embedder_view=self.embedder_view, mode=mode)
        pred_coarse['t'] = depth_samples
        # contains rgb_samples [B, N_rays, N, 3] and density_samples [B, N_rays, N]
        pred_coarse = self.nerf.composite(opt,ray,pred_coarse,depth_samples)
        # new elements are rgb,depth,depth_var, opacity,weights,loss_weight_sparsity 
        # [B,HW,K] or [B, N_rays, K], loss_weight_sparsity is [B]
        weights = pred_coarse['weights']

        pred.update(pred_coarse)

        # render with fine MLP from coarse MLP
        compute_fine_sampling = True
        if hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x:
            compute_fine_sampling = False

        if opt.nerf.fine_sampling and compute_fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                # weights are [B, num_rays, Nf, 1]
                det = mode not in ['train', 'test-optim'] or (not opt.nerf.sample_stratified)
                depth_samples_fine = self.sample_depth_from_pdf(opt, weights=weights[...,0], 
                                                                n_samples_coarse=opt.nerf.sample_intvs, 
                                                                n_samples_fine=opt.nerf.sample_intvs_fine, 
                                                                depth_range=depth_range, det=det) # [B,HW,Nf,1]
                # print(depth_samples_fine.min(), depth_samples_fine.max())
                depth_samples_fine = depth_samples_fine.detach()

            depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
 
            depth_samples = depth_samples.sort(dim=2).values
            pred_fine = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,
                                                       embedder_pts=self.embedder_pts, 
                                                       embedder_view=self.embedder_view, mode=mode)
            
            pred_fine['t'] = depth_samples
            pred_fine = self.nerf_fine.composite(opt,ray,pred_fine,depth_samples)
            pred_fine = edict({k+'_fine': v for k, v in pred_fine.items()})
            pred.update(pred_fine)
        return pred

    def render_by_slices(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int, 
                         intr: torch.Tensor, depth_range: List[float], iter: int, mode: str = None) -> Dict[str, Any]:
        """Main rendering function for the entire image of size (H, W). By slice because
        two many rays to render in a single forward pass. 

        Args:
            opt (edict): settings
            pose (torch.Tensor): w2c poses at which to render (L, 3, 4)
            intr (torch.Tensor): corresponding intr matrices (L, 3, 3)
            H, W (int, int): size of rendered image
            depth_range (list): Min and max value for the depth sampling along the ray. 
                                The same for all rays here. 
            iter (int): iteration
            mode (str, optional): Defaults to None.
        """
        ret_all = edict(rgb=[],rgb_var=[],depth=[],depth_var=[], opacity=[], 
                        normal=[], all_cumulated=[])
        if opt.nerf.fine_sampling:
            if (hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
                and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x):
                # we skip the fine sampling
                ret_all.update({})
            else:   
                ret_all.update(rgb_fine=[],rgb_var_fine=[], depth_fine=[],
                            depth_var_fine=[], opacity_fine=[], normal_fine=[], all_cumulated_fine=[])
        # render the image by slices for memory considerations
        for c in range(0,H*W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,H*W),device=self.device)
            ret = self.render(opt,pose,H=H, W=W, intr=intr,ray_idx=ray_idx,depth_range=depth_range, iter=iter, mode=mode) # [B,R,3],[B,R,1]
            for k in ret_all: 
                if k in ret.keys(): ret_all[k].append(ret[k])
            torch.cuda.empty_cache()
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1) if len(ret_all[k]) > 0 else None
        return ret_all

    def sample_depth(self, opt: Dict[str, Any], batch_size: int, n_samples: int, H: int, W: int, depth_range: List[float], \
                     num_rays: int = None, mode: str = None) -> torch.Tensor:
        """Sample depths along ray. The same depth range is applied for all the rays. 

        Args:
            opt (edict): settings
            batch_size (int): 
            n_samples (int): Number of depth samples along each ray
            H (int): img size
            W (int): img size
            depth_range (list)
            num_rays (int, optional): Defaults to None.
            mode (str, optional): Defaults to None.

        Returns:
            depth_samples (torch.Tesnor), shape # [B,num_rays,n_samples,1]
        """

        depth_min, depth_max = depth_range
        num_rays = num_rays or H*W
        
        if opt.nerf.sample_stratified and mode not in ['val', 'eval', 'test']:
            # random samples
            rand_samples = torch.rand(batch_size,num_rays,n_samples,1,device=self.device)  # [B,HW,N,1]
        else:
            rand_samples = 0.5 * torch.ones(batch_size,num_rays,n_samples,1,device=self.device)

        rand_samples += torch.arange(n_samples,device=self.device)[None,None,:,None].float() # the added part is [1, 1, N, 1] ==> [B,HW,N,1]
        depth_samples = rand_samples/n_samples*(depth_max-depth_min)+depth_min # [B,HW,N,1]

        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        # when inverse depth, Depth range is [1, 0] but we actually take the inverse, 
        # so the depth samples are between 1 and a very large number. 
        return depth_samples  

    def sample_depth_from_pdf(self, opt: Dict[str, Any], weights: torch.Tensor, n_samples_coarse: int, 
                              n_samples_fine: int, depth_range: List[float], det: bool) -> torch.Tensor:
        """ 
        Args:
            weights [B, num_rays, N]: Weights assigned to each sampled color.
        """
        depth_min, depth_max = depth_range

        pdf = weights / (weights.sum(dim=-1, keepdims=True) + 1e-6)
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]

        # here add either uniform or random
        if det:
            # take uniform samples
            grid = torch.linspace(0,1,n_samples_fine+1,device=self.device) # [Nf+1]
        else:
            grid = torch.rand(n_samples_fine+1).to(self.device)
        
        # unif corresponds to interval mid points
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,n_samples_coarse+1,device=self.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=n_samples_coarse)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=n_samples_coarse)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        # print(depth_low, depth_high)
        return depth_samples[...,None] # [B,HW,Nf,1]


    #------------ SPECIFIC RENDERING where each ray is rendered up to a max depth (different for each ray) -------
    def render_up_to_maxdepth_at_specific_pose_and_rays\
        (self, opt: Dict[str, Any], data_dict: Dict[str, Any], pose: torch.Tensor, intr: torch.Tensor, 
         H: int, W: int, depth_max: torch.Tensor, iter: int, pixels: torch.Tensor = None, 
         ray_idx: torch.Tensor = None, mode: str = 'train', per_slice: str = False) -> Dict[str, Any]:
        """Rendering of a specified set of pixels at a predefined pose, up to a specific depth. 

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
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            depth_max: Max depth to sample for each ray (L, N)
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations. 
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
            per_slice: render per slice? in case the number of pixels is too large to be processed all at once. 
        """
        if len(pose.shape) == 2:
            pose = pose.unsqueeze(0)
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0)

        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = data_dict.depth_range[0]

        # max_depth should be [B, N]
        ret = self.render_by_slices_w_custom_rays(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                                                  H=H, W=W, depth_range=depth_range, depth_max=depth_max, iter=iter) \
                if per_slice else \
                self.render_to_max(opt,pose,intr=intr,pixels=pixels,ray_idx=ray_idx,mode=mode, 
                                 H=H, W=W, depth_min=depth_range[0], depth_max=depth_max, iter=iter) # [B,N,3],[B,N,1]   
        ret.ray_idx = ray_idx  
        # ret.update(rgb_fine=rgb_fine,depth_fine=depth_fine,opacity_fine=opacity_fine) # [1, N_rays, K] or [N, N_rays, K]
        return ret

    def render_to_max(self, opt: Dict[str, Any], pose: torch.Tensor, H: int, W: int, intr: torch.Tensor, 
                      pixels: torch.Tensor = None, ray_idx: torch.Tensor = None, 
                      depth_max: torch.Tensor = None, depth_min: float = None, 
                      iter: int = None, mode: str = None) -> Dict[str, Any]:
        """Rendering function for a specified set of pixels at a predefined pose, up to a specific depth. 

        Args:
            opt (edict): settings
            pose: w2c poses at which to render (L, 3, 4) or (3, 4)
            intr: corresponding intr matrices (L, 3, 3) or (3, 3)
            H, W: size of rendered image
            depth_max: A different depth max for each pixels, (L, N) or (N)
            depth_min: Min depth 
            iter (int): iteration
            pixels, ray_idx: if any of the two is specified, will render only at these locations. 
                             (L, N, 2) or (N, 2)  / (L, N) or (N)
            mode (str, optional): Defaults to None.
        """
        batch_size = len(pose)
        
        if pixels is not None:
            center, ray = camera.get_center_and_ray_at_pixels(pose, pixels, intr=intr)
            # [B, N_rays, 3] and [B, N_rays, 3]
        else:
            # from ray_idx
            center,ray = camera.get_center_and_ray(pose, H, W, intr=intr) # [B,HW,3]
            while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
                center,ray = camera.get_center_and_ray(pose, H, W,intr=intr) # [B,HW,3]
        
            if ray_idx is not None:
                if len(ray_idx.shape) == 2 and ray_idx.shape[0] == batch_size:
                    n_rays = ray_idx.shape[-1]
                    #ray_idx is B, N-rays, ie different ray indices for each image in batch
                    batch_idx_flattened = torch.arange(start=0, end=batch_size).unsqueeze(-1).repeat(1, n_rays).long().view(-1)
                    ray_idx_flattened = ray_idx.reshape(-1).long()

                    # consider only subset of rays
                    center,ray = center[batch_idx_flattened, ray_idx_flattened],ray[batch_idx_flattened, ray_idx_flattened]  # [B*N_rays, 3]
                    center = center.reshape(batch_size, n_rays, 3)
                    ray = ray.reshape(batch_size, n_rays, 3)
                else:
                    # consider only subset of rays
                    # ray_idx is (-1)
                    center,ray = center[:,ray_idx],ray[:,ray_idx]  # [B, N_rays, 3]

        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)
        # render with main MLP

        pred = edict(origins=center, viewdirs=ray)

        # can only be in metric depth 
        depth_samples = self.sample_depth_diff_max_range_per_ray(opt,batch_size,num_rays=ray.shape[1], 
                                                                 n_samples=opt.nerf.sample_intvs, H=H, W=W, 
                                                                 depth_max=depth_max, depth_min=depth_min, 
                                                                 mode=mode) # [B,HW,N,1] or [B, N_rays, N, 1]

        pred_coarse = self.nerf.forward_samples(opt,center,ray,depth_samples,
                                                embedder_pts=self.embedder_pts, 
                                                embedder_view=self.embedder_view, mode=mode)
        pred_coarse['t'] = depth_samples
        # contains rgb_samples [B, N_rays, N, 3] and density_samples [B, N_rays, N]
        pred_coarse = self.nerf.composite(opt,ray,pred_coarse,depth_samples)
        # new elements are rgb,depth,depth_var, opacity,weights,loss_weight_sparsity 
        # [B,HW,K] or [B, N_rays, K], loss_weight_sparsity is [B]
        weights = pred_coarse['weights']

        pred.update(pred_coarse)

        compute_fine_sampling = True
        # render with fine MLP from coarse MLP
        if hasattr(opt.nerf, 'ratio_start_fine_sampling_at_x') and opt.nerf.ratio_start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.max_iter * opt.nerf.ratio_start_fine_sampling_at_x:
            compute_fine_sampling = False
        elif hasattr(opt.nerf, 'start_fine_sampling_at_x') and opt.nerf.start_fine_sampling_at_x is not None \
            and iter is not None and iter < opt.nerf.start_fine_sampling_at_x:
            compute_fine_sampling = False

        if opt.nerf.fine_sampling and compute_fine_sampling:
            # we use the same samples to compute opacity
            pred_fine = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,
                                                       embedder_pts=self.embedder_pts, 
                                                       embedder_view=self.embedder_view, mode=mode)
            
            pred_fine['t'] = depth_samples
            pred_fine = self.nerf_fine.composite(opt,ray,pred_fine,depth_samples)
            pred_fine = edict({k+'_fine': v for k, v in pred_fine.items()})
            pred.update(pred_fine)
        return pred

    def sample_depth_diff_max_range_per_ray(self, opt: Dict[str, Any], batch_size: int, n_samples: int, H: int, W: int, 
                                            depth_min: float, depth_max: float, 
                                            num_rays: int = None, mode: str = None) -> torch.Tensor:
        """Sample depths along ray. A different depth max is applied for each ray. 

        Args:
            opt (edict): settings
            batch_size (int): 
            n_samples (int): Number of depth samples along each ray
            H (int): img size
            W (int): img size
            depth_min (float)
            depth_max (torch.Tensor): Max depth to sample for each ray, (batch_size, num_rays)
            num_rays (int, optional): Defaults to None.
            mode (str, optional): Defaults to None.

        Returns:
            depth_samples (torch.Tesnor), shape # [B,num_rays,n_samples,1]
        """
        num_rays = num_rays or H*W
        
        rand_samples = torch.ones(batch_size,num_rays,n_samples,1,device=self.device)

        rand_samples += torch.arange(n_samples,device=self.device)[None,None,:,None].float() # the added part is [1, 1, N, 1] ==> [B,HW,N,1]

        # depth_max is (B, HW)
        depth_samples = rand_samples/n_samples*(depth_max.unsqueeze(-1).unsqueeze(-1)-depth_min)+depth_min # [B,HW,N,1]

        # can only be in metric here!
        return depth_samples  

