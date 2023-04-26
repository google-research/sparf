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


import os
from itertools import permutations
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent / '../../third_party/DenseMatching'))

from source.utils.vis_rendering import make_matching_plot_fast
from third_party.DenseMatching.utils_flow.pixel_wise_mapping import warp
from third_party.DenseMatching.utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from third_party.DenseMatching.models.PDCNet.base_pdcnet import estimate_probability_of_confidence_interval_of_mixture_density, estimate_average_variance_of_mixture_density
from third_party.DenseMatching.utils_flow.pixel_wise_mapping import remap_using_correspondence_map



class FlowSelectionWrapper(nn.Module):
    """
    Wrapper for flow networks to compute flows/correspondence maps relating input image pairs. 
    It was designed for a few input views only. Including many might lead to out-of-memory issues
    """

    def __init__(self, ckpt_path, num_views, backbone='PDCNet', batch_size=5):
        """_summary_

        Args:
            ckpt_path (str): path to the checkpoint for the correspondence network
            num_views (int): number of training views
            backbone (str, optional): Defaults to 'PDCNet'.
            batch_size (int, optional): For computing the correspondence in batch. Defaults to 5.
        """
        super().__init__()
    
        self.backbone = backbone
        self.confidence_map_type = 'p_r'
        self.load_flow_network(backbone, ckpt_path=ckpt_path)

        self.batch_size = batch_size
        self.num_views = num_views
        self.combi_list = get_combi_list(
            num_views,
            method='all') 
        # shape is 2x(num_views*(num_views - 1))
        # 2x(num_views*(num_views - 1)). [[0, 0, 0, ... 1, 1, 1], [1, 2, 3, ..9, 0, 2, 3, 4, 5, ]]
        # all combinations except for oneself

    def load_flow_network(self, backbone, ckpt_path=None):
        self.flow_net = flow_net_model_select(backbone)

        # load checkpoint
        if ckpt_path is not None:
            self.flow_net = self.load_network(backbone, ckpt_path)

        # fix_flow_weights
        self.flow_net.requires_grad_ = False
        if self.backbone != 'SPSG':
            for param in self.flow_net.parameters():
                param.requires_grad = False
        
        # always fix the flow net batch norm to eval!
        self.flow_net.eval()
        return 

    def load_network(self, backbone, checkpoint_path):
        """
        Loads a network checkpoint file.
        """

        if not os.path.isfile(checkpoint_path):
            raise ValueError('The checkpoint that you chose does not exist, {}'
                    .format(checkpoint_path))

        # Load checkpoint
        if hasattr(self.flow_net, 'load_weights'):
            self.flow_net.load_weights(checkpoint_path)
        else:
            # Load checkpoint
            print(f'Loading flow checkpoint from {checkpoint_path}')
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
            # TODO better handling
            if 'state_dict' in checkpoint_dict.keys():
                self.flow_net.load_state_dict(checkpoint_dict['state_dict'])
            else:
                self.flow_net.load_state_dict(checkpoint_dict)

        return self.flow_net


    def compute_flow_and_confidence_map_of_combi_list(self, images, combi_list_tar_src, plot=False, 
                                                      use_homography=False):
        '''Computing flow and confidence map of set of images given combi_list.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''
        if self.backbone == 'SPSG':
            return self.compute_matches_spsg(images, combi_list_tar_src, plot=plot)
        else:
            return self.compute_matches_pdcnet(images, combi_list_tar_src, plot, use_homography)

    def compute_flow_and_confidence_map_and_cc_of_combi_list(self, images, combi_list_tar_src, plot=False, 
                                                             use_homography=False):
        '''Computing flow and confidence map of set of images given combi_list. Apply cyclic consistency 
        as an additional filtering mechanism for the matches. 
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''
        if self.backbone == 'SPSG':
            return self.compute_matches_spsg(images, combi_list_tar_src, plot=plot, return_dummy_cc_map=True)
        else:
            return self.compute_matches_pdcnet_with_cc(images, combi_list_tar_src, plot, use_homography)


    # ---------------------------- SPSG matches --------------------------------
    def compute_matches_spsg(self, images, combi_list_tar_src, plot=False, return_dummy_cc_map=False):
        '''Computing flow and confidence map of set of images given combi_list, using SuperPoint and SuperGlue.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                correspondence_map (torch.Tensor): correspondence map from target to source, 
                                                   shape [len(combi_list), 2/1, H, W]
                conf_map (torch.Tensor): correspondence map from target to source, 
                                         shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''
        plot_ = plot and combi_list_tar_src.shape[1] < 100

        B, _, H, W = images.shape


        images_proc = self.flow_net.pre_process_img(images * 255)
        batch_size = 50

        # extract keypoints for all images
        kp_dict =  {}  
        # kp_dict will contain 'keypoint', 'scores'
        # in each, there is a list of lists, i.e. an element per image
        for idx_start in range(0, images.shape[0], batch_size):
            if idx_start == images.shape[0] - 1:
                kp_template_dict_ = self.flow_net.get_keypoints\
                    (images_proc[idx_start].unsqueeze(0))
            else:
                kp_template_dict_ = self.flow_net.get_keypoints\
                    (images_proc[idx_start:idx_start+batch_size])

            for k, v in kp_template_dict_.items():
                if k in kp_dict.keys():
                    kp_dict[k].extend(kp_template_dict_[k])
                else:
                    kp_dict[k] = kp_template_dict_[k]
            torch.cuda.empty_cache()

        correspondence_map = torch.zeros(combi_list_tar_src.shape[1], H, W, 2).to(images.device)
        conf_map = torch.zeros(combi_list_tar_src.shape[1], H, W, 1).to(images.device)

        # establish matches
        plot_list = [] if plot_ else None
        for idx in range(combi_list_tar_src.shape[1]):  # (2xN)
            id_target, id_source = combi_list_tar_src[:, idx] 
            source_kp_dict = {k: [v[id_source]] for k, v in kp_dict.items()}
            target_kp_dict = {k: [v[id_target]] for k, v in kp_dict.items()}
            pred = self.flow_net.get_matches_and_confidence(source_img=images_proc[id_source].unsqueeze(0), 
                                                            target_img=images_proc[id_target].unsqueeze(0), 
                                                            source_kp_dict=source_kp_dict, 
                                                            target_kp_dict=target_kp_dict, 
                                                            preprocess_image=False)
            # 'kp_source': mkpts0, 'kp_target': mkpts1, 'confidence_value': mconf
            pred_kp_target = torch.from_numpy(pred['kp_target']).to(images.device)  # Nx2
            diff = torch.round(pred_kp_target) - pred_kp_target
            pred_kp_target = torch.round(pred_kp_target).long()
            pred_kp_source = torch.from_numpy(pred['kp_source']).to(images.device) + diff  # Nx2
            
            if plot_:
                plot_list.append(make_matching_plot_fast(
                    image1=(images[id_source].permute(1, 2, 0).cpu().detach().numpy() * 255), 
                    image0=(images[id_target].permute(1, 2, 0).cpu().detach().numpy() * 255), 
                    kpts1=pred['kp_source'][:500], kpts0=pred['kp_target'][:500], 
                    text=['{} to {}'.format(id_source, id_target), 
                          '{} matches'.format(pred_kp_source.shape[0]), 'Top 500 matches']))
            pred_conf = torch.from_numpy(pred['confidence_value']).to(images.device)
            
            correspondence_map[idx, pred_kp_target[:, 1], pred_kp_target[:, 0]] = pred_kp_source
            conf_map[idx, pred_kp_target[:, 1], pred_kp_target[:, 0]] = pred_conf.reshape(-1, 1)
        
        ret = [correspondence_map.permute(0, 3, 1, 2), conf_map.permute(0, 3, 1, 2)]
        if return_dummy_cc_map:
            ret += [torch.ones_like(conf_map.permute(0, 3, 1, 2))]

        if plot:
            if plot_list is not None:
                plot_list = np.concatenate(plot_list, axis=0)
                plot_list = torch.from_numpy(plot_list.astype(np.float32)).permute(2, 0, 1)
            ret += [plot_list]
        return ret

    # --------------------- main function to compute pdcnet matches ------------------------
    def compute_matches_pdcnet(self, images, combi_list_tar_src, plot=False, 
                               use_homography=False):
        '''Computing flow and confidence map of set of images given combi_list, using PDC-Net.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                mapping_self_to_neighbor (torch.Tensor): correspondence map from target to source, 
                                                          shape [len(combi_list), 2/1, H, W]
                confidence_map (torch.Tensor): correspondence map from target to source, 
                                               shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''

        n_views, _, H, W = images.shape
        img_size = (H, W)

        # set deterministic combination if not chosen online
        if combi_list_tar_src is None:
            combi_list_tar_src = self.combi_list

        images = images * 255.

        extract_features = False
        if use_homography:
            flow_data = images
        else:
            flow_data = self.process_data_for_flow_net(images, extract_features=extract_features)

        # can query projection points in self
        return_confidence_map = self.confidence_map_type != 'cyclic_consistency_error'
        flow_self_to_neighbor, confidence_map = \
            self.compute_flow_combinations(flow_data, torch.flip(combi_list_tar_src, [0]),  # the first element will now be the source
                                           img_size, return_confidence_map=return_confidence_map, 
                                           use_homography=use_homography)  # (B, 2, H, W)
        confidence_map = confidence_map # (B, 1, H, W)

        if self.confidence_map_type == 'cyclic_consistency_error': 
            # can query projection points in neighbor views
            flow_neighbor_to_self, conf_map_neighbor_to_self = \
                self.compute_flow_combinations(flow_data, combi_list_tar_src, img_size, use_homography=use_homography)

            # we want consistency error in self coordinates
            cyclic_consistency_error = flow_self_to_neighbor + warp(flow_neighbor_to_self, flow_self_to_neighbor)
            cyclic_consistency_error = torch.norm(cyclic_consistency_error, dim=1, keepdim=True)  # (B, 1, H, W)
            confidence_map = 1.0 / (1.0 + cyclic_consistency_error) 
        else:
            if confidence_map is None:
                raise ValueError

        ret = []
        # return_correspondence_map:
        mapping_self_to_neighbor = convert_flow_to_mapping(flow_self_to_neighbor, output_channel_first=True)  # (B, 2, H, W)
        ret += [mapping_self_to_neighbor, confidence_map] # (B, 2, H, W) and (B, 1, H, W)

        if plot:
            flow_plot = None
            if confidence_map.shape[0] < 1000:
                flow_plot = self.visualize_mapping_combinations(images / 255., mapping_self_to_neighbor, 
                                                                confidence_map, combi_list_tar_src, save_path=None)
                flow_plot = torch.from_numpy( flow_plot.astype(np.float32)/255.).permute(2, 0, 1)
            
            ret += [flow_plot]
        return ret # combi_list.shape[1], 3, H, W


    def compute_matches_pdcnet_with_cc(self, images, combi_list, plot=False, use_homography=False):
        '''Computing flow and confidence map of set of images using given combi_list.
        Args:
            images: channel first normalized tensor of images [B, C, H, W]
            combi_list: torch tensor index pairs of images that need to be computed, 2xN, the first one is the target
        Returns:
            List containing:
                mapping_self_to_neighbor (torch.Tensor): correspondence map from target to source, 
                                                          shape [len(combi_list), 2/1, H, W]
                confidence_map (torch.Tensor): correspondence map from target to source, 
                                               shape [len(combi_list), 2/1, H, W]
                flow_plot (torch.Tensor): image plot
        '''

        n_views, _, H, W = images.shape
        img_size = (H, W)

        # set deterministic combination if not chosen online
        if combi_list is None:
            combi_list = self.combi_list

        images = images * 255.

        extract_features = False
        if use_homography:
            flow_data = images
        else:
            flow_data = self.process_data_for_flow_net(images, extract_features=extract_features)

        # can query projection points in self
        return_confidence_map = True
        flow_self_to_neighbor, confidence_map = self.compute_flow_combinations\
            (flow_data, torch.flip(combi_list, [0]),  # the first element will now be the target
             img_size, return_confidence_map=return_confidence_map, use_homography=use_homography)  # (B, 2, H, W)
        confidence_map = confidence_map  # (B, 1, H, W)

        flow_neighbor_to_self, conf_map_neighbor_to_self = self.compute_flow_combinations\
            (flow_data, combi_list, img_size, use_homography=use_homography)

        gc.collect()
        with torch.no_grad():
            # we want consistency error in self coordinates
            if flow_neighbor_to_self.shape[0] > 500:
                inter = 100
                cyclic_consistency_error = []
                for i_start in range(0, flow_neighbor_to_self.shape[0], inter):
                    cyclic_consistency_error_ = flow_self_to_neighbor[i_start: i_start+inter] + \
                        warp(flow_neighbor_to_self[i_start: i_start+inter], flow_self_to_neighbor[i_start: i_start+inter])
                    cyclic_consistency_error.append(cyclic_consistency_error_.cpu())
                    torch.cuda.empty_cache()

                cyclic_consistency_error = torch.cat(cyclic_consistency_error, dim=0) if len(cyclic_consistency_error) > 1\
                    else cyclic_consistency_error[0]
                cyclic_consistency_error = cyclic_consistency_error.to(confidence_map.device)
            else:
                cyclic_consistency_error = flow_self_to_neighbor + warp(flow_neighbor_to_self, flow_self_to_neighbor)
            cyclic_consistency_error = torch.norm(cyclic_consistency_error, dim=1, keepdim=True)  # (B, 1, H, W)
            confidence_map_from_cc = 1.0 / (1.0 + cyclic_consistency_error) 

        ret = []
        # return_correspondence_map:
        mapping_self_to_neighbor = convert_flow_to_mapping(flow_self_to_neighbor, output_channel_first=True)  # (B, 2, H, W)
        ret += [mapping_self_to_neighbor, confidence_map, confidence_map_from_cc]

        if plot:
            flow_plot = None
            if confidence_map.shape[0] < 1000:
                flow_plot = self.visualize_mapping_combinations(images / 255., mapping_self_to_neighbor, 
                                                                confidence_map, combi_list, save_path=None)
                flow_plot = torch.from_numpy( flow_plot.astype(np.float32)/255.).permute(2, 0, 1)
            
            ret += [flow_plot]
        # all the correspondences stuff are (B, 2/1, H, W)
        return ret  

    # ------------- functions for processing and computing matches for PDCNet --------------------
        # ------------------------------- PDCNet matches ------------------------------------
    @staticmethod
    @torch.no_grad()
    def pre_process_imgs(
            imgs,
            mean_vector=[0.485, 0.456, 0.406],
            std_vector=[0.229, 0.224, 0.225]):

        """
        Preprocesses image for PDC Net
        Args:
            imgs: unnormalized float imgs in B,C,H,W format in uint8
        Returns:
            imgs_ (torch.Tensor): [B, C, H, W] dividable by 8, normalized with imagenet 
            img_256 (torch.Tensor): [B, C, 256, 256], normalized with imagenet 
            scale_x (float): factor between original H and the one dividable by 8, for horizontal direction
            scale_y (float): factor between original H and the one dividable by 8, for horizontal direction
        """

        # might need to interpolate in case image size is not divisible by 8
        _, _, H, W = imgs.shape

        # img has shape Bx3xHxW
        H_int = int(math.floor(int(H / 8.0) * 8.0)) if H > 256 else 256
        W_int = int(math.floor(int(W / 8.0) * 8.0)) if W > 256 else 256

        # need to interpolate
        imgs_ = F.interpolate(input=imgs.float(),
                              size=(H_int, W_int),
                              mode='area').byte().float().div(255.)

        mean = torch.as_tensor(mean_vector, dtype=imgs_.dtype, device=imgs_.device)
        std = torch.as_tensor(std_vector, dtype=imgs_.dtype, device=imgs_.device)
        imgs_.sub_(mean[:, None, None]).div_(std[:, None, None])

        # resolution 256x256
        imgs_256 = F.interpolate(input=imgs.float(),
                                 size=(256, 256),
                                 mode='area').byte().float().div(255.)

        imgs_256.sub_(mean[:, None, None]).div_(std[:, None, None])

        scale_x = float(W) / float(W_int)  # will multiply the flow with this
        scale_y = float(H) / float(H_int)

        return imgs_, imgs_256, scale_x, scale_y

    
    def process_data_for_flow_net(self, imgs, extract_features=True):
        with torch.no_grad():
            imgs, imgs_256, scale_x, scale_y = self.pre_process_imgs(imgs)

        imgs_pyr, imgs_pyr_256 = None, None
        # extract features only once
        if extract_features:
            imgs_pyr, imgs_pyr_256 = [], []
            inter = 500
            for i_start in range(0, imgs.shape[0], inter):
                imgs_pyr_, imgs_pyr_256_ = self.flow_net.extract_pyramid(imgs[i_start:i_start+inter], 
                                                                        imgs_256[i_start:i_start+inter])
                imgs_pyr.append(imgs_pyr_)
                imgs_pyr_256.append(imgs_pyr_256_)
                torch.cuda.empty_cache()

            imgs_pyr = torch.cat(imgs_pyr, dim=0) if len(imgs_pyr) > 1 else imgs_pyr[0]
            imgs_pyr_256 = torch.cat(imgs_pyr_256, dim=0) if len(imgs_pyr_256)> 1 else imgs_pyr_256[0]


        return (imgs, imgs_256, imgs_pyr, imgs_pyr_256, scale_x, scale_y)

    def compute_flow_combinations(self, flow_data, combi_list, output_shape, return_confidence_map=False, use_homography=False):
        '''Compute flow for combination specified in combi_list in batched

        Args:
            flow_data: output after preporcessing images for PDCNet
            combi_list: 2D index tensor of flow to be computed
                        batched_combi_list shape [2, self.batch_size], first element is source, second is target
            output_shape: original img_shape
        '''

        def output_to_flow_and_uncertainty(output):
            # for pdcnet
            p_r = None
            flow_est_list = output['flow_estimates']
            flow_est = flow_est_list[-1]
            if 'uncertainty_estimates' in output.keys():
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the confidence value
                log_var_map = uncertainty_list[0]
                weight_map = uncertainty_list[1]
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.) / 0.5730
            return flow_est, p_r


        # batch combilist

        batched_flow, batched_conf_map = [], []

        if use_homography:
            imgs = flow_data   # (0, 255) 
            for idx in range(combi_list.shape[1]):
                # batched_combi_list shape [2, self.batch_size], first element is source, second is target
                src_imgs = imgs[combi_list[0, idx], ...].unsqueeze(0)
                tgt_imgs = imgs[combi_list[1, idx], ...].unsqueeze(0)

                estimated_flow, uncertainty_dict = self.flow_net.estimate_flow_and_confidence_map_with_homo\
                    (src_imgs, tgt_imgs, inference_parameters=self.flow_net.inference_parameters, 
                    scaling=1.0/4., mode='channel_first')

                batched_flow.append(estimated_flow)
                if return_confidence_map:
                    conf_map = uncertainty_dict['p_r']  / 0.5730
                    batched_conf_map.append(conf_map)

                torch.cuda.empty_cache() # otherwise GPU memory filles up quickly

            flow_est = torch.cat(batched_flow, dim=0)
            if return_confidence_map:
                batched_conf_map = torch.cat(batched_conf_map, dim=0)
            else:
                batched_conf_map = None
        else:
            batched_combi_list = torch.split(combi_list, self.batch_size, dim=1)  # list of elements of size [2, self.batch_size]
            imgs, imgs_256, imgs_pyr, imgs_pyr_256, scale_x, scale_y = flow_data

            for idx in batched_combi_list:
                # batched_combi_list shape [2, self.batch_size], first element is source, second is target
                src_imgs = imgs[idx[0], ...]
                tgt_imgs = imgs[idx[1], ...]

                src_imgs_256 = imgs_256[idx[0], ...]
                tgt_imgs_256 = imgs_256[idx[1], ...]

                if imgs_pyr is not None:
                    src_imgs_pyr = [pyr[idx[0], ...] for pyr in imgs_pyr]
                    tgt_imgs_pyr = [pyr[idx[1], ...] for pyr in imgs_pyr]

                    src_imgs_pyr_256 = [pyr[idx[0], ...] for pyr in imgs_pyr_256]
                    tgt_imgs_pyr_256 = [pyr[idx[1], ...] for pyr in imgs_pyr_256]

                    # batch process this
                    _ , output = self.flow_net.forward(
                        tgt_imgs, src_imgs,
                        tgt_imgs_256, src_imgs_256,
                        im_target_pyr=tgt_imgs_pyr,
                        im_source_pyr=src_imgs_pyr,
                        im_target_pyr_256=tgt_imgs_pyr_256,
                        im_source_pyr_256=src_imgs_pyr_256
                    )
                else:
                    _ , output = self.flow_net.forward(
                        tgt_imgs, src_imgs,
                        tgt_imgs_256, src_imgs_256
                    )


                flow_est, conf_map = output_to_flow_and_uncertainty(output)
                flow_est = F.interpolate(input=flow_est,
                                        size=output_shape,
                                        mode='bilinear',
                                        align_corners=False)
                batched_flow.append(flow_est)
                if conf_map is not None and return_confidence_map:
                    conf_map = F.interpolate(input=conf_map, size=output_shape, mode='bilinear',
                                             align_corners=False)
                    batched_conf_map.append(conf_map)

                torch.cuda.empty_cache() # otherwise GPU memory filles up quickly


            flow_est = torch.cat(batched_flow, dim=0)
            flow_est[:, 0, :, :] *= scale_x
            flow_est[:, 1, :, :] *= scale_y
            if len(batched_conf_map) > 0 and return_confidence_map:
                batched_conf_map = torch.cat(batched_conf_map, dim=0)
            else:
                batched_conf_map = None

        return flow_est, batched_conf_map

    @torch.no_grad()
    def visualize_mapping_combinations(self, images, mapping_est, batched_conf_map, combi_list, save_path):
        return visualize_mapping_combinations(images, mapping_est, batched_conf_map, combi_list, save_path)
    
    # ------------- on image pair -----------------------------
    def pair_flow_forward(self, src_img, target_img, return_correspondence_map=False):
        '''
        for an image pair only, computes the flow field relating the target to the source image. 
        src_img: BxCxHxW normalized to [0, 1]
        params:target_img:'
        '''
        H, W = target_img.shape[-2:]
        img_size = (H, W)

        flow_data = self.process_data_for_flow_net(src_img  * 255. )
        flow_data_tgt = self.process_data_for_flow_net(target_img  * 255.)

        src_imgs, src_imgs_256, src_imgs_pyr, src_imgs_pyr_256, scale_x, scale_y = flow_data
        tgt_imgs, tgt_imgs_256, tgt_imgs_pyr, tgt_imgs_pyr_256, scale_x, scale_y = flow_data_tgt

        def output_to_flow_and_uncertainty(output):
            # for pdcnet
            p_r = None
            flow_est_list = output['flow_estimates']
            flow_est = flow_est_list[-1]
            if 'uncertainty_estimates' in output.keys():
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the confidence value
                log_var_map = uncertainty_list[0]
                weight_map = uncertainty_list[1]
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.) / 0.5730
            return flow_est, p_r

        _ , output = self.flow_net.forward(
            tgt_imgs, src_imgs,
            tgt_imgs_256, src_imgs_256,
            im_target_pyr=tgt_imgs_pyr,
            im_source_pyr=src_imgs_pyr,
            im_target_pyr_256=tgt_imgs_pyr_256,
            im_source_pyr_256=src_imgs_pyr_256
        )

        flow_est, p_r = output_to_flow_and_uncertainty(output)

        flow_est = F.interpolate(input=flow_est,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)

        flow_est[:, 0, :, :] *= scale_x
        flow_est[:, 1, :, :] *= scale_y
        if p_r is not None:
            p_r = F.interpolate(input=p_r,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)
        if return_correspondence_map:
            mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)  # (B, 2, H, W)
            return mapping, p_r
        return flow_est, p_r

    def pair_flow_forward_w_uncertainty(self, src_img, target_img, return_correspondence_map=False, return_conf_from_cc=False):
        '''
        for an image pair only, computes the flow field relating the target to the source image. 
        src_img: BxCxHxW normalized to [0, 1]
        params:target_img:'
        '''
        H, W = target_img.shape[-2:]
        img_size = (H, W)

        flow_data = self.process_data_for_flow_net(src_img  * 255. )
        flow_data_tgt = self.process_data_for_flow_net(target_img  * 255.)

        src_imgs, src_imgs_256, src_imgs_pyr, src_imgs_pyr_256, scale_x, scale_y = flow_data
        tgt_imgs, tgt_imgs_256, tgt_imgs_pyr, tgt_imgs_pyr_256, scale_x, scale_y = flow_data_tgt

        def output_to_flow_and_uncertainty(output):
            # for pdcnet
            p_r = None
            flow_est_list = output['flow_estimates']
            flow_est = flow_est_list[-1]
            if 'uncertainty_estimates' in output.keys():
                uncertainty_list = output['uncertainty_estimates'][-1]  # contains log_var_map and weight_map

                # get the confidence value
                log_var_map = uncertainty_list[0]
                weight_map = uncertainty_list[1]
                p_r = estimate_probability_of_confidence_interval_of_mixture_density(weight_map, log_var_map, R=1.) / 0.5730
                var = estimate_average_variance_of_mixture_density(weight_map, log_var_map)
            return flow_est, p_r, var

        _ , output = self.flow_net.forward(
            tgt_imgs, src_imgs,
            tgt_imgs_256, src_imgs_256,
            im_target_pyr=tgt_imgs_pyr,
            im_source_pyr=src_imgs_pyr,
            im_target_pyr_256=tgt_imgs_pyr_256,
            im_source_pyr_256=src_imgs_pyr_256
        )

        flow_est, p_r, var = output_to_flow_and_uncertainty(output)

        flow_est = F.interpolate(input=flow_est,
                                size=img_size,
                                mode='bilinear',
                                align_corners=False)

        flow_est[:, 0, :, :] *= scale_x
        flow_est[:, 1, :, :] *= scale_y
        p_r = F.interpolate(input=p_r,size=img_size,mode='bilinear', align_corners=False)
        var = F.interpolate(input=var,size=img_size,mode='bilinear', align_corners=False)

        ret = []
        if return_correspondence_map:
            mapping = convert_flow_to_mapping(flow_est, output_channel_first=True)  # (B, 2, H, W)
            ret += [mapping, p_r, var]
        else:
            ret += [flow_est, p_r, var]

        if return_conf_from_cc:
            _ , output_src_to_tar = self.flow_net.forward(
                src_imgs, tgt_imgs, 
                src_imgs_256, tgt_imgs_256, 
                im_target_pyr=src_imgs_pyr,
                im_source_pyr=tgt_imgs_pyr,
                im_target_pyr_256=src_imgs_pyr_256,
                im_source_pyr_256=tgt_imgs_pyr_256
            )
            flow_est_src_to_tar, p_r_src_to_tar, var_src_to_tar = output_to_flow_and_uncertainty(output_src_to_tar)

            flow_est_src_to_tar = F.interpolate(input=flow_est_src_to_tar,
                                    size=img_size,
                                    mode='bilinear',
                                    align_corners=False)

            flow_est_src_to_tar[:, 0, :, :] *= scale_x
            flow_est_src_to_tar[:, 1, :, :] *= scale_y

            consistency_error = flow_est + warp(flow_est_src_to_tar, flow_est)  #flow_neighbor_to_self was created by just exchanging the source and the target basically 
            conf_from_consistency_error = 1. / (1 + torch.norm(consistency_error, dim=1, keepdim=True))
            ret += [conf_from_consistency_error]
        return ret

    def switch_to_train(self):
        self.flow_net.train()

    def switch_to_eval(self):
        self.flow_net.eval()


def get_combi_list(num_views, method='all') -> torch.tensor:
    """Compute list of image pairs. 
    Args:
        num_views int): number of total views
        method (str, optional): _description_. Defaults to 'random'.

    Returns:
        torch.tensor: list of image pair indexes, in format (2, N)
    """
    if method == 'all':
        combi_list = permutations(range(num_views), 2)
        # if num_views=10, 
        # [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), 
        # (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), 
        # (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),...]
        # choose num_neighbors ones
        combi_list = np.array(list(combi_list)).astype(np.int).T
        # 2x(num_views*(num_views - 1)). [[0, 0, 0, ... 1, 1, 1], [1, 2, 3, ..9, 0, 2, 3, 4, 5, ]]
        # all combinations except for oneself
        assert combi_list.shape[-1] == num_views * (num_views - 1)
    elif method == 'random':
        # choose for each 1 view
        combi_list = np.stack((np.arange(num_views), np.random.permutation(num_views))).astype(np.int)  # 2x10
    else:
        raise

    return torch.from_numpy(combi_list)


def flow_net_model_select(backbone, train_features=False):
    if backbone == 'PDCNet':
        global_optim_iter=3
        local_optim_iter=3
        from third_party.DenseMatching.models.PDCNet.PDCNet import PDCNet_vgg16

        global_gocor_arguments = {'optim_iter': global_optim_iter, 'steplength_reg': 0.1,
                                    'train_label_map': False, 'apply_query_loss': True,
                                    'reg_kernel_size': 3, 'reg_inter_dim': 16,
                                    'reg_output_dim': 16}

        # for global gocor, we apply L_r only
        local_gocor_arguments = {'optim_iter': local_optim_iter, 'steplength_reg': 0.1}

        flow_net = PDCNet_vgg16(
            global_corr_type='GlobalGOCor', global_gocor_arguments=global_gocor_arguments,
            normalize='leakyrelu', same_local_corr_at_all_levels=True,
            local_corr_type='LocalGOCor', local_gocor_arguments=local_gocor_arguments,
            local_decoder_type='OpticalFlowEstimatorResidualConnection',
            global_decoder_type='CMDTopResidualConnection',
            corr_for_corr_uncertainty_decoder='corr', train_features=train_features, 
            give_layer_before_flow_to_uncertainty_decoder=True,
            var_2_plus=520 ** 2, var_2_plus_256=256 ** 2, var_1_minus_plus=1.0, var_2_minus=2.0)

    elif backbone == 'SPSG':
        from source.utils.spsg_matcher.superglue_module import SPSGInference
        flow_net = SPSGInference()
        compute_flow = True
    return flow_net


def visualize_mapping_combinations(images, mapping_est, batched_conf_map, combi_list, save_path=None, min_conf=0.8):
    # flow_est [N, 2, H, W] where N is combi_list.shape[1]
    mapping_est = mapping_est.detach()
    batched_conf_map = batched_conf_map.detach()
    H, W = mapping_est.shape[-2:]
    mapping_est = mapping_est.cpu().numpy()
    # images [n_views, 3, H, W]
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    batched_conf_map = batched_conf_map.squeeze(1).cpu().numpy()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import math

    n_flows = combi_list.shape[1]
    def plot_mapping_set(n_start, n_end):
        w = 10
        n_flows = n_end - n_start
        h = w * math.ceil(n_flows / 4.)
        fig = plt.figure(figsize=(w, h), tight_layout=True)
        spec2 = gridspec.GridSpec(ncols=4, nrows=n_flows, figure=fig)
        for ind, i in enumerate(range(n_start, n_end)):
            i_self, i_other_img = combi_list[:, i]
            row_nu = ind
            image_target = images[i_self]
            image_source = images[i_other_img]
            warped = remap_using_correspondence_map(image_source, mapping_est[i, 0], mapping_est[i, 1])

            plt.subplot(spec2[row_nu, 0])
            plt.imshow(image_source)
            plt.title(f'Source Image {i_other_img}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 1])
            plt.imshow(image_target)
            plt.title(f'Target Image, {i_self}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 2])
            plt.imshow(warped)
            plt.title(f'Warped source {i_other_img} to  {i_self}')
            plt.axis("off")

            plt.subplot(spec2[row_nu, 3])
            plt.imshow(batched_conf_map[i])
            plt.title(f'conf map {i_other_img} to  {i_self}, {(batched_conf_map[i] > min_conf).sum()} conf px')
            plt.axis("off")

        fig.tight_layout(pad=0)
        canvas = FigureCanvas(fig)
        canvas.draw()      
        # draw the canvas, cache the renderer
        width, height = canvas.get_width_height() #fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close()
        return image 
    
    all_images = []
    for start_i in range(0, n_flows, 50):
        all_images.append(plot_mapping_set(start_i, min(n_flows, start_i+50)))

    def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    def stack_images_rows_with_pad(list_of_images):
        max_h = list_of_images[0].shape[0]
        return [pad_along_axis(x, max_h, axis=0) for x in list_of_images]

    if len(all_images) > 1:
        all_images = stack_images_rows_with_pad(all_images)
        image = np.concatenate(all_images, axis=1)
    else:
        image = all_images[0]
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('Written image to {}'.format(save_path))
    return image

    