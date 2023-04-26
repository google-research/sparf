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
import torchvision.transforms.functional as torchvision_F
import tqdm
import imageio
from easydict import EasyDict as edict
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.datasets.rendering_path import generate_spiral_path, generate_spiral_path_dtu
from source.utils.vis_rendering import *
import source.models.renderer as renderer
from source.training import base
from source.training.core.metrics import compute_metrics, compute_depth_error_on_rays, compute_mse_on_rays
from source.utils import camera
from source.models.flow_net import FlowSelectionWrapper
from source.utils.colmap_initialization.triangulation_w_known_poses \
    import compute_triangulation_pdcnet, compute_triangulation_sp_sg
from source.training.core.sampling_strategies import RaySamplingStrategy
from source.utils.torch import exponentiel_lr_starting_at_x



class NerfTrainerPerScene(base.PerSceneTrainer):
    """ Base class for training and evaluating a NeRF model, considering fixed ground-truth poses. """

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)
        if (self.settings.test_metrics_only or 
            self.settings.render_video_pose_only or 
            self.settings.render_video_only):
            self.init_for_eval(opt)
        else:
            self.init_for_training(opt)

    def init_for_training(self, opt: Dict[str, Any]):
        # define and load the datasets
        # this needs to be first for the joint pose-NeRF refinements, since there will be pose parameters 
        # for each of the training images. 
        self.load_dataset(opt)

        # define the model
        self.build_networks(opt) 

        # define optimizer and scheduler
        self.setup_optimizer(opt)

        self.define_loss_module(opt)
        return

    def init_for_eval(self, opt: Dict[str, Any]):
        # define and load the datasets
        # this needs to be first for joint_pose_nerf_training, since there will be pose parameters for eahc of the training images. 
        self.load_dataset(opt)

        # define the model
        opt.use_flow = False
        opt.load_colmap_depth = False
        self.build_networks(opt) # initialize self.net with the network

        # only needs photometric loss for test-time pose photometric optimization
        opt.loss_type = 'photometric' 
        self.define_loss_module(opt)
        return 

    def plot_training_set(self):
        value = self.train_data.all.image   # (B, 3, H, W)
        self.writer.add_images(f'train/training_images', value, 1, dataformats='NCHW')
        return

    def load_dataset(self,opt: Dict[str, Any],eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        
        # prefetch all training data
        self.train_data.prefetch_all_data()
        self.train_data.all = edict(self.to_cuda(self.train_data.all))  
        # contains the data corresponding to the entire scene
        # important keys are 'image', 'pose' (w2c), 'intr'
        self.plot_training_set()
        
        
        # pixels/ray sampler
        self.sampling_strategy = RaySamplingStrategy(opt, data_dict=self.train_data.all, device=self.device)


        self.logger.info('Depth type {}'.format(opt.nerf.depth.param))
        if opt.nerf.depth.param == 'inverse':
            depth_range = opt.nerf.depth.range
        else:
            # use the one from the dataset
            depth_range = self.train_data.all.depth_range[0]
        self.logger.info('depth range {} to {}'.format(depth_range[0], depth_range[1]))
        return

    def build_nerf_net(self, opt: Dict[str, Any]):
        self.net = renderer.Graph(opt, self.device)
        return
    
    def build_correspondence_net(self, opt: Dict[str, Any]):
        self.flow_net = FlowSelectionWrapper(ckpt_path=opt.flow_ckpt_path,
                                             backbone=opt.flow_backbone,
                                             batch_size=opt.flow_batch_size,
                                             num_views=len(self.train_data)).to(self.device)
        return
        
    def get_colmap_triangulation(self):
        """Triangulate 3D points using the ground-truth poses and correspondences extracted
        by multiple alternative matches. This is basically what is done in DS-NeRF to obtain the
        colmap depth maps. This is used for comparison but does not correspond to a realistic scenario, 
        because ground-truth poses are used. 
        """
        matcher = self.settings.matcher_for_colmap_triang
        directory = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        if matcher == 'sp_sg':
            colmap_depth_map, colmap_conf_map = compute_triangulation_sp_sg\
                (opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
        elif matcher == 'pdcnet':
            colmap_depth_map, colmap_conf_map = compute_triangulation_pdcnet\
                (opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
        else:
            raise ValueError
        
        # adding to the data_dict
        self.train_data.all.colmap_depth = colmap_depth_map.to(self.device) 
        self.train_data.all.colmap_conf = colmap_conf_map.to(self.device)         

        if 'depth_gt' in self.train_data.all.keys():
            # look at the average error in depth of colmap
            depth_gt = self.train_data.all.depth_gt
            valid_depth_gt = self.train_data.all.valid_depth_gt
            colmap_depth = self.train_data.all.colmap_depth
            valid_colmap_depth = colmap_depth.gt(1e-6)

            mask = valid_depth_gt & valid_colmap_depth
            depth_gt = depth_gt[mask]
            colmap_depth = colmap_depth[mask]
            error = torch.abs(depth_gt - colmap_depth).mean()
            self.logger.info('colmap depth error {}'.format(error))
            self.write_event('train', {'colmap_depth_err': error}, self.iteration)

        '''
        if self.settings.load_colmap_depth and self.settings.debug:
            for i in range(self.train_data.all.image.shape[0]):
                for j in range(i+1, self.train_data.all.image.shape[0]):
                    verify_colmap_depth(data_dict=self.train_data.all, poses_w2c=self.train_data.all.pose, 
                                        index_0=i, index_1=j)
        '''
        return 

    def build_networks(self, opt: Dict[str, Any]):
        self.logger.info("building networks...")
        self.build_nerf_net(opt)

        if opt.load_colmap_depth:
            self.get_colmap_triangulation()

        if opt.use_flow:
            self.build_correspondence_net(opt)
        return 

    def setup_optimizer(self,opt: Dict[str, Any]):
        self.logger.info("setting up optimizer for NeRF...")

        optim = torch.optim.Adam(params=self.net.nerf.parameters(), 
                                 lr=opt.optim.lr, betas=(0.9, 0.999))
        if opt.nerf.fine_sampling:
            optim.add_param_group(dict(params=self.net.nerf_fine.parameters(),
                                       lr=opt.optim.lr, betas=(0.9, 0.999))) 

        self.register_optimizer(optim)

        # set up scheduler
        if opt.optim.sched:
            self.logger.info('setting up scheduler for NeRF...')
            if opt.optim.start_decrease > 0:
                sched = exponentiel_lr_starting_at_x(opt, self.optimizer)
            else:
                # normal schedular, starting from the first iteration
                scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)
                if opt.optim.lr_end:
                    assert(opt.optim.sched.type=="ExponentialLR")
                    max_iter = opt.optim.max_iter if hasattr(opt.optim, 'max_iter') else opt.max_iter
                    opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./max_iter)
                    print(max_iter)
                kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
                sched = scheduler(optimizer=self.optimizer, **kwargs)
            self.register_scheduler(sched)
        return 
        
    def train_step(self, iteration: int, data_dict: Dict[str, Any]
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Forward pass of the training step. Loss computation
        Args:
            iteration (int): 
            data_dict (edict): Scene data. dict_keys(['idx', 'image', 'intr', 'pose'])
                                image: tensor of images with shape (B, 3, H, W)
                                intr: tensor of intrinsic matrices of shape (B, 3, 3)
                                pose: tensor of ground-truth w2c poses of shape (B, 3, 4)
                                idx: indices of images, shape is (B)

        Returns:
            output_dict: output from the renderer
            results_dict: dict to log, also contains the loss for back-propagation
            plotting_dict: dict with figures to log to tensorboard. 
        """

        plot = False
        if iteration % self.settings.vis_steps == 0:
            plot = True

        # data_dict also corresponds to the whole scene here 
        B = data_dict.image.shape[0]

        # sample rays from all images, and render depth and rgb from those viewpoint directions. 
        rays = self.sampling_strategy(self.settings.nerf.rand_rays,
                                      sample_in_center=iteration < self.settings.precrop_iters)
        output_dict = self.net.render_image_at_specific_rays(self.settings, data_dict, 
                                                             ray_idx=rays, iter=iteration, mode="train")
        
        # current estimate of the pose
        poses_w2c = self.net.get_w2c_pose(self.settings, data_dict, mode='train')  # is it world to camera
        data_dict.poses_w2c = poses_w2c

        loss_dict, stats_dict, plotting_dict = self.loss_module.compute_loss\
            (self.settings, data_dict, output_dict, mode="train", plot=plot, iteration=iteration)
        
        # metrics for logging
        output_dict['mse'], output_dict['mse_fine'] = compute_mse_on_rays(data_dict, output_dict)
        results_dict = self.make_result_dict(self.settings, data_dict, output_dict, loss_dict)
        results_dict['avg_pred_depth'] = output_dict.depth.detach().mean()
        results_dict.update(stats_dict)
        results_dict['loss'] = loss_dict['all']  # the actual loss, used to back-propagate

        if iteration % self.settings.log_steps == 0:
            # relatively heavy computation, so do not do it at each step
            # for example, pose alignement + registration evaluation
            results_dict.update(self.make_results_dict_low_freq(self.settings, data_dict, output_dict, loss_dict))
        
        if plot:
            # render the full image
            with torch.no_grad():
                self.net.eval()
                img_idx = np.random.randint(B)
                output_dict = self.net.render_image_at_specific_rays\
                    (self.settings, data_dict, img_idx=img_idx, iter=iteration, mode="train")  
                # will render the full image
                plotting_dict_ = self.visualize(self.settings, data_dict, output_dict, 
                                                split='train')
                self.net.train()
                plotting_dict.update(plotting_dict_)

        # update the progress bar
        if hasattr(self.net.nerf, 'progress') and self.settings.barf_c2f is not None and self.settings.apply_cf_pe:
            # will update progress here 
            self.net.nerf.progress.data.fill_(self.iteration_nerf/self.settings.max_iter)
            if self.settings.nerf.fine_sampling and hasattr(self.net.nerf_fine, 'progress'):
                self.net.nerf_fine.progress.data.fill_(self.iteration_nerf/self.settings.max_iter)
        
        return output_dict, results_dict, plotting_dict 

    @torch.no_grad()
    def make_result_dict(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                         loss: Dict[str, Any], metric: Dict[str, Any]=None, 
                         split: str='train') -> Dict[str, Any]:
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = super().make_result_dict(opt,data_dict, output_dict,loss,metric=metric,split=split)

        # log learning rate
        if split=="train":
            lr = self.optimizer.param_groups[0]["lr"]
            stats_dict["lr"] = lr
            if hasattr(opt, 'nerf') and opt.nerf.fine_sampling:
                lr = self.optimizer.param_groups[1]["lr"]
                stats_dict["lr_fine"] = lr

        # log progress bar
        if hasattr(self.net.nerf, 'progress'):
            stats_dict['cf_pe_progress'] = self.net.nerf.progress.data.item()
        
        # log PSNR
        if 'mse' in output_dict.keys():
            # compute PSNR
            psnr = -10* output_dict['mse'].log10()
            stats_dict['PSNR'] = psnr
            if 'mse_fine' in output_dict.keys() and output_dict['mse_fine'] is not None:
                psnr = -10*output_dict['mse_fine'].log10()
                stats_dict['PSNR_fine'] = psnr

        # if depth is available, compute depth error
        if 'depth_gt' in data_dict.keys() and 'depth' in output_dict.keys():
            scaling_factor_for_pred_depth = 1.
            if self.settings.model == 'joint_pose_nerf_training' and \
                hasattr(self.net, 'sim3_est_to_gt_c2w'):
                # adjust the scaling of the depth since the optimized scene geometry + poses 
                # are all valid up to a 3D similarity
                scaling_factor_for_pred_depth = (self.net.sim3_est_to_gt_c2w.trans_scaling_after * self.net.sim3_est_to_gt_c2w.s) \
                    if self.net.sim3_est_to_gt_c2w.type == 'align_to_first' else self.net.sim3_est_to_gt_c2w.s

            abs_e, rmse = compute_depth_error_on_rays(data_dict, output_dict, output_dict.depth, 
                                                      scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
            stats_dict['depth_abs'] = abs_e
            stats_dict['depth_rmse'] = rmse
            if 'depth_fine' in output_dict.keys():
                abs_e, rmse = compute_depth_error_on_rays(data_dict, output_dict, output_dict.depth_fine, 
                                                          scaling_factor_for_pred_depth=scaling_factor_for_pred_depth)
                stats_dict['depth_abs_fine'] = abs_e
                stats_dict['depth_rmse_fine'] = rmse
        return stats_dict

    @torch.no_grad()
    def make_results_dict_low_freq(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],loss: Dict[str, Any],
                                   metric: Dict[str, Any]=None, idx_optimized_pose: List[int]=None, split:str="train"):
        """
        Here corresponds to dictionary which will be saved in tensorboard and also logged"""
        return {}

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        """Get ground-truth (canonical) camera poses"""
        pose_GT = self.train_data.get_all_camera_poses(opt).to(self.device)
        return None, pose_GT

    @torch.no_grad()
    def evaluate_full(self, opt: Dict[str, Any], plot: bool=False, 
                      save_ind_files: bool=False, out_scene_dir: str=''):
        """
        Does the actual evaluation here on the test set. Important that opt is given as variable to change
        the test time optimization input. 
        Args:
            opt (edict): settings
            plot (bool, optional): Defaults to False.
            save_ind_files (bool, optional): Defaults to False
            out_scene_dir (str, optional): Path to dir, to save visualization if plot is True. Defaults to ''.
        Returns: dict with evaluation results
        """
        self.net.eval()
        # loader = tqdm.tqdm(self.val_loader,desc="evaluating",leave=False)
        res = []
        if plot:
            os.makedirs(out_scene_dir, exist_ok=True)

        results_dict = {'single': {}}
        for i, batch in enumerate(self.val_loader):
            # batch contains a single image here
            data_dict = edict(self.to_cuda(batch))
            
            file_id = os.path.basename(batch['rgb_path'][0]).split('.')[0] if 'rgb_path' \
                in batch.keys() else 'idx_{}'.format(batch['idx'])
                
            total_img_coarse, total_img_fine = [], []

            if opt.model in ["joint_pose_nerf_training", 'nerf_fixed_noisy_poses'] and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                data_dict = self.evaluate_test_time_photometric_optim(opt, data_dict)
                # important is data_dict.pose_refine_test
            H, W = data_dict.image.shape[-2:]
            opt.H, opt.W = H, W

            output_dict = self.net.forward(opt,data_dict,mode="eval", iter=None)  

            # evaluate view synthesis, coarse
            scaling_factor_for_pred_depth = 1.
            if self.settings.model == 'joint_pose_nerf_training' and hasattr(self.net, 'sim3_est_to_gt_c2w'):
                # adjust the rendered depth, since the optimized scene geometry and poses are valid up to a 3D
                # similarity, compared to the ground-truth. 
                scaling_factor_for_pred_depth = (self.net.sim3_est_to_gt_c2w.trans_scaling_after * self.net.sim3_est_to_gt_c2w.s) \
                    if self.net.sim3_est_to_gt_c2w.type == 'align_to_first' else self.net.sim3_est_to_gt_c2w.s

            # gt image
            gt_rgb_map = data_dict.image  # (B, 3, H, W)

            # rendered image and depth map
            pred_rgb_map = output_dict.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            pred_depth = output_dict.depth  # [B, -1, 1]
    
            results = compute_metrics(data_dict, output_dict, pred_rgb_map, pred_depth, gt_rgb_map, 
                                      lpips_loss=self.lpips_loss, 
                                      scaling_factor_for_pred_depth=scaling_factor_for_pred_depth, suffix='_c')

            if 'depth_fine' in output_dict.keys():
                pred_rgb_map_fine = output_dict.rgb_fine.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                pred_depth_fine = output_dict.depth_fine
                results_fine = compute_metrics(data_dict, output_dict, pred_rgb_map_fine, 
                                               pred_depth_fine, gt_rgb_map, 
                                               scaling_factor_for_pred_depth=scaling_factor_for_pred_depth, 
                                               lpips_loss=self.lpips_loss, suffix='_f') 
                results.update(results_fine)       

            res.append(results)

            message = "==================\n"
            message += "{}, curr_id: {}, shape {}x{} \n".format(self.settings.scene, file_id, H, W)
            for k, v in results.items():
                message += 'current {}: {:.2f}\n'.format(k, v)
            self.logger.info(message)
            
            results_dict['single'][file_id] = results

            # plotting   
            depth_range = None
            if plot:
                # invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                # invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                depth = output_dict.depth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) * scaling_factor_for_pred_depth # [B,1,H,W]
                depth_var = output_dict.depth_var.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                
                if hasattr(data_dict, 'depth_range'):
                    depth_range = data_dict.depth_range[0].cpu().numpy().tolist()

                self.visualize_eval(total_img_coarse, data_dict.image, pred_rgb_map, depth, 
                                    depth_var, depth_range=depth_range)
                if 'depth_gt' in data_dict.keys():
                    depth_gt = data_dict.depth_gt[0]
                    depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), 
                                                          range=depth_range, append_cbar=False)).astype(np.uint8)
                    total_img_coarse += [depth_gt_colored]

                if 'depth_fine' in output_dict.keys():
                    #invdepth_fine = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    #invdepth_map_fine = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    depth = output_dict.depth_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) * scaling_factor_for_pred_depth  # [B,1,H,W]
                    depth_var = output_dict.depth_var_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                    self.visualize_eval(total_img_fine, data_dict.image, pred_rgb_map_fine, depth, 
                                        depth_var, depth_range=depth_range)
                    if 'depth_gt' in data_dict.keys():
                        depth_gt = data_dict.depth_gt[0]
                        depth_gt_colored = (255 * colorize_np(depth_gt.cpu().squeeze().numpy(), 
                                                            range=depth_range, append_cbar=False)).astype(np.uint8)
                        total_img_fine += [depth_gt_colored]
                
                # save the final image
                total_img_coarse = np.concatenate(total_img_coarse, axis=1)
                if len(total_img_fine) > 2:
                    total_img_fine = np.concatenate(total_img_fine, axis=1)
                    total_img = np.concatenate((total_img_coarse, total_img_fine), axis=0)
                else:
                    total_img = total_img_coarse
                if 'depth_gt' in data_dict.keys():
                    name = '{}_gt_rgb_depthc_depthvarc_depthgt.png'.format(file_id)
                else:
                    name = '{}_gt_rgb_depthc_depthvarc.png'.format(file_id)
                imageio.imwrite(os.path.join(out_scene_dir, name), total_img)  


            if save_ind_files: 
                pred_img_plot = pred_rgb_map_fine if 'depth_fine' in output_dict.keys() else pred_rgb_map
                pred_depth_plot = pred_depth_fine.view(-1,opt.H,opt.W,1).permute(0,3,1,2) if 'depth_fine' in output_dict.keys() \
                    else pred_depth.view(-1,opt.H,opt.W,1).permute(0,3,1,2)
                
                depth_gt_image = data_dict.depth_gt if 'depth_gt' in data_dict.keys() else None
                self.save_ind_files(out_scene_dir, file_id, data_dict.image, pred_img_plot, 
                                    pred_depth_plot * scaling_factor_for_pred_depth, 
                                    depth_range=depth_range, depth_gt=depth_gt_image)

        # compute average results over the full test set
        avg_results = {}
        keys = res[0].keys()
        for key in keys:
            avg_results[key] = np.mean([r[key] for r in res])
        results_dict['mean'] = avg_results
        
        # log results
        message = "------avg over {}-------\n".format(self.settings.scene)
        for k, v in avg_results.items():
            message += 'current {}: {:.3f}\n'.format(k, v)
        self.logger.info(message)
        return results_dict

    @torch.no_grad()
    def generate_videos_synthesis(self,opt: Dict[str, Any],eps=1e-10):
        """
        Will generate a video by sampling poses and rendering the corresponding images. 
        """
        opt.output_path = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        self.net.eval()

        pose_pred,pose_GT = self.get_all_training_poses(opt)

        poses = pose_pred if opt.model=="joint_pose_nerf_training" else pose_GT
        poses_c2w = camera.pose.invert(poses)

        intr = self.train_data.all.intr[:1].to(self.device) # grab intrinsics
        focal = intr[0][0, 0]
        depth_range = self.train_data.all['depth_range'][0].tolist()

        # alternative 1 
        if 'llff' in self.settings.dataset: 
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)
            
            pose_novel_c2w = generate_spiral_path(poses_c2w, bounds=np.array(depth_range), 
                                                  n_frames=60).to(self.device)

            pose_novel = camera.pose.invert(pose_novel_c2w)               

        elif 'dtu' in self.settings.dataset:
            # dtu
            n_frame = 60
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)

            pose_novel_c2w = generate_spiral_path_dtu(poses_c2w, n_rots=1, n_frames=n_frame).to(self.device)
            pose_novel_c2w = pose_novel_c2w
            pose_novel_1 = camera.pose.invert(pose_novel_c2w) 

            # rotate novel views around the "center" camera of all poses
            scale = 1
            test_poses_w2c = poses
            idx_center = (test_poses_w2c-test_poses_w2c.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel_2 = camera.get_novel_view_poses(opt,test_poses_w2c[idx_center],N=n_frame,scale=scale).to(self.device)

            pose_novel = torch.cat((pose_novel_1, pose_novel_2), dim=0)
            # render the novel views
        else:
            n_frame = 60
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)

            scale = 1
            test_poses_w2c = poses
            idx_center = (test_poses_w2c-test_poses_w2c.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,test_poses_w2c[idx_center],N=n_frame,scale=scale).to(self.device)
        
        pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)

        H, W = self.train_data.all.image.shape[-2:]
        for i,pose_aligned in enumerate(pose_novel_tqdm):
            ret = self.net.render_image_at_specific_pose_and_rays\
                (opt, self.train_data.all, pose_aligned, intr, H, W, mode='val', iter=None)
            if 'rgb_fine' in ret:
                depth = ret.depth_fine
                rgb_map = ret.rgb_fine.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
            else:
                depth = ret.depth
                rgb_map = ret.rgb.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
            depth_map = depth.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
            color_depth_map = colorize_np(depth_map.cpu().squeeze().numpy(), range=depth_range, append_cbar=False)
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
            imageio.imwrite("{}/depth_{}.png".format(novel_path,i), (255*color_depth_map).astype(np.uint8))

        # write videos
        self.logger.info("writing videos...")
        
        rgb_vid_fname = "{}/novel_view_{}_{}_rgb".format(opt.output_path, opt.dataset, opt.scene)
        depth_vid_fname = "{}/novel_view_{}_{}_depth".format(opt.output_path, opt.dataset, opt.scene)
        os.system("ffmpeg -y -framerate 10 -i {0}/rgb_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
        os.system("ffmpeg -y -framerate 10 -i {0}/depth_%d.png -pix_fmt yuv420p {1}.mp4 >/dev/null 2>&1".format(novel_path,depth_vid_fname))

        os.system("ffmpeg -f image2 -framerate 10 -i {0}/rgb_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
        os.system("ffmpeg -f image2 -framerate 10 -i {0}/depth_%d.png -loop -1 {1}.gif >/dev/null 2>&1".format(novel_path,depth_vid_fname))
        self.logger.info('Finished creating video')
        return