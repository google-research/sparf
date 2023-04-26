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
import matplotlib.pyplot as plt
import time
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.models.poses_models.axis_rotation import AxisRotationPoseParameters
from source.models.poses_models.two_columns import FirstTwoColunmnsPoseParameters
from source.models.poses_models.quaternion import QuaternionsPoseParameters
from source.training.engine.iter_based_trainer import get_log_string
import source.training.nerf_trainer as nerf
from source.models.renderer import Graph
import source.utils.camera as camera
import source.utils.vis_rendering as util_vis
from source.utils.geometry.align_trajectories import \
    (align_ate_c2b_use_a2b, align_translations, backtrack_from_aligning_and_scaling_to_first_cam, backtrack_from_aligning_the_trajectory)
from source.utils.colmap_initialization.sfm import compute_sfm_pdcnet


class CommonPoseEvaluation:

    def set_initial_poses(self, opt: Dict[str, Any]) -> Tuple[torch.Tensor, List[int], List[int]]:
        """ Define initial poses (to be optimized) """

        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(self.device)
        n_poses = len(self.train_data)
        valid_poses_idx = np.arange(start=0, stop=n_poses, step=1).tolist()
        index_images_excluded = []
        
        if opt.camera.optimize_relative_poses:
            if opt.camera.initial_pose == 'noisy_gt':
                # fix the x first one to the GT poses and the ones to optimize to the GT ones 
                # + noise
                n_poses = len(self.train_data)
                initial_poses_w2c = pose_GT_w2c
                n_first_fixed_poses = opt.camera.n_first_fixed_poses 
                n_optimized_poses = n_poses - n_first_fixed_poses
                
                # same noise level in rotation and translation
                se3_noise = torch.randn(n_optimized_poses, 6, device=self.device)*opt.camera.noise
                pose_noise = camera.lie.se3_to_SE3(se3_noise)  # (n_optimized_poses, 3, 4)
                pose_noise = torch.cat((torch.eye(3, 4)[None, ...].repeat(n_first_fixed_poses, 1, 1).to(self.device), 
                                        pose_noise), dim=0)
                
                initial_poses_w2c = camera.pose.compose([pose_noise, initial_poses_w2c])
            elif opt.camera.initial_pose == 'given':
                initial_poses_w2c = self.train_data.all.pose_initial.to(self.device)
        else:
            if opt.camera.initial_pose == 'identity':
                initial_poses_w2c = torch.eye(4, 4)[None, ...].repeat(n_poses, 1, 1).to(self.device)
                initial_poses_w2c, trans_scaling = align_translations(pose_GT_w2c, initial_poses_w2c)

            elif opt.camera.initial_pose == 'noisy_gt':
                # similar to barf
                # same noise level in rotation and translation
                se3_noise = torch.randn(n_poses, 6, device=self.device)*opt.camera.noise
                    
                pose_noise = camera.lie.se3_to_SE3(se3_noise)
                initial_poses_w2c = camera.pose.compose([pose_noise, pose_GT_w2c])
            elif opt.camera.initial_pose == 'given':
                initial_poses_w2c = self.train_data.all.pose_initial.to(self.device)
            elif 'sfm' in opt.camera.initial_pose:
                # initial poses obtained by COLMAP with different matchers
                # we all save it into a common directory, to make sure that if it was done before, it is not done again

                directory_save = os.path.join(self._base_save_dir, 'colmap_initial_results', self.settings.dataset)
                if self.settings.train_sub is not None and self.settings.train_sub != 0:
                    directory_save = os.path.join(directory_save, 'subset_{}'.format(self.settings.train_sub))
                directory_save = os.path.join(directory_save, self.settings.scene)
                
                directory = directory_save
                os.makedirs(directory_save, exist_ok=True)

                if self.settings.load_colmap_depth:
                    if 'pdcnet' in opt.camera.initial_pose:
                        initial_poses_w2c, valid_poses_idx, index_images_excluded, colmap_depth_map, \
                        colmap_conf_map = compute_sfm_pdcnet(opt=self.settings, data_dict=self.train_data.all, 
                                                             save_dir=directory, 
                                                             load_colmap_depth=True)
                    else:
                        raise ValueError

                    initial_poses_w2c = initial_poses_w2c.to(self.device).float()
                     # pre-aligned initial poses to the ground-truth one
                     # this is important so that scene is still more or less centered around 0
                    initial_poses_w2c, ssim_est_gt_c2w = self.prealign_w2c_small_camera_systems\
                        (opt, initial_poses_w2c[:, :3], pose_GT_w2c[:, :3])
                    trans_scaling = ssim_est_gt_c2w.s

                    self.train_data.all.colmap_depth = colmap_depth_map.to(self.device) * trans_scaling
                    self.train_data.all.colmap_conf = colmap_conf_map.to(self.device)
                else:
                    if 'pdcnet' in opt.camera.initial_pose:
                        initial_poses_w2c, valid_poses_idx, index_images_excluded = \
                            compute_sfm_pdcnet(opt=self.settings, data_dict=self.train_data.all, save_dir=directory)
                    else:
                        raise ValueError

                    initial_poses_w2c = initial_poses_w2c.to(self.device).float()
                    initial_poses_w2c, ssim_est_gt_c2w = self.prealign_w2c_small_camera_systems\
                        (opt, initial_poses_w2c[:, :3], pose_GT_w2c[:, :3])
                    trans_scaling = ssim_est_gt_c2w.s
            else:
                raise ValueError
        return initial_poses_w2c[:, :3], valid_poses_idx, index_images_excluded

    @torch.no_grad()
    def prealign_w2c_large_camera_systems(self,opt: Dict[str, Any],pose_w2c: torch.Tensor,
                                          pose_GT_w2c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the 3D similarity transform relating pose_w2c to pose_GT_w2c. Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)

        if self.settings.camera.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=self.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            try:
                pose_aligned_c2w, ssim_est_gt_c2w = align_ate_c2b_use_a2b(pose_c2w, pose_GT_c2w, method='sim3')
                pose_aligned_w2c = camera.pose.invert(pose_aligned_c2w[:, :3])
                ssim_est_gt_c2w.type = 'traj_align'
            except:
                self.logger.info("warning: SVD did not converge...")
                pose_aligned_w2c = pose_w2c
                ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), type='traj_align', 
                                        t=torch.zeros(1,3,1,device=self.device), s=1.)
        return pose_aligned_w2c, ssim_est_gt_c2w

    @torch.no_grad()
    def prealign_w2c_small_camera_systems(self,opt: Dict[str, Any],pose_w2c: torch.Tensor,
                                          pose_GT_w2c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the transformation from pose_w2c to pose_GT_w2c by aligning the each pair of pose_w2c 
        to the corresponding pair of pose_GT_w2c and computing the scaling. This is more robust than the
        technique above for small number of input views/poses (<10). Save the inverse 
        transformation for the evaluation, where the test poses must be transformed to the coordinate 
        system of the optimized poses. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        def alignment_function(poses_c2w_from_padded: torch.Tensor, 
                               poses_c2w_to_padded: torch.Tensor, idx_a: int, idx_b: int):
            """Args: FInd alignment function between two poses at indixes ix_a and idx_n

                poses_c2w_from_padded: Shape is (B, 4, 4)
                poses_c2w_to_padded: Shape is (B, 4, 4)
                idx_a:
                idx_b:

            Returns:
            """
            # We take a copy to keep the original poses unchanged.
            poses_c2w_from_padded = poses_c2w_from_padded.clone()
            # We use the distance between the same two poses in both set to obtain
            # scale misalgnment.
            dist_from = torch.norm(
                poses_c2w_from_padded[idx_a, :3, 3] - poses_c2w_from_padded[idx_b, :3, 3]
            )
            dist_to = torch.norm(
                poses_c2w_to_padded[idx_a, :3, 3] - poses_c2w_to_padded[idx_b, :3, 3])
            scale = dist_to / dist_from

            # alternative for scale
            # dist_from = poses_w2c_from_padded[idx_a, :3, 3] @ poses_c2w_from_padded[idx_b, :3, 3]
            # dist_to = poses_w2c_to_padded[idx_a, :3, 3] @ poses_c2w_to_padded[idx_b, :3, 3]
            # scale = onp.abs(dist_to /dist_from).mean()

            # We bring the first set of poses in the same scale as the second set.
            poses_c2w_from_padded[:, :3, 3] = poses_c2w_from_padded[:, :3, 3] * scale

            # Now we simply apply the transformation that aligns the first pose of the
            # first set with first pose of the second set.
            transformation_from_to = poses_c2w_to_padded[idx_a] @ camera.pose_inverse_4x4(
                poses_c2w_from_padded[idx_a])
            poses_aligned_c2w = transformation_from_to[None] @ poses_c2w_from_padded

            poses_aligned_w2c = camera.pose_inverse_4x4(poses_aligned_c2w)
            ssim_est_gt_c2w = edict(R=transformation_from_to[:3, :3].unsqueeze(0), type='traj_align', 
                                    t=transformation_from_to[:3, 3].reshape(1, 3, 1), s=scale)

            return poses_aligned_w2c[:, :3], ssim_est_gt_c2w

        pose_c2w = camera.pose.invert(pose_w2c)
        pose_GT_c2w = camera.pose.invert(pose_GT_w2c)
        B = pose_c2w.shape[0]

        if self.settings.camera.n_first_fixed_poses > 1:
            # the trajectory should be consistent with the first poses 
            ssim_est_gt_c2w = edict(R=torch.eye(3,device=self.device).unsqueeze(0), 
                                    t=torch.zeros(1,3,1,device=self.device), s=1.)
            pose_aligned_w2c = pose_w2c
        else:
            # try every combination of pairs and get the rotation/translation
            # take the one with the smallest error
            # this is because for small number of views, the procrustes alignement with SVD is not robust. 
            pose_aligned_w2c_list = []
            ssim_est_gt_c2w_list = []
            error_R_list = []
            error_t_list = []
            full_error = []
            for pair_id_0 in range(min(B, 10)):  # to avoid that it is too long
                for pair_id_1 in range(min(B, 10)):
                    if pair_id_0 == pair_id_1:
                        continue
                    
                    pose_aligned_w2c_, ssim_est_gt_c2w_ = alignment_function\
                        (camera.pad_poses(pose_c2w), camera.pad_poses(pose_GT_c2w),
                         pair_id_0, pair_id_1)
                    pose_aligned_w2c_list.append(pose_aligned_w2c_)
                    ssim_est_gt_c2w_list.append(ssim_est_gt_c2w_ )

                    error = self.evaluate_camera_alignment(opt, pose_aligned_w2c_, pose_GT_w2c)
                    error_R_list.append(error.R.mean().item() * 180. / np.pi )
                    error_t_list.append(error.t.mean().item())
                    full_error.append(error.t.mean().item() * (error.R.mean().item() * 180. / np.pi))

            ind_best = np.argmin(full_error)
            # print(np.argmin(error_R_list), np.argmin(error_t_list), ind_best)
            pose_aligned_w2c = pose_aligned_w2c_list[ind_best]
            ssim_est_gt_c2w = ssim_est_gt_c2w_list[ind_best]

        return pose_aligned_w2c, ssim_est_gt_c2w

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt: Dict[str, Any],pose_aligned_w2c: torch.Tensor,
                                  pose_GT_w2c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Measures rotation and translation error between aligned and ground-truth world-to-camera poses. 
        Attention, we want the translation difference between the camera centers in the world 
        coordinate! (not the opposite!)
        Args:
            opt (edict): settings
            pose_aligned_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        Returns:
            error: edict with keys 'R' and 't', for rotation (in radian) and translation erorrs (not averaged)
        """
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

        error = edict(R=R_error,t=t_error)  # not meaned here
        return error
    
    @torch.no_grad()
    def evaluate_any_poses(self, opt: Dict[str, Any], pose_w2c: torch.Tensor, 
                           pose_GT_w2c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluates rotation and translation errors before and after alignment. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_GT_w2c (torch.Tensor): Shape is (B, 3, 4)
        """
        stats_dict = {}
        error = self.evaluate_camera_alignment(opt,pose_w2c.detach(),pose_GT_w2c)
        stats_dict['error_R_before_align'] = error.R.mean() * 180. / np.pi  # radtodeg
        stats_dict['error_t_before_align'] = error.t.mean()

        if pose_w2c.shape[0] > 10:
            pose_aligned,_ = self.prealign_w2c_large_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        else:
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose_w2c.detach(),pose_GT_w2c)
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT_w2c)
        stats_dict['error_R'] = error.R.mean() * 180. / np.pi  # radtodeg
        stats_dict['error_t'] = error.t.mean()
        return stats_dict

    @torch.no_grad()
    def evaluate_poses(self, opt: Dict[str, Any], 
                       idx_optimized_pose: List[int]=None) -> Dict[str, torch.Tensor]: 
        pose,pose_GT = self.get_all_training_poses(opt, idx_optimized_pose=idx_optimized_pose)
        return self.evaluate_any_poses(opt, pose, pose_GT)

    def visualize_any_poses(self, opt: Dict[str, Any], pose_w2c: torch.Tensor, pose_ref_w2c: torch.Tensor, 
                            step: int=0, idx_optimized_pose: List[int]=None, split: str="train") -> Dict[str, torch.Tensor]:
        """Plot the current poses versus the reference ones, before and after alignment. 

        Args:
            opt (edict): settings
            pose_w2c (torch.Tensor): Shape is (B, 3, 4)
            pose_ref_w2c (torch.Tensor): Shape is (B, 3, 4)
            step (int, optional): Defaults to 0.
            split (str, optional): Defaults to "train".
        """
        
        plotting_dict = {}
        fig = plt.figure(figsize=(20,10) if 'nerf_synthetic' in opt.dataset else ((16,8)))


        # we show the poses without the alignment here
        pose_aligned, pose_ref = pose_w2c.detach().cpu(), pose_ref_w2c.detach().cpu()
        if 'llff' in opt.dataset:
            pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,ep=self.iteration)
        else:

            pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                        ep=self.iteration)
        
        '''
        if self.settings.debug:
            import imageio
            imageio.imwrite('pose_before_align_{}.png'.format(self.iteration), pose_vis)
        '''
        pose_vis = torch.from_numpy(pose_vis.astype(np.float32)/255.).permute(2, 0, 1)
        plotting_dict['poses_before_align'] = pose_vis

        # trajectory alignment
        if pose_w2c.shape[0] > 9:
            # the alignement will work well when more than 10 poses are available
            pose_aligned,_ = self.prealign_w2c_large_camera_systems(opt,pose_w2c.detach(),pose_ref_w2c)
        else:
            # for few number of images/poses, the above alignement doesnt work well
            # align the first camera and scale with the relative to the second
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose_w2c.detach(),pose_ref_w2c)
        
        pose_aligned = pose_aligned.detach().cpu()
        if 'llff' in opt.dataset:
            pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                ep=self.iteration)
        else:
            pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                        ep=self.iteration)
        

        pose_vis = torch.from_numpy(pose_vis.astype(np.float32)/255.).permute(2, 0, 1)
        plotting_dict['poses_after_align'] = pose_vis
        return plotting_dict


    def visualize_poses(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                        step: int=0,idx_optimized_pose: List[int]=None, 
                        split: str="train") -> Dict[str, torch.Tensor]:
        pose,pose_ref_ = self.get_all_training_poses(opt, idx_optimized_pose=idx_optimized_pose)
        return self.visualize_any_poses(opt, pose, pose_ref_, step, split=split)
    
    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self, opt: Dict[str, Any], 
                                             data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run test-time optimization. Optimizes over data_dict.se3_refine_test"""
        # only optimizes for the test pose here
        data_dict.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=self.device))
        optimizer = getattr(torch.optim,opt.optim.algo_pose)
        optim_pose = optimizer([dict(params=[data_dict.se3_refine_test],lr=opt.optim.lr_pose)])
        #iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in range(opt.optim.test_iter):
            optim_pose.zero_grad()
            
            data_dict.pose_refine_test = camera.lie.se3_to_SE3(data_dict.se3_refine_test)
            output_dict = self.net.forward(opt,data_dict,mode="test-optim", iter=None)

            # current estimate of the pose
            poses_w2c = self.net.get_pose(self.settings, data_dict, mode='test-optim')  # is it world to camera
            data_dict.poses_w2c = poses_w2c

            # iteration needs to reflect the overall training
            loss, stats_dict, plotting_dict = self.loss_module.compute_loss\
                (opt, data_dict, output_dict, iteration=self.iteration, mode='test-optim')
            loss.all.backward()
            optim_pose.step()
            # iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return data_dict



class PoseAndNerfTrainerPerScene(nerf.NerfTrainerPerScene, CommonPoseEvaluation):
    """Base class for joint pose-NeRF training. Inherits from NeRFTrainerPerScene. 
    """
    def __init__(self,opt: Dict[str, Any]):
        super().__init__(opt)

    def build_pose_net(self, opt: Dict[str, Any]):
        """Defines initial poses. Define parametrization of the poses. 
        """
        
        # get initial poses (which will be optimized)
        # if load_colmap_depth, it will be integrated to the data here! 
        initial_poses_w2c, valid_poses_idx, index_images_excluded = self.set_initial_poses(opt)
        if opt.load_colmap_depth:
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
                self.write_event('train', {'colmap_depth_err': error}, self.iteration)
            
        assert len(valid_poses_idx) + len(index_images_excluded) == initial_poses_w2c.shape[0]

        # log and save to tensorboard initial pose errors
        self.logger.critical('Found {}/{} valid initial poses'\
            .format(len(valid_poses_idx), initial_poses_w2c.shape[0]))

        # evaluate initial poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(self.device)

        stats_dict = self.evaluate_any_poses(opt, initial_poses_w2c, pose_GT)
        self.initial_pose_error = stats_dict

        message = get_log_string(stats_dict)
        self.logger.critical('All initial poses: {}'.format(message))
        self.write_event('train', {'nbr_excluded_poses': len(index_images_excluded)}, self.iteration)
        self.write_event('train', stats_dict, self.iteration)

        # define pose parametrization
        if opt.camera.pose_parametrization == 'axis_angle':
            # initially used in barf
            self.pose_net = AxisRotationPoseParameters(opt, nbr_poses=len(self.train_data), 
                                                       initial_poses_w2c=initial_poses_w2c, 
                                                       device=self.device)
        elif opt.camera.pose_parametrization == 'two_columns':
            # used in scnet, gbarf
            # optimize camera to pose, the two first columns of rotation matrix + translation vector
            self.pose_net = FirstTwoColunmnsPoseParameters(opt, nbr_poses=len(self.train_data), 
                                                           initial_poses_w2c=initial_poses_w2c, 
                                                           device=self.device)
        elif opt.camera.pose_parametrization == 'quaternion': 
            self.pose_net = QuaternionsPoseParameters(opt, nbr_poses=len(self.train_data), 
                                                      initial_poses_w2c=initial_poses_w2c, 
                                                      device=self.device)
        else:
            raise ValueError
        return 

    def build_nerf_net(self, opt: Dict[str, Any], pose_net: torch.nn.Module):
        self.logger.info('Creating NerF model for joint pose-NeRF training')
        self.net = Graph(opt, self.device, pose_net)
        return

    def build_networks(self,opt: Dict[str, Any]):
        self.logger.info("building networks...")

        if opt.use_flow:
            self.build_correspondence_net(opt)

        self.build_pose_net(opt) 

        self.build_nerf_net(opt, self.pose_net)
        
        plotting_dict = self.visualize_poses\
            (opt,data_dict=None,output_dict=None,step=0, split='train')
        self.write_image('train', plotting_dict, self.iteration)
        return 

    def setup_optimizer(self,opt: Dict[str, Any]):
        super().setup_optimizer(opt)
        self.logger.info('setting up optimizer for camera poses')

        optimizer = getattr(torch.optim,opt.optim.algo_pose)
        self.optimizer_pose = optimizer([dict(params=self.pose_net.parameters(),
                                              lr=opt.optim.lr_pose)])
        # set up scheduler
        if opt.optim.sched_pose:
            self.logger.info('setting up scheduler for camera poses')
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched_pose.type)
            if opt.optim.lr_pose_end:
                assert(opt.optim.sched_pose.type=="ExponentialLR")
                max_iter = opt.optim.max_iter if hasattr(opt.optim, 'max_iter') else opt.max_iter
                opt.optim.sched_pose.gamma = (opt.optim.lr_pose_end/opt.optim.lr_pose)**(1./max_iter)
            kwargs = { k:v for k,v in opt.optim.sched_pose.items() if k!="type" }
            self.scheduler_pose = scheduler(self.optimizer_pose,**kwargs)
        return 
        
    def update_parameters(self, loss_out: Dict[str, Any]):
        """Update NeRF MLP parameters and pose parameters"""
        if self.settings.optim.warmup_pose:
            # simple linear warmup of pose learning rate
            self.optimizer_pose.param_groups[0]["lr_orig"] = self.optimizer_pose.param_groups[0]["lr"] # cache the original learning rate
            self.optimizer_pose.param_groups[0]["lr"] *= min(1,self.iteration/self.settings.optim.warmup_pose)

        loss_out.backward()

        # camera update
        if self.iteration % self.grad_acc_steps == 0:
            do_backprop = self.after_backward(self.pose_net, self.iteration, 
                                              gradient_clipping=self.settings.pose_gradient_clipping) 
            if do_backprop:
                self.optimizer_pose.step()

            self.optimizer_pose.zero_grad()  # puts the gradient to zero for the poses

            if self.settings.optim.warmup_pose:
                self.optimizer_pose.param_groups[0]["lr"] = self.optimizer_pose.param_groups[0]["lr_orig"] 
                # reset learning rate
            if self.settings.optim.sched_pose: self.scheduler_pose.step()

        # backprop of the nerf network
        do_backprop = self.after_backward(self.net.get_network_components(), self.iteration, 
                                          gradient_clipping=self.settings.nerf_gradient_clipping)
        if self.iteration % self.grad_acc_steps == 0:
            if do_backprop:
                # could be skipped in case of large gradients or something, but we still want to put the optimizer
                # gradient to zero, so it is not further accumulated
                # skipped the accumulation gradient step
                self.optimizer.step()
            self.optimizer.zero_grad()

            # scheduler, here after each step
            if self.scheduler is not None: self.scheduler.step()
        return

    @torch.no_grad()
    def inference(self):
        pose,pose_GT = self.get_all_training_poses(self.settings)

        # computes the alignement between optimized poses and gt ones. Will be used to transform the test
        # poses to coordinate system of optimized poses for the evaluation. 
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().inference()
        return 

    @torch.no_grad()
    def inference_debug(self):
        pose,pose_GT = self.get_all_training_poses(self.settings)
        # computes the alignement between optimized poses and gt ones. Will be used to transform the test
        # poses to coordinate system of optimized poses for the evaluation. 
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().inference_debug()
        return 

    @torch.no_grad()
    def generate_videos_synthesis(self, opt: Dict[str, Any]):
        pose,pose_GT = self.get_all_training_poses(self.settings)
        if pose.shape[0] > 9:
            # alignment of the trajectory
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(self.settings,pose,pose_GT)
        else:
            # alignment of the first cameras
            _,self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(self.settings,pose,pose_GT)
        super().generate_videos_synthesis(opt)
        return 

    @torch.no_grad()
    def make_result_dict(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                         loss: Dict[str, Any],metric: Dict[str, Any]=None,split: str="train"):
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = super().make_result_dict(opt,data_dict,output_dict,loss,metric=metric,split=split)
        if split=="train":
            # log learning rate
            if hasattr(self, 'optimizer_pose'):
                lr = self.optimizer_pose.param_groups[0]["lr"]
                stats_dict["lr_pose"] = lr
        return stats_dict

    @torch.no_grad()
    def make_results_dict_low_freq(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                                   loss: Dict[str, Any],metric: Dict[str, Any]=None,split:str="train") -> Dict[str, torch.Tensor]:
        """Make logging dict. Corresponds to dictionary which will be saved in tensorboard and also logged"""
        stats_dict = {}
        # compute pose error, this is heavy, so just compute it after x iterations
        if split=="train":
            stats_dict = self.evaluate_poses(opt)
        return stats_dict

    @torch.no_grad()
    def visualize(self,opt: Dict[str, Any],data_dict: Dict[str, Any],output_dict: Dict[str, Any],
                  step: int=0, split: str="train") -> Dict[str, torch.Tensor]:
        plotting_dict = super().visualize(opt,data_dict, output_dict, step=step,split=split)
        if split == 'train' and step == 0:
            # only does that once
            plotting_dict_ = self.visualize_poses(opt,data_dict,output_dict,step,split=split)
            plotting_dict.update(plotting_dict_)
        return plotting_dict

    @torch.no_grad()
    def get_all_training_poses(self,opt: Dict[str, Any], 
                               idx_optimized_pose: List[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(self.device)
        pose_w2c = self.net.pose_net.get_w2c_poses()  
        if idx_optimized_pose is not None:
            pose_GT_w2c = pose_GT_w2c[idx_optimized_pose].reshape(-1, 3, 4)
            pose_w2c = pose_w2c[idx_optimized_pose].reshape(-1, 3, 4)
        return pose_w2c, pose_GT_w2c

    @torch.no_grad()
    def evaluate_full(self,opt: Dict[str, Any], plot: bool=False, save_ind_files: bool=False, 
                      out_scene_dir: str='') -> Dict[str, torch.Tensor]:
        self.net.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)
        if pose.shape[0] > 9:
            # alignment of the trajectory
            pose_aligned, self.net.sim3_est_to_gt_c2w = self.prealign_w2c_large_camera_systems(opt,pose,pose_GT)
        else:
            # alignment of the first cameras
            pose_aligned, self.net.sim3_est_to_gt_c2w = self.prealign_w2c_small_camera_systems(opt,pose,pose_GT)
            
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        self.logger.info("--------------------------")
        self.logger.info("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        self.logger.info("trans: {:10.5f}".format(error.t.mean()))
        self.logger.info("--------------------------")
        # dump numbers
        # evaluate novel view synthesis
        results_dict = super().evaluate_full(opt, plot=plot, out_scene_dir=out_scene_dir)
        results_dict['rot_error'] = np.rad2deg(error.R.mean().item())
        results_dict['trans_error'] = error.t.mean().item()

        # init error
        results_dict['init_rot_error']= self.initial_pose_error['error_R_before_align'].item()
        results_dict['init_trans_error'] = self.initial_pose_error['error_t_before_align'].item()
        return results_dict

    @torch.no_grad()
    def generate_videos_pose(self,opt: Dict[str, Any]):
        self.net.eval()

        opt.output_path = '{}/{}'.format(self._base_save_dir, self.settings.project_path)
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []

        for ep in range(0,opt.max_iter+1,self.snapshot_steps):
            # load checkpoint (0 is random init)
            if ep!=0:
                checkpoint_path = '{}/{}/iter-{:04d}.pth.tar'.format(self._base_save_dir, self.settings.project_path, ep)
                if not os.path.exists(checkpoint_path):
                    continue
                self.load_snapshot(checkpoint=ep)

            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_w2c_small_camera_systems(opt,pose.detach(),pose_ref)
            # self.evaluate_camera_alignment(opt,pose_aligned,pose_ref_)
            pose_aligned = pose_aligned.detach().cpu()
            pose_ref = pose_ref.detach().cpu()

            fig = plt.figure(figsize=(16,8))
            if 'llff' in opt.dataset:
                pose_vis = util_vis.plot_save_poses(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                    ep=ep, path=cam_path)
            else:
                pose_vis = util_vis.plot_save_poses_blender(opt,fig,pose_aligned,pose_ref_w2c=pose_ref,
                                                            ep=ep, path=cam_path)
            ep_list.append(ep)
            plt.close()
        # write videos
        self.logger.info("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 20 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)
        return 
        
        
# ============================ computation graph for forward/backprop ============================

class Graph(Graph):

    def __init__(self, opt: Dict[str, Any], device: torch.device, pose_net: torch.nn.Module):
        super().__init__(opt, device)

        # nerf networks already done 
        self.pose_net = pose_net

    def get_w2c_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],
                     mode: str=None) -> torch.Tensor:
        if mode=="train":
            pose = self.pose_net.get_w2c_poses()  # get the current estimates of the camera poses, which are optimized 
        elif mode in ["val","eval","test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end 
            # align test pose to refined coordinate system (up to sim3)
            assert hasattr(self, 'sim3_est_to_gt_c2w')
            pose_GT_w2c = data_dict.pose
            ssim_est_gt_c2w = self.sim3_est_to_gt_c2w
            if ssim_est_gt_c2w.type == 'align_to_first':
                pose = backtrack_from_aligning_and_scaling_to_first_cam(pose_GT_w2c, ssim_est_gt_c2w)
            elif ssim_est_gt_c2w.type == 'traj_align':
                pose = backtrack_from_aligning_the_trajectory(pose_GT_w2c, ssim_est_gt_c2w)
            else:
                raise ValueError
            # Here, we align the test pose to the poses found during the optimization (otherwise wont be valid)
            # that's pose. And can learn an extra alignement on top
            # additionally factorize the remaining pose imperfection
            if opt.optim.test_photo and mode!="val":
                pose = camera.pose.compose([data_dict.pose_refine_test,pose])
        else: 
            raise ValueError
        return pose

    def get_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        return self.get_w2c_pose(opt, data_dict, mode)

    def get_c2w_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode: str=None) -> torch.Tensor:
        w2c = self.get_w2c_pose(opt, data_dict, mode)
        return camera.pose.invert(w2c)
