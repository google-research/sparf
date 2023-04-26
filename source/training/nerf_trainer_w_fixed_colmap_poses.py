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
import torch.nn as nn
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

from source.training.engine.iter_based_trainer import get_log_string
import source.training.nerf_trainer as nerf
from source.models.renderer import Graph
import source.utils.camera as camera
from source.training.joint_pose_nerf_trainer import CommonPoseEvaluation


class InitialPoses(torch.nn.Module):
    def __init__(self, opt: Dict[str, Any], nbr_poses: int, initial_poses_w2c: torch.Tensor,  
                 device: torch.device):
        super().__init__()

        self.opt = opt
        self.nbr_poses = nbr_poses  # corresponds to the total number of poses!
        self.device = device
        self.initial_poses_w2c = initial_poses_w2c  # corresponds to initialization of all the poses
        # including the ones that are fixed
        self.initial_poses_c2w = camera.pose.invert(self.initial_poses_w2c)

    def get_initial_w2c(self) -> torch.Tensor:
        return self.initial_poses_w2c
        
    def get_c2w_poses(self) -> torch.Tensor:
        return self.initial_poses_c2w

    def get_w2c_poses(self) -> torch.Tensor:
        return self.initial_poses_w2c


class NerfTrainerPerSceneWColmapFixedPoses(nerf.NerfTrainerPerScene, CommonPoseEvaluation):

    def __init__(self,opt):
        super().__init__(opt)

    def build_pose_net(self, opt: Dict[str, Any]):

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

            '''
            if self.settings.debug:
                for i in range(self.train_data.all.image.shape[0]):
                    for j in range(i+1, self.train_data.all.image.shape[0]):
                        verify_colmap_depth(data_dict=self.train_data.all, poses_w2c=initial_poses_w2c, 
                                            index_0=i, index_1=j)
            '''

        assert len(valid_poses_idx) + len(index_images_excluded) == initial_poses_w2c.shape[0]

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


        self.pose_net = InitialPoses(opt, nbr_poses=len(self.train_data), 
                                     initial_poses_w2c=initial_poses_w2c, 
                                     device=self.device)

    def build_nerf_net(self, opt: Dict[str, Any], pose_net: Any):
        self.logger.info('Creating NerF model for training with fixed colmap poses')
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

    @torch.no_grad()
    def evaluate_poses(self, opt: Dict[str, Any], idx_optimized_pose: List[int]=None
                       ) -> Dict[str, torch.Tensor]: 
        pose,pose_GT = self.get_all_training_poses(opt, idx_optimized_pose=idx_optimized_pose)
        return self.evaluate_any_poses(opt, pose, pose_GT)

    @torch.no_grad()
    def get_all_training_poses(self, opt: Dict[str, Any], idx_optimized_pose: List[int]=None
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get ground-truth (canonical) camera poses
        pose_GT_w2c = self.train_data.get_all_camera_poses(opt).to(self.device)
        pose_w2c = self.net.pose_net.get_w2c_poses() 
        if idx_optimized_pose is not None:
            pose_GT_w2c = pose_GT_w2c[idx_optimized_pose].reshape(-1, 3, 4)
            pose_w2c = pose_w2c[idx_optimized_pose].reshape(-1, 3, 4)
        return pose_w2c, pose_GT_w2c

    @torch.no_grad()
    def evaluate_full(self,opt: Dict[str, Any], plot=False, 
                      save_ind_files: bool=False, out_scene_dir=''
                      ) -> Dict[str, torch.Tensor]:
        self.net.eval()
        # evaluate novel view synthesis
        results_dict = super().evaluate_full(opt, plot=plot, out_scene_dir=out_scene_dir)

        # init error
        results_dict['init_rot_error']= self.initial_pose_error['error_R_before_align'].item()
        results_dict['init_trans_error'] = self.initial_pose_error['error_t_before_align'].item()
        return results_dict


# ============================ computation graph for forward/backprop ============================

class Graph(Graph):
    """NeRF (mlp + rendering) system when considering fixed noisy poses. """
    def __init__(self, opt: Dict[str, Any], device: torch.device, 
                 pose_net: Any):
        super().__init__(opt, device)

        # nerf networks already done 
        self.pose_net = pose_net

    def get_w2c_pose(self,opt: Dict[str, Any],data_dict: Dict[str, Any],mode=None) -> torch.Tensor:
        if mode=="train":
            pose = self.pose_net.get_w2c_poses()
        elif mode in ["val","eval","test-optim", "test"]:
            # val is on the validation set
            # eval is during test/actual evaluation at the end 
            # align test pose to refined coordinate system (up to sim3)

            # the poses were aligned at the beginning and fixed, they do not need further alignement
            pose = data_dict.pose

            # Because the fixed poses might be noisy, here can learn an extra alignement on top
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