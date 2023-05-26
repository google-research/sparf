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

import glob
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Any, Dict, Tuple


from source.datasets.base import Dataset
from source.utils import camera as camera


def as_intrinsics_matrix(intrinsics: np.ndarray) -> np.ndarray:
    """
    Get matrix representation of intrinsics.
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


class BaseRGBDDataset(Dataset):
    def __init__(self, args: Dict[str, Any], split: str, scale: float=1.):
        super().__init__(args, split)
        self.scale = scale
        self.png_depth_scale = None
        self.distortion = None

    def compute_3d_bounds(self, opt: Dict[str, Any], H: int, W: int, intrinsics: np.ndarray, 
                          poses_w2c: np.ndarray, depth_range: List[float]) -> np.ndarray:
        """Computes the center of the 3D points corresponding to the far range. This
        will be used to re-center the cameras, such as the scene is more or less
        centered at the origin. """
        poses_w2c = torch.from_numpy(poses_w2c).float()
        intrinsics = torch.from_numpy(intrinsics).float()
        # Compute boundaries of 3D space
        max_xyz = torch.full((3,), -1e6)
        min_xyz = torch.full((3,), 1e6)
        near, far = depth_range


        rays_o, rays_d = camera.get_center_and_ray(poses_w2c[:, :3], H, W, intr=intrinsics) 
        # (B, HW, 3), (B, HW, 3)
        
        points_3D_max = rays_o + rays_d * far # [H, W, 3]
        points_3D = points_3D_max
        
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
        bb_center = (max_xyz + min_xyz).detach() / 2.
        return bb_center.numpy()

    def get_all_camera_poses(self, args: Dict[str, Any]):
        # of the current split
        if isinstance(self.render_poses_c2w, list):
            return torch.from_numpy(np.linalg.inv(np.stack(self.render_poses_c2w, axis=0)))[:, :3].float()  # (B, 3, 4)
        else:
            # numpy array
            return torch.from_numpy(np.linalg.inv(self.render_poses_c2w))[:, :3].float()  # (B, 3, 4)

    def read_image_and_depth(self, color_path: str, depth_path: str, K: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]: 
        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError
        
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        # the intrinsics correspond to the depth size, so the image need to be resized to depth size. 
        color_data = cv2.resize(color_data, (W, H))
        return color_data, depth_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int)

        Returns:
            a dictionary for each image index containing the following elements: 
                * idx: the index of the image
                * rgb_path: the path to the RGB image. Will be used to save the renderings with a name. 
                * image: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are 
                            normalized to [0, 1] (not [0, 255]). 
                * intr: intrinsics parameters, numpy array of shape [3, 3]
                * pose:  world-to-camera transformation matrix in OpenCV format, numpy array of shaoe [3, 4]
                * depth_range: depth_range, numpy array of shape [1, 2]
                * scene: self.scenes[render_scene_id]

                * depth_gt: ground-truth depth map, numpy array of shape [H, W]
                * valid_depth_gt: mask indicating where the depth map is valid, bool numpy array of shape [H, W]
        """
        
        rgb_file = self.render_rgb_files[idx]
        depth_file = self.render_depth_files[idx]
        scene = self.scene
        render_pose_c2w = self.render_poses_c2w[idx].copy()
        render_intrinsics = self.intrinsics.copy()

        rgb, depth = self.read_image_and_depth(rgb_file, depth_file, render_intrinsics)
        depth = depth*self.scale
        
        render_pose_c2w[:3, 3] *= self.scale
        render_pose_w2c = np.linalg.inv(render_pose_c2w)

        edge_h = self.crop_edge_h
        edge_w = self.crop_edge_w
        if edge_h > 0 or edge_w > 0:
            # crop image edge, there are invalid value on the edge of the color image
            rgb = rgb[edge_h:-edge_h, edge_w:-edge_w]
            depth = depth[edge_h:-edge_h, edge_w:-edge_w]
            # need to adapt the intrinsics accordingly
            render_intrinsics[0, 2] -= edge_w
            render_intrinsics[1, 2] -= edge_h

        # process by resizing and cropping 
        rgb, render_intrinsics, depth = self.preprocess_image_and_intrinsics(
            rgb, intr=render_intrinsics, depth=depth, channel_first=False)

        ret =  {
            'idx': idx, 
            'rgb_path': rgb_file,
            'image': rgb.permute(2, 0, 1), # torch tensor (3, self.H, self.W)
            'depth_gt': depth, # (H, W)
            'valid_depth_gt': depth > 0., 
            'intr': render_intrinsics[:3, :3].astype(np.float32),  # 3x3
            'pose':  render_pose_w2c[:3].astype(np.float32),   # # 3x4, world to camera
            'scene': self.scene
        }
        
        depth_range = torch.tensor([self.near*(1-self.args.increase_depth_range_by_x_percent), 
                                    self.far*(1+self.args.increase_depth_range_by_x_percent)])
        ret['depth_range'] = depth_range
        return ret


class ReplicaPerScene(BaseRGBDDataset):
    def __init__(self, args: Dict[str, Any], split: str, scenes: str, scale: float=1.):
        super(ReplicaPerScene, self).__init__(args, split, scale)
        
        self.base_dir = args.env.replica
        self.scene = scenes
        self.input_folder = os.path.join(args.env.replica, scenes)
        self.color_paths = np.array(sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg')))
        self.depth_paths = np.array(sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')  # camera to world in self.poses_c2w

        # fixed for the whole dataset
        self.H, self.W, fx, fy, cx, cy = 680, 1200, 600.0, 600.0, 599.5, 339.5
        self.intrinsics = as_intrinsics_matrix((fx, fy, cx, cy))
        self.png_depth_scale = 6553.5 #for depth image in png format
        self.crop_edge_w = self.crop_edge_h = 0 # does not crop any edges

        self.scale = 1.  # for the translation pose and the depth
        
        # recenter all the cameras 
        translation_recenter = True
        if translation_recenter:
            avg_trans = self.poses_c2w[:, :3, -1].mean(0).reshape(1, -1)
            self.poses_c2w[:, :3, -1] -= avg_trans

        self.define_train_and_test_splits(self.color_paths, self.depth_paths, self.poses_c2w.copy())
    
    def __len__(self):
        return len(self.render_rgb_files)

    def define_train_and_test_splits(self, color_paths: List[str], depth_paths: List[str], c2w_poses: np.ndarray):
        
        # these were compute with fps.
        # define depth ranges
        if self.scene == 'room1' or self.scene == 'office1' or self.scene == 'office0':
            self.near, self.far = 0.1, 4.5
        else:
            self.near, self.far = 0.1, 6.5
                
        start = 0
        n_train_img = len(self.poses_c2w)
        if self.scene == 'office0':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 50
            else:
                train_interval = 80
            test_interval = 10
        elif self.scene == 'office1':
            if self.args.train_sub is not None and self.args.train_sub > 6:
                train_interval = 80
            elif self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 200
            test_interval = 50
        elif self.scene == 'office2':
            if self.args.train_sub is not None and self.args.train_sub > 6:
                train_interval = 80
            elif self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 150
            test_interval = 10
        elif self.scene == 'office3':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 200
            else:
                train_interval = 350
            test_interval = 30
        elif self.scene == 'office4':
            start = 850
            train_interval = 100
            test_interval = 30
        elif self.scene == 'room0':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                train_interval = 100
            else:
                train_interval = 250
            test_interval = 10
        elif self.scene == 'room1':
            if self.args.train_sub is not None and self.args.train_sub > 3:
                start = 300
                train_interval = 100
            else:
                train_interval = 50
            test_interval = 10
        else:
            train_interval = 80
            test_interval = 10  
        i_train = np.arange(start, n_train_img)[::train_interval].astype(np.int)

        if self.args.train_sub is not None:
            i_train = i_train[:self.args.train_sub]

        end_test = i_train[-1] + test_interval
        i_test = np.array([int(j) for j in np.arange(start, end_test) if (j not in i_train)])[::test_interval]


        train_rgb_files = color_paths[i_train]
        train_depth_files = depth_paths[i_train]
        train_poses_c2w = c2w_poses[i_train]

        depth_range = [self.near, self.far]
        
        # this is very IMPORTANT
        # need to adjust the pose such as the scene is centered around 0 for the 
        # selected training images
        # otherwise, the range of 3D points covered by the scenes is very far off. 
        avg_trans = self.compute_3d_bounds(self.args, self.H, self.W, self.intrinsics, 
                                           np.linalg.inv(train_poses_c2w), depth_range)
        # rescale all the poses
        c2w_poses[:, :3, -1] -=  avg_trans.reshape(1, -1) 
        self.poses_c2w[:, :3, -1] -= avg_trans.reshape(1, -1)
        train_poses_c2w = c2w_poses[i_train]

        test_rgb_files = color_paths[i_test]
        test_depth_files = depth_paths[i_test]
        test_poses_c2w = c2w_poses[i_test]

        if self.split == 'train':
            self.render_rgb_files = train_rgb_files
            self.render_poses_c2w = train_poses_c2w
            self.render_depth_files = train_depth_files
        else:
            self.render_rgb_files = test_rgb_files
            self.render_poses_c2w = test_poses_c2w
            self.render_depth_files = test_depth_files

    def load_poses(self, path: str):
        poses_c2w = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4).astype(np.float32)
            poses_c2w.append(c2w)
        self.poses_c2w = np.stack(poses_c2w, axis=0)  # (B, 4, 4)
        self.valid_poses = np.arange(self.poses_c2w.shape[0]).tolist()
        # here, all the poses are valid
        return True