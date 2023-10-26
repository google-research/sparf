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
import imageio
import torch
import numpy as np
from pathlib import Path
import sys
from easydict import EasyDict as edict
from typing import Any, Union, List, Dict, Tuple

from source.datasets.base import Dataset


class MyData(Dataset):
    """Reading and loading custom data. You need to implement the load_scene_data function (or equivalent)
    to provide a list of images to read, along with their intrinsics and initial camera pose parameters.
    In __get__item, the initial poses correspond to field 'w2c_pose_initial'. Since we need pseudo-ground truth
    poses to run the code, the field 'pose' contains the same initial poses.
    """
    def __init__(self, args: Dict[str, Any], split: str, scenes: str='', **kwargs):
        super().__init__(args, split)
        self.base_dir = self.args.env.lens

        self.all_poses_c2w = []

        self.render_rgb_files, self.render_intrinsics, self.render_poses_c2w_colmap = [], [], []

        # need to update the depth range here
        self.near_depth = 0.1
        self.far_depth = 5.

        if isinstance(scenes, list):
            scenes = scenes[0]
        self.scene = scenes

        print(f"Loading scene {self.scene} from Lens Dataset from split {self.split}...")

        """Here, need to compile all the names of images, as well as read the corresponding initial
        (and ground truth) poses if provided. """

        scene_path = os.path.join(self.base_dir, self.scene) 
        rgb_files, intrinsics, colmap_poses_c2w = \
            self.load_scene_data(scene_path, recenter_cameras=True)
        
        self.all_poses_c2w = colmap_poses_c2w

        # split the files into train and test
        # you can implement different methods here
        all_indices = np.arange(len(rgb_files))
        split_indices = {
            'test': all_indices[all_indices % self.args.dtuhold == 0],
            'train': all_indices,
        }

        indices_train = split_indices['train']
        indices_test = split_indices['test']

        # train split
        if self.args.train_sub is not None:
            # here, will take the subset of 1, 3 or 9
            indices_train = indices_train[:self.args.train_sub]

        if self.args.val_sub is not None:
            indices_test = indices_test[:self.args.val_sub]

        train_rgb_files = np.array(rgb_files)[indices_train]
        train_intrinsics = np.array(intrinsics)[indices_train]
        train_poses_colmap = np.array(colmap_poses_c2w)[indices_train]

        test_rgb_files = np.array(rgb_files)[indices_test]
        test_intrinsics = np.array(intrinsics)[indices_test]
        test_poses_colmap = np.array(colmap_poses_c2w)[indices_test]
        
        # rendering split 
        if 'train' in self.split:
            render_rgb_files = train_rgb_files
            render_intrinsics = train_intrinsics
            render_poses_colmap = train_poses_colmap
        else:
            render_rgb_files = test_rgb_files
            render_intrinsics = test_intrinsics
            render_poses_colmap = test_poses_colmap

        self.render_rgb_files.extend(render_rgb_files.tolist())
        self.render_poses_c2w.extend(render_poses_colmap)
        self.render_intrinsics.extend(render_intrinsics)

        print(f"In total there are {len(self.render_rgb_files)} images in this dataset")

    def load_scene_data(self, scene_path: str, recenter_cameras: bool=True) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Reads the scene data.
        Args:
            scene_path (str)
            recenter_cameras (bool): recenter the cameras such that they are at 0 (only works for object-centered scenes).

        Returns:
            rgb_files: List of N image files to read
            intrinsics: Corresponding intrinsic parameters. Shape is (N, 3, 3)
            colmap_poses_c2w: Initial poses corresponding to the images. Shape is (N, 3, 4) or (N, 4, 4)

        """

        """
        # As an example, 
        metadata_path = os.path.join(scene_path, 'metadata.json')
        f = open(metadata_path)
        json_file = json.load(f)
        intrinsics = []
        arcore_poses_c2w = []
        rgb_files = []

        for image_id in range(len(json_file['images'])):
            image_path, pose_ar_core_c2w, pose_ar_core_w2c, intrinsic = \
                get_arcore_sample(image_id, json_file, scene_path)
            rgb_files.append(image_path)
            intrinsics.append(intrinsic)

            # they are in opengl format ==> put them in opencv!!
            pose_opencv_c2w = pose_ar_core_c2w.copy()
            pose_opencv_c2w[:3, 1] *= -1.0
            pose_opencv_c2w[:3, 2] *= -1.0
            arcore_poses_c2w.append(pose_opencv_c2w)

        intrinsics = np.stack(intrinsics, axis=0)

        if recenter_cameras:
            poses = np.where(np.isinf(arcore_poses_c2w), np.zeros_like(arcore_poses_c2w), arcore_poses_c2w)
            # could also just take the valid one 
            poses_translation = poses[:, :3, 3].mean(0).reshape(1, -1)
            arcore_poses_c2w[:, :3, 3] -= poses_translation

            # scale so that they are at a distance of 2
            arcore_poses_c2w[:, :3, 3] *= abs(2. / min(poses_translation.reshape(-1)))

        return rgb_files, intrinsics, colmap_poses_c2w
        """
        raise NotImplementedError

    def get_all_camera_poses(self, args: Dict[str, Any]):
        # of the current split
        return torch.inverse(torch.from_numpy(np.stack(self.render_poses_c2w_colmap, axis=0)))[:, :3].float()  # (B, 3, 4)

    def __len__(self):
        return len(self.render_rgb_files)

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
                * w2c_pose_initial: world-to-camera initial transformation matricx.
                * pose:  world-to-camera transformation matrix, numpy array of shape [3, 4]. In that case,
                         I use the same than the initial poses (since no real ground-truth is provided)
                * depth_range: depth_range, numpy array of shape [1, 2]
                * scene: scene name
        """
        rgb_file = self.render_rgb_files[idx]

        # read the initial poses
        render_pose_c2w = self.render_poses_c2w[idx]
        render_pose_w2c = np.linalg.inv(render_pose_c2w)

        render_intrinsics = self.render_intrinsics[idx]

        # read and handle the image to render
        rgb = imageio.imread(rgb_file)[:, :, :3]
        h, w = rgb.shape[:2]

        rgb, render_intrinsics = \
            self.preprocess_image_and_intrinsics(rgb, intr=render_intrinsics, channel_first=False)

        near_depth = self.near_depth * (1-self.args.increase_depth_range_by_x_percent)
        far_depth = self.far_depth * (1+self.args.increase_depth_range_by_x_percent)
        depth_range = torch.tensor([near_depth, far_depth], dtype=torch.float32)

        ret = {
            'idx': idx, 
            "rgb_path": rgb_file,
            'image': rgb.permute(2, 0, 1), # torch tensor 3, self.H, self.W
            'intr': render_intrinsics[:3, :3].astype(np.float32),
            'w2c_pose_initial':  render_pose_w2c[:3].astype(np.float32),   # # 3x4, world to camera
            'pose': render_pose_w2c[:3].astype(np.float32), # dummy ground-truth
            "depth_range": depth_range,
            'scene': self.scene
        }
        return ret

