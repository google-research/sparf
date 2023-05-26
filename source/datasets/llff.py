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
import torch.nn.functional as torch_F
import PIL
import imageio
from easydict import EasyDict as edict
import imageio
import cv2
from pathlib import Path
from typing import Any, Dict, Tuple


import source.datasets.base as base
import source.utils.camera as camera


class LLFFPerScene(base.Dataset):
    """LLFF dataloader, modified from BARF. 
    Note that we flip all poses, such that they face towards the +z direction. 
    In the original dataset, they face in -z direction. This is so they face in the same
    direction than the initial identity poses. 
    """
    def __init__(self, args: Dict[str, Any], split: str="train", **kwargs):
        self.raw_H,self.raw_W = 3024, 4032  # size of the original images
        super().__init__(args,split)

        # define path to the scene
        self.path = os.path.join(args.env.llff, args.scene) 
        self.scene = args.scene

        # define path to the images of the scene
        # if want to read the resized images directly
        imgdir_suffix = ''
        if hasattr(self.args, 'llff_img_factor') and self.args.llff_img_factor > 1:
            imgdir_suffix = f'_{self.args.llff_img_factor}'
            factor = self.args.llff_img_factor
        else:
            factor = 1

        self.path_image = "{}/images".format(self.path)  + imgdir_suffix
        image_fnames = [f for f in sorted(os.listdir(self.path_image)) if \
            f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]  # N

        poses_c2w_opengl, bounds = self.parse_cameras_and_bounds(factor=factor)  
        # (N, 3, 4) and (N, 2)
        self.list = list(zip(image_fnames,poses_c2w_opengl,bounds)) 
        
        # here the bounds are actually different for each image!!
        self.bounds = bounds
        self.near = bounds.min() * .9
        self.far = bounds.max() * 1.

        # Select the split.
        # follow standard practise of using 1/8th of the images as test split, 
        # and the rest as training split
        all_indices = np.arange(len(self.list)).astype(np.int32)
        if self.split == 'train':
            indices = all_indices[all_indices % self.args.llffhold != 0]
        else:
            indices = all_indices[all_indices % self.args.llffhold == 0]

        self.list = [self.list[i] for i in indices]
        
        if split == 'train' and self.args.train_sub is not None:
            idx_sub = np.linspace(0, len(self.list) - 1, self.args.train_sub)
            idx_sub = [round(i) for i in idx_sub]
            self.list = [self.list[i] for i in idx_sub]

        if split != 'train' and self.args.val_sub is not None: 
            self.list = self.list[:self.args.val_sub]

        # preload dataset
        if self.args.preload:
            self.images = self.preload_threading(self.get_image)
            self.cameras = self.preload_threading(self.get_camera,data_str="cameras")

    def parse_cameras_and_bounds(self, factor=1):
        """The outputted cameras are in C2W in OpenGL format, i.e. 
        the coordinate system of this function output is [right, up, backwards]. 
        This function was taken from the data loading function from the original NeRF 
        (https://github.com/bmild/nerf/blob/master/load_llff.py#L243).
        """
        fname = "{}/poses_bounds.npy".format(self.path)
        data = torch.tensor(np.load(fname),dtype=torch.float32)

        # parse cameras (intrinsics and poses)
        cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
        poses_c2w_llff = cam_data[...,:4] # [N,3,4]
        
        # Correct rotation matrix ordering to get OpenGL format
        poses_c2w_opengl = poses_c2w_llff.clone()
        poses_c2w_opengl[...,0],poses_c2w_opengl[...,1] = poses_c2w_llff[...,1],-poses_c2w_llff[...,0]

        raw_H,raw_W,self.focal = cam_data[0,:,-1]
        
        assert raw_H == self.raw_H and raw_W == self.raw_W

        # adjust the intrinsics to correspond to the down-sampled image (with this factor)
        self.raw_H /= float(factor)
        self.raw_W /= float(factor)
        self.focal /= float(factor)

        # parse depth bounds
        bounds = data[:,-2:] # [N,2]
        scale = 1./(bounds.min()*0.75) # not sure how this was determined

        # rescale the poses
        poses_c2w_opengl[...,3] *= scale
        bounds *= scale
    
        # roughly center camera poses
        poses_c2w_opengl = self.center_camera_poses(poses_c2w_opengl)  
        return poses_c2w_opengl, bounds

    def center_camera_poses(self, poses: torch.Tensor):
        # compute average pose
        center = poses[...,3].mean(dim=0)
        v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
        v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
        return poses

    def get_all_camera_poses(self, args):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(p) for p in pose_raw_all],dim=0)
        return pose_all

    def __getitem__(self,idx: int) -> Dict[str, Any]:
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

        """
        sample = dict(idx=idx)
        image = self.images[idx] if self.args.preload else self.get_image(idx)  # PIL image, HxWx3 

        depth_range = torch.tensor([self.near, self.far])  
        # the one used in nerf pytorch
        intr,pose_w2c = self.cameras[idx] if self.args.preload else self.get_camera(idx)
        image, intr = self.preprocess_image_and_intrinsics(image, intr=intr) 

        sample.update(
            rgb_path=self.list[idx][0],
            scene=self.scene,
            depth_range=depth_range, 
            image=image,  # torch tensor (3, self.H, self.W)
            intr=intr, # numpy array (3, 3), corresponds to the processed image
            pose=pose_w2c, # numpy array  (3, 4), world to camera
        )
        return sample

    def get_image(self, idx: int):
        image_fname = "{}/{}".format(self.path_image,self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) 
        # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self, idx: int):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_c2w_opengl = self.list[idx][1]
        pose_w2c_opencv = self.parse_raw_camera(pose_c2w_opengl)
        return intr, pose_w2c_opencv

    def parse_raw_camera(self, pose_c2w_opengl: torch.Tensor):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))

        # Transforms OpenGL to OpenCV
        pose_c2w_opencv = camera.pose.compose([pose_flip,pose_c2w_opengl[:3]])

        # Gets W2C from C2W matrix
        pose_w2c_opencv = camera.pose.invert(pose_c2w_opencv)

        # Right now, the poses are facing in direction -Z. We want them to face direction +Z. 
        # That amounts to a rotation of 180 degrees around the x axis, ie the same pose_flip. 
        # Therefore, when initializing with identity matrix, the identity and ground-truth 
        # poses will face in the same direction. 
        pose_w2c_opencv_facing_plus_z = camera.pose.compose([pose_flip,pose_w2c_opencv])
        return pose_w2c_opencv_facing_plus_z

