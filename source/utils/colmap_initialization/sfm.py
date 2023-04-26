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

"""Functions to get the poses using COLMAP with different matchers. """

from pathlib import Path
from pathlib import Path
import numpy as np
import imageio 
import os
import torch
import shutil
from easydict import EasyDict as edict
import sys
import cv2
import pycolmap
from pathlib import Path
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional

import source.utils.colmap_initialization.pdcnet_for_hloc as pdcnet_utils
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img
from third_party.colmap_read_write_model import read_images_binary_to_poses, read_images_binary, read_points3D_binary
from source.utils.colmap_initialization.reconstruction_know_intrinsics_for_hloc import reconstruction_w_known_intrinsics

sys.path.append(str(Path(__file__).parent / '../../../third_party/Hierarchical-Localization'))
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils.parsers import parse_retrieval





mapper_options = {
'ba_refine_focal_length': False,# arg (=1)
'ba_refine_principal_point': False, # arg (=0)
'ba_refine_extra_params': False, # (=1)
'min_num_matches': 5,  # arg (=15)
'ba_local_max_num_iterations': 25, # arg (=25)
'ba_global_max_num_iterations': 50, # arg (=50)
# 'init_min_tri_angle': 4, 

# 'ransac': 
# 'init_min_num_inliers': 25, # arg (=100)
# 'init_max_error': 4, # arg (=4)
# 'abs_pose_max_error': 4, # arg (=12)   # because we use much smaller images 
# 'abs_pose_min_num_inliers': 5, # arg (=30)
# 'abs_pose_min_inlier_ratio': 0.01, # arg (=0.25)
}

def define_pycolmap_camera(K, height, width): 
    """To use in HLOC. """
    focal_length = K[0, 0]
    cx=K[0, 2]
    cy=K[1, 2]
    camera = pycolmap.Camera(model='SIMPLE_PINHOLE', width=width, 
                             height=height, params=[focal_length, cx, cy])
    return camera


def _load_colmap_depth(basedir: str, factor: float, height: int, width: int, scale: float):
    """Creates depth maps from the 3D points obtained with COLMAP, for each image. 
    Also generates the corresponding weight maps, using the reprojection error of
    the 3D points. 

    Args:
        basedir (str): Path to directory where the outputted file of COLMAP are, i.e
                       images.bin and points3D.bin
        factor (float): scaling factor to apply to the 3D points
        height (int), width (int): dimensions of the outputted depth maps
        scale (float): scaling factor to apply to the created depth maps. 

    Returns:
        depth_dict: dictionary containing for each image the depth map, in shape (H, W)
        weight_dict: dictionary containing for each image the confidence of the depth, 
                     as the inverse of the reprojection error. Shape is (H, W)
    """

    images = read_images_binary(
        os.path.join(basedir, 'images.bin'))  # 
    points = read_points3D_binary(
        os.path.join(basedir, 'points3D.bin'))


    depth_dict, weight_dict = {}, {}

    for id_im, image in images.items():
        name = image.name
        errors = []
        for i in range(len(image.xys)):
            id_3d = image.point3D_ids[i]
            if id_3d == -1:
                continue
            errors.append(points[id_3d].error)
    err_mean = np.mean(errors)
    print('Mean reprj. error', err_mean)

    for id_im, image in images.items():
        name = image.name
        rot = image.qvec2rotmat()
        t = image.tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([rot, t], 1), bottom], 0)
        pose_c2w = np.linalg.inv(w2c)

        depth_image = np.zeros((height, width), dtype=np.float32)
        weight_image = np.zeros((height, width), dtype=np.float32)

        x_vals = []
        y_vals = []

        points_2d = []
        points_3d = []
        errors = []

        for i in range(len(image.xys)):
            id_3d = image.point3D_ids[i]
            if id_3d == -1:
                continue

            coord = image.xys[i] / factor
            x, y = np.round(coord).astype(np.int32)
            if y >= height or y < 0 or x >= width or x < 0:
                continue
            x_vals.append(x)
            y_vals.append(y)

            points_3d.append(points[id_3d].xyz)
            points_2d.append(image.xys[i])
            errors.append(points[id_3d].error)

    if len(points_3d) > 0:
        depths = (pose_c2w[:3, 2].T
                @ (np.asarray(points_3d) - pose_c2w[:3, 3]).T).T
        weights = 2 * np.exp(-(errors / err_mean)**2)
        
        print('Found {} colmap depth points'.format(len(y_vals)))
        depth_image[tuple(y_vals), tuple(x_vals)] = depths * scale
        weight_image[tuple(y_vals), tuple(x_vals)] = weights
    depth_dict[name] = depth_image
    weight_dict[name] = weight_image

    return depth_dict, weight_dict


def verify_colmap_depth(data_dict: Dict[str, Any], 
                        poses_w2c: torch.Tensor, index_0: int=0, index_1: int=1):
    """Function to verify the depth from COLMAP is valid. We do this by plotting the
    correspondences (created with the depth map and the w2c image poses. """
    colmap_depth = data_dict.colmap_depth.to(poses_w2c.device)
    # pose_w2c_ = data_dict.pose_w2c
    poses_w2c_ = torch.eye(4).unsqueeze(0).repeat(poses_w2c.shape[0], 1, 1).to(poses_w2c.device)
    poses_w2c_[:, :3] = poses_w2c
    intr = data_dict.intr.to(poses_w2c.device)

    images = data_dict.image.permute(0, 2, 3, 1)
    images_numpy = (images.cpu().numpy() * 255).astype(np.uint8)

    relative_0_to_1 = poses_w2c_[index_1] @ pose_inverse_4x4(poses_w2c_)[index_0]

    pts_valid_depth_in_0_h, pts_valid_depth_in_0_w = torch.where(colmap_depth[index_0] > 1e-6)
    pixels_in_0 = torch.stack((pts_valid_depth_in_0_w, pts_valid_depth_in_0_h), dim=-1).float()  # (N, 2)
    depths_in_0 = colmap_depth[index_0][pts_valid_depth_in_0_h, pts_valid_depth_in_0_w]  # (N)

    pixels_in_1 = batch_project_to_other_img(pixels_in_0.unsqueeze(0), di=depths_in_0.reshape(1, -1), 
                                             Ki=intr[index_0].unsqueeze(0), Kj=intr[index_1].unsqueeze(0), 
                                             T_itoj=relative_0_to_1.unsqueeze(0))
    pixels_in_1 = pixels_in_1.reshape(-1, 2)
    image_plot = make_matching_plot_fast(
        images_numpy[index_0], images_numpy[index_1], 
        kpts0=pixels_in_0.cpu().numpy().astype(np.int64)[::100], 
        kpts1=pixels_in_1.cpu().numpy().astype(np.int64)[::100], 
        color=None, 
        path='/home/jupyter/shared/colmap_matches_{}_to_{}.png'.format(index_0, index_1))
    return 


def get_poses_and_depths_and_idx(sfm_dir: str, image_list: List[str], B: int, H: int, W: int):
    """Reads from the files outputted by COLMAP and outputs the initial poses, 
    valid indices and the depth maps. 

    Args:
        sfm_dir (str):  Path to directory where the outputted file of COLMAP are, i.e
                        images.bin and points3D.bin 
        image_list (list): list of image names, to make sure the outputted poses are
                           in the same order than the list of images. 
        B (int): number of images
        H (int), W (int): Dimension of the images (to use in the depth maps)

    Returns:
        initial_poses_w2c: The world-to-camera poses obtained by COLMAP (opencv/colmap format)
                            Shape is (B, 4, 4)
        valid_poses_idx: List of bools, indicating is a pose was found
        excluded_poses_idx: List of bools, indicating indices for which no pose was found
        cm_depth_map: Depth maps created from the 3D points obtained by COLMAP. 
                      torch.Tensor of shape (B, H, W)
        cm_conf_map: Confidence maps created from the reprojection error. 
                     torch.Tensor of shape (B, H, W)
    """
    valid_poses_idx, excluded_poses_idx = [], []
    if os.path.exists(str(sfm_dir / 'images.bin')):
        dict_images_to_pose = read_images_binary_to_poses(sfm_dir / 'images.bin')
        dict_depth_maps, dict_conf_maps = _load_colmap_depth(str(sfm_dir), 1., H, W, scale=1.)

        initial_poses_w2c = []
        cm_depth_map, cm_conf_map = [], []

        for i, image_name in enumerate(image_list):
            if image_name in dict_images_to_pose:
                initial_poses_w2c.append(dict_images_to_pose[image_name])
                cm_depth_map.append(dict_depth_maps[image_name])
                cm_conf_map.append(dict_conf_maps[image_name])
                valid_poses_idx.append(i)
            else:
                initial_poses_w2c.append(np.eye(4))
                cm_depth_map.append(np.zeros((H, W), np.float32))
                cm_conf_map.append(np.zeros((H, W), np.float32))

                excluded_poses_idx.append(i)
        initial_poses_w2c = np.stack(initial_poses_w2c, axis=0)
        initial_poses_w2c = torch.from_numpy(initial_poses_w2c)
        cm_depth_map = np.stack(cm_depth_map, axis=0)
        cm_conf_map = np.stack(cm_conf_map, axis=0)
    else:
        print('Reconstruction failed')
        excluded_poses_idx = np.arange(start=0, stop=len(image_list), step=1).tolist()
        # they are all identity!! 
        initial_poses_w2c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
        cm_depth_map = np.zeros((B, H, W), dtype=np.float32)
        cm_conf_map = np.zeros((B, H, W), dtype=np.float32)
    return initial_poses_w2c, valid_poses_idx, excluded_poses_idx, \
        torch.from_numpy(cm_depth_map), torch.from_numpy(cm_conf_map)


def get_poses_and_idx(sfm_dir: str, image_list: List[str], B: int, H: int, W: int):
    """Reads from the files outputted by COLMAP and outputs the initial poses, 
    valid indices. 

    Args:
        sfm_dir (str):  Path to directory where the outputted file of COLMAP are, i.e
                        images.bin and points3D.bin 
        image_list (list): list of image names, to make sure the outputted poses are
                           in the same order than the list of images. 
        B (int): number of images
        H (int), W (int): Dimension of the images (to use in the depth maps)

    Returns:
        initial_poses_w2c: The world-to-camera poses obtained by COLMAP (opencv/colmap format)
                            Shape is (B, 4, 4)
        valid_poses_idx: List of bools, indicating is a pose was found
        excluded_poses_idx: List of bools, indicating indices for which no pose was found
    """
    valid_poses_idx, excluded_poses_idx = [], []
    if os.path.exists(str(sfm_dir / 'images.bin')):
        dict_images_to_pose = read_images_binary_to_poses(sfm_dir / 'images.bin')
        initial_poses_w2c = []

        for i, image_name in enumerate(image_list):
            if image_name in dict_images_to_pose:
                initial_poses_w2c.append(dict_images_to_pose[image_name])
                valid_poses_idx.append(i)
            else:
                initial_poses_w2c.append(np.eye(4))
                excluded_poses_idx.append(i)
        initial_poses_w2c = np.stack(initial_poses_w2c, axis=0)
        initial_poses_w2c = torch.from_numpy(initial_poses_w2c)
    else:
        print('Reconstruction failed')
        excluded_poses_idx = np.arange(start=0, stop=len(image_list), step=1).tolist() # all identity
        # valid_poses_idx = np.arange(start=0, stop=len(image_list), step=1).tolist()
        initial_poses_w2c = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    return initial_poses_w2c, valid_poses_idx, excluded_poses_idx


def compute_sfm_inloc(opt: Dict[str, Any], data_dict: Dict[str, Any], 
                      save_dir: str, load_colmap_depth: bool=False):
    """COLMAP with SuperPoint-SuperGlue. That is the default hloc. """

    outputs = Path(save_dir) / 'init_sfm'
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    feature_conf = extract_features.confs['superpoint_max']  
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] =  'outdoor'

    # get image list 
    rgb_paths = data_dict.rgb_path
    intrinsics = data_dict.intr[0]  # (3, 3)
    image_list = [os.path.basename(path) for path in rgb_paths]
    images = (data_dict.image.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    B, H, W = images.shape[:3]
    
    overwrite = False

    if not os.path.exists(str(sfm_dir / 'images.bin')) or overwrite:
        # create a fake image dir and save the images there
        image_dir = os.path.join(outputs, 'images')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        for image_id, image_name in enumerate(image_list):
            imageio.imwrite(os.path.join(image_dir, image_name), images[image_id])

        # list of all iamge pairs 
        pairs_from_exhaustive.main(output=sfm_pairs, image_list=image_list)

        # Extract and match local features
        feature_path = extract_features.main(feature_conf, Path(image_dir), outputs, 
                                             overwrite=overwrite)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], 
                                         outputs, overwrite=overwrite)

        camera_known_intrinsics = define_pycolmap_camera(intrinsics, height=H, width=W)
        model = reconstruction_w_known_intrinsics\
            (sfm_dir, Path(image_dir), sfm_pairs, feature_path, match_path, 
            camera_mode='SINGLE', cam=camera_known_intrinsics, image_list=image_list, verbose=False, verbose_features=True,   
            mapper_options=mapper_options)

    if load_colmap_depth:
        return get_poses_and_depths_and_idx(sfm_dir, image_list, B, H, W)
    return get_poses_and_idx(sfm_dir, image_list, B, H, W)


def compute_sfm_pdcnet(opt: Dict[str, Any], data_dict: Dict[str, Any], 
                       save_dir: str, load_colmap_depth: bool=False):
    """Run COLMAP with PDC-Net. """
    # set-up the paths
    outputs = Path(save_dir) / 'init_sfm'
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_pdcnet'
    save_dir = str(sfm_dir)
    name_file = 'pdcnet_megadepth'

    # cfg
    cfg = edict(pdcnet_utils.default_pdcnet_cfg)
    cfg['path_to_pre_trained_models'] = os.path.dirname(opt.flow_ckpt_path)
    #'/home/jupyter/shared/pre_trained_models'

    # get image list 
    rgb_paths = data_dict.rgb_path
    intrinsics = data_dict.intr[0]  # (3, 3)
    image_list = [os.path.basename(path) for path in rgb_paths]
    images = (data_dict.image.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    B, H, W = images.shape[:3]
    
    overwrite = False
    if not os.path.exists(str(sfm_dir / 'images.bin')) or overwrite:
        # create a fake image dir and save the images there
        image_dir = os.path.join(outputs, 'images')
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        for image_id, image_name in enumerate(image_list):
            imageio.imwrite(os.path.join(image_dir, image_name), images[image_id])
        
        # get name of the images
        rgb_paths = data_dict.rgb_path
        images_name = [os.path.basename(path) for path in rgb_paths]
        
        # list of all iamge pairs 
        pairs_from_exhaustive.main(output=sfm_pairs, image_list=image_list)
        # saved in path at sfm_pairs
        pairs = parse_retrieval(sfm_pairs)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]

        # Extract and match local features
        # extract keypoints from all images at original resolution
        keypoint_extractor_module = pdcnet_utils.KeypointExtractor(cfg=cfg)
        feature_path = pdcnet_utils.extract_keypoints_and_save_h5(
            keypoint_extractor_module=keypoint_extractor_module, images_dir=image_dir, image_names=image_list,
            export_dir=save_dir, name=name_file, cfg=cfg, overwrite=cfg.overwrite)

        match_path = Path(os.path.join(save_dir, name_file + '.h5'))
        if not os.path.exists(str(match_path)) or cfg.overwrite:
            matches_dict = pdcnet_utils.retrieve_matches_at_keypoints_locations_from_pair_list(
                cfg, cfg, pair_names=pairs, images_dir=image_dir,
                path_to_h5_keypoints=os.path.join(save_dir, name_file + '_keypoints.h5'), key_for_keypoints='keypoints',
                name_to_pair_function='names_to_pair')

            match_path = pdcnet_utils.save_matches_to_h5(matches_dict, pairs, export_dir=save_dir, match_name=name_file, overwrite=cfg.overwrite)
        
        #3D reconstruction
        #Run COLMAP on the features and matches.
        camera_known_intrinsics = define_pycolmap_camera(intrinsics, height=H, width=W)
        model = reconstruction_w_known_intrinsics\
            (sfm_dir, Path(image_dir), sfm_pairs, feature_path, match_path, 
            camera_mode='SINGLE', cam=camera_known_intrinsics, image_list=image_list, verbose=True,  
            mapper_options=mapper_options)

    if load_colmap_depth:
        return get_poses_and_depths_and_idx(sfm_dir, image_list, B, H, W)
    return get_poses_and_idx(sfm_dir, image_list, B, H, W)
