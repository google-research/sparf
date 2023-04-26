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

"""Functions to triangulate (obtain 3D points) based on correspondences estimated by different
matchers, and considering known ground-truth poses. This is not a realistic scenario. 
But this is what is used in DS-NeRF. """


from pathlib import Path
from pathlib import Path
import numpy as np
import imageio 
import os
from easydict import EasyDict as edict
import sys
from pathlib import Path
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional


from source.utils.colmap_initialization import pdcnet_for_hloc as pdcnet_utils
from source.utils.colmap_initialization.sfm import get_poses_and_depths_and_idx, define_pycolmap_camera
from source.utils.colmap_initialization.triangulation_for_hloc import triangulation_from_given_camera_param

sys.path.append(str(Path(__file__).parent / '../../../third_party/Hierarchical-Localization'))
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils.parsers import parse_retrieval


def compute_triangulation_sp_sg(opt: Dict[str, Any], data_dict: Dict[str, Any], save_dir: str):
    """Get triangulation using correspondences predicted by SuperGlue-SuperPoint"""
    # set-up the paths
    outputs = Path(save_dir) / 'init_sfm'
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue_fixed_poses'

    feature_conf = extract_features.confs['superpoint_max']  # ['superpoint_aachen']
    feature_conf['model']['max_keypoints'] = 10000
    # if 'llff' in opt.dataset:
    # use original
    # forces more keypoints
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] = 'indoor'

    # get image list 
    rgb_paths = data_dict.rgb_path
    intrinsics = data_dict.intr[0]  # (3, 3)
    pose_w2c = data_dict.pose  # (N, 3, 3)  # we use the gt poses here!!
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
        # feature_path = extract_features.main_from_data_dict(feature_conf, data_dict, outputs)
        feature_path = extract_features.main(feature_conf, Path(image_dir), outputs)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)


        # run triangulation
        camera_known_intrinsics = define_pycolmap_camera(intrinsics, height=H, width=W)
        try:
            model = triangulation_from_given_camera_param\
                (sfm_dir, image_dir=Path(image_dir), pairs=sfm_pairs, cam=camera_known_intrinsics, 
                features=feature_path, matches=match_path, 
                image_list=image_list, pose_w2c=pose_w2c.cpu().numpy(), verbose=True)
        except:
            print('Reconstruction failed')
            
    poses_w2c_obtained, valid, excluded, colmap_depth_map, colmap_conf_map = \
        get_poses_and_depths_and_idx\
        (sfm_dir, image_list, B, H, W)
    # print(poses_w2c_obtained, pose_w2c)
    # assert poses_w2c_obtained == pose_w2c.cpu().numpy()
    # they are equal 
    return colmap_depth_map, colmap_conf_map


def compute_triangulation_pdcnet(opt: Dict[str, Any], data_dict: Dict[str, Any], save_dir: str):
    """Get triangulation using correspondences predicted by PDC-Net. """
    # set-up the paths
    outputs = Path(save_dir) / 'init_sfm'
    if not os.path.isdir(outputs):
        os.makedirs(outputs)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_pdcnet_fixed_poses'
    save_dir = str(sfm_dir)
    name_file = 'pdcnet_megadepth'

    # cfg
    cfg = edict(pdcnet_utils.default_pdcnet_cfg)
    cfg['path_to_pre_trained_models'] = os.path.dirname(opt.flow_ckpt_path)
    cfg['mask_type_for_pose_estimation'] = 'proba_interval_1_above_50'

    # get image list 
    rgb_paths = data_dict.rgb_path
    intrinsics = data_dict.intr[0]  # (3, 3)
    pose_w2c = data_dict.pose  # (N, 3, 3)  # we use the gt poses here!!
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
                name_to_pair_function='names_to_pair', save_flow_dir=cfg.flow_dir)

            match_path = pdcnet_utils.save_matches_to_h5(matches_dict, pairs, export_dir=save_dir, match_name=name_file, overwrite=cfg.overwrite)
        
        # run triangulation
        camera_known_intrinsics = define_pycolmap_camera(intrinsics, height=H, width=W)
        model = triangulation_from_given_camera_param\
            (sfm_dir, image_dir=Path(image_dir), pairs=sfm_pairs, cam=camera_known_intrinsics, 
            features=feature_path, matches=match_path, 
            image_list=image_list, pose_w2c=pose_w2c.cpu().numpy(), verbose=True)

    poses_w2c_obtained, valid, excluded, colmap_depth_map, colmap_conf_map = \
        get_poses_and_depths_and_idx\
        (sfm_dir, image_list, B, H, W)
    # print(poses_w2c_obtained, pose_w2c)
    # assert poses_w2c_obtained == pose_w2c.cpu().numpy()
    # they are equal 
    return colmap_depth_map, colmap_conf_map