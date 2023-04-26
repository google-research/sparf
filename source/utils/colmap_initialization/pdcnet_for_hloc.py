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

""" To use PDC-Net within COLMAP, in  the HLOC pipeline. Since PDC-Net outputs
dense matches, we need to create fake keypoints on a grid. """
import torch
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm
import importlib
import imageio
import cv2
import os
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional
torch.set_grad_enabled(False)

from pathlib import Path

env_path = str(Path(__file__).parent /  '../../../third_party/DenseMatching')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
from model_selection import select_model
from validation.utils import (resize_image, matches_from_flow)


def pad_to_same_shape(im1: np.ndarray, im2: np.ndarray):
    # pad to same shape
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)
    shape = im1.shape
    return im1, im2


# ----------------- NAMING FUNCTIONS --------------------------------
def names_to_pair(name0: str, name1: str):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_imc(img_fname0: str, img_fname1: str):
    name0 = os.path.splitext(os.path.basename(img_fname0))[0]
    name1 = os.path.splitext(os.path.basename(img_fname1))[0]
    key = '{}-{}'.format(name0, name1)  # must always be in that specific order
    return key


def name_to_keypoint_imc(img_fname: str):
    return os.path.splitext(os.path.basename(img_fname))[0]


def save_matches_to_h5(matches_dict: Dict[str, Any], 
                       pair_names: List[str], 
                       export_dir: str, 
                       match_name: str, 
                       overwrite: bool=False):
    """Save the matches to h5 file, in the format wanted by HLOC (modified by us). """
    match_path = os.path.join(export_dir, match_name + '.h5')
    if os.path.exists(match_path) and not overwrite:
        return Path(match_path)
    match_file = h5py.File(match_path, 'w')
    pbar = tqdm(enumerate(pair_names), total=len(pair_names))
    for i, pair in pbar:
        name0, name1 = pair
        name_of_pair = names_to_pair(name0, name1)
        grp = match_file.create_group(name_of_pair)
        matches = matches_dict[name_of_pair][:, :2].astype(np.int32)
        # we only want the indexes, not the confidence score
        grp.create_dataset('matches', data=matches) 
        
        # different from the matches0 used in sparse matching
        # here already are the indices of the source and target matching keypoints
        # grp.create_dataset('matching_scores0', data=np.ones_like(matches[:, 0:1]).astype(np.float32))

    match_file.close()
    print('Finished exporting matches.')
    return Path(match_path)

def remove_duplicates(matches: np.ndarray) -> np.ndarray:
    """
    Args:
        matches: Nx2 or Nx3, contains index of keypoints
    Returns:
        matches
    """
    final_matches = []
    kps_final_matches = []
    if matches.shape[0] == 0:
        return matches
    else:
        for i in matches.tolist():
            # take only coordinates, ignore score if there is one
            i_ = i[:2]
            if i not in kps_final_matches:
                kps_final_matches.append(i_)
                final_matches.append(i)
        matches = np.array(final_matches)
        return matches

# ------------------ KEYPOINTS EXTRACTION FUNCTIONS ---------------------------------------
def get_grid_keypoints(data: Dict[str, Any], cfg: Dict[str, Any]):
    hA, wA = data['size_original']
    scaling_kp = cfg.scaling_kp

    size_of_keypoints_s = np.int32([hA // scaling_kp, wA // scaling_kp])

    # creates all the keypoint from the original image ==> thats in resolution resized // 4 or not,
    # at each keypoitn location.
    # dense grip at pixel location
    XA, YA = np.meshgrid(np.linspace(0, size_of_keypoints_s[1] - 1, size_of_keypoints_s[1]),
                         np.linspace(0, size_of_keypoints_s[0] - 1, size_of_keypoints_s[0]))

    YA = YA.flatten()
    XA = XA.flatten()

    # put them in dimension of original image scaling
    YA = YA / float(size_of_keypoints_s[0]) * float(hA)
    XA = XA / float(size_of_keypoints_s[1]) * float(wA)

    XA = np.round(XA.reshape(-1, 1), 2)
    YA = np.round(YA.reshape(-1, 1), 2)

    # give keypoint.
    keypoints_A = np.concatenate((XA, YA), axis=1)
    return keypoints_A


class KeypointExtractor:
    """
    Class responsible for extracting keypoints from an image.
    """
    def __init__(self, cfg):
        self.extractor_name = cfg.keypoint_extractor

        assert cfg.keypoint_extractor == 'dense_grid'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_keypoints(self, image: np.ndarray, cfg: Dict[str, Any]):
        if self.extractor_name == 'dense_grid':
            kp = get_grid_keypoints({'size_original': image.shape[:2]}, cfg)
        else:
            h, w = image.shape[:2]
            image_resized, scales0 = resize_image(
                image, self.device, 0.0, min_size=cfg.min_size,
                keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)
            h_resized, w_resized = image_resized.shape[:2]
            # extract  keypoints at resized resolution
            kp, _ = self.extractor_model.find_and_describe_keypoints(image_resized)

            # rescale them to original resolution
            kp[:, 0] *= float(w) / float(w_resized)
            kp[:, 1] *= float(h) / float(h_resized)
        return kp


def extract_keypoints_and_save_h5(keypoint_extractor_module: Callable[[Any], Any], 
                                  images_dir: str, image_names: List[str], 
                                  export_dir: str, name: str, 
                                  cfg: Dict[str, Any], overwrite: bool=False) -> str:
    """Extract keypoints as a grid (to simulate sparse keupoint extraction) and 
    save them in h5 file, to fit the format wanted by HLOC. """
    feature_path = Path(export_dir, name + '_keypoints.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    if os.path.exists(str(feature_path)) and not overwrite:
        return feature_path


    feature_file = h5py.File(str(feature_path), 'w')

    print('Compute keypoints over a grid')
    pbar = tqdm(enumerate(image_names), total=len(image_names))
    for i, image_name in pbar:
        image = imageio.imread(os.path.join(images_dir, image_name))
        kp = keypoint_extractor_module.get_keypoints(image, cfg=cfg)
        grp = feature_file.create_group(image_name)
        grp.create_dataset('keypoints', data=kp)
    feature_file.close()
    return feature_path


def extract_keypoints_from_image_list(keypoint_extractor_module: Callable[[Any], Any], 
                                      images_dir: str, image_names: List[str], 
                                      cfg: Dict[str, Any], 
                                      name_to_keypoint_function: str=None):
    """Extract grid keypoints for each image in a list"""
    kp_dict = {}
    print('Compute keypoints over a grid')
    pbar = tqdm(enumerate(image_names), total=len(image_names))
    for i, image_name in pbar:
        image = imageio.imread(os.path.join(images_dir, image_name))
        kp = keypoint_extractor_module.get_keypoints(image, cfg=cfg)

        name_of_keypoint = image_name
        if name_to_keypoint_function is not None:
            expr_module = importlib.import_module('validation.compute_matches_at_sparse_keypoints_from_pair_list')
            name_of_keypoint = getattr(expr_module, name_to_keypoint_function)(image_name)
        kp_dict[name_of_keypoint] = kp
    return kp_dict



# ----------------------- PREPROCESSING -------------------------------------------------------------
def get_image_pair_info_(source: np.ndarray, target: np.ndarray, cfg: Dict[str, Any]):
    """
    Resize and process the images as required in config.
    Args:
        source: numpy array HxWx3
        target: numpy array HxWx3
        cfg: config, must contain fields 'keep_original_image_when_smaller_reso' and 'min_size'
    Returns:
        data: dictionary with fields 'source' and 'target'. Each is a dictionary with fields
        'image_original', 'size_original', 'image_resized', 'size_resized',
        'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_resized, scales0 = resize_image(
        source, device, 0.0, min_size=cfg.min_size,
        keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)
    target_resized, scales1 = resize_image(
        target, device, 0.0, min_size=cfg.min_size,
        keep_original_image_when_smaller_reso=cfg.keep_original_image_when_smaller_reso)

    source_numpy, target_numpy = pad_to_same_shape(source_resized, target_resized)
    source_torch = torch.Tensor(source_numpy).permute(2, 0, 1).unsqueeze(0)
    target_torch = torch.Tensor(target_numpy).permute(2, 0, 1).unsqueeze(0)

    data_source = {'image_original': source, 'size_original': source.shape[:2],
                   'image_resized': source_resized, 'size_resized': source_resized.shape[:2],
                   'image_resized_padded': source_numpy, 'size_resized_padded': source_numpy.shape[:2],
                   'image_resized_padded_torch': source_torch}
    data_target = {'image_original': target, 'size_original': target.shape[:2],
                   'image_resized': target_resized, 'size_resized': target_resized.shape[:2],
                   'image_resized_padded': target_numpy, 'size_resized_padded': target_numpy.shape[:2],
                   'image_resized_padded_torch': target_torch}
    data = {'source': data_source, 'target': data_target}
    return data


def get_matches_at_keypoints_dense_grid(flow: torch.Tensor, mask: torch.Tensor, 
                                        data_source: Dict[str, Any], data_target: Dict[str, Any], 
                                        cfg: Dict[str, Any], name_image: str, 
                                        confidence_map: torch.Tensor=None):
    """
    From flow and mask relating the target to the source, get the matches in the form of index of corresponding
    keypoints. The keypoints were previously created densely in a grid of a specific size defined in the config.
    Args:
        flow: torch tensor of size (b, 2, h, w)
        mask: torch tensor of size (b, h, w)
        data_source: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        data_target: dict with keys 'image_original', 'size_original', 'image_resized', 'size_resized',
                     'image_resized_padded', 'size_resized_padded', 'image_resized_padded_torch'
        cfg: config, default is
            cfg = {'estimate_at_quarter_resolution': True,
                   'apply_ransac_both_ways': False, 'ransac_type': 'magsac', 'ransac_inlier_ratio': 1.0,
                   'min_nbr_matches': 30, 'min_nbr_matches_after_ransac': 15, 'scaling_kp': 2
           }
        name_image:
        confidence_map: torch tensor of size b, h, w

    Returns:
        matches: if confidence_map is None:
                     Nx2 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively
                 else:
                     Nx3 for N matches, contains the index of the corresponding keypoints in source and target
                     image respectively, and the confidence score
    """

    hA, wA = data_source['size_original']
    hB, wB = data_target['size_original']

    Ish, Isw = data_source['size_resized']
    Ith, Itw = data_target['size_resized']
    scaling_kp = cfg.scaling_kp

    size_of_keypoints_s = np.int32([hA // scaling_kp, wA // scaling_kp])
    size_of_keypoints_t = np.int32([hB // scaling_kp, wB // scaling_kp])

    # get the actual keypoints from the match
    # here correct because we made sure that resized image was dividable by 4
    if cfg.estimate_at_quarter_resolution:
        scaling = 4.0
        # if the images were not dividable by 4.0, scaling might not be exactly 4.0
        # that was the actual output of the network
        size_of_flow_padded = [h // scaling for h in data_target['size_resized_padded']]
        scaling_for_keypoints = np.float32(data_target['size_resized_padded'])[::-1] / \
                            np.float32(size_of_flow_padded)[::-1]
    else:
        scaling_for_keypoints = [1.0, 1.0]

    pA, pB = matches_from_flow(flow, mask, scaling=1.0)
    if pA.shape[0] < cfg.min_nbr_matches:
        # not enough matches
        matches = np.empty([0, 3], dtype=np.float32) if confidence_map is not None else np.empty([0, 2], dtype=np.int32)
    else:

        XA_match, YA_match = pA[:, 0].copy(), pA[:, 1].copy()
        XB_match, YB_match = pB[:, 0].copy(), pB[:, 1].copy()

        # scale those to size_of_keypoints
        XA_match = XA_match * size_of_keypoints_s[1] / float(Isw) * scaling_for_keypoints[0]
        YA_match = YA_match * size_of_keypoints_s[0] / float(Ish) * scaling_for_keypoints[1]
        XB_match = XB_match * size_of_keypoints_t[1] / float(Itw) * scaling_for_keypoints[0]
        YB_match = YB_match * size_of_keypoints_t[0] / float(Ith) * scaling_for_keypoints[1]

        XA_match = np.int32(np.round(XA_match))
        YA_match = np.int32(np.round(YA_match))
        XB_match = np.int32(np.round(XB_match))
        YB_match = np.int32(np.round(YB_match))

        idx_A = (YA_match * size_of_keypoints_s[1] + XA_match).reshape(-1)
        idx_B = (YB_match * size_of_keypoints_t[1] + XB_match).reshape(-1)

        #assert (keypoints_source[idx_A, 0] - XA_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_s[1]) * float(wA)), 2).sum())
        #assert (keypoints_source[idx_A, 1] - YA_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_s[0]) * float(hA)), 2).sum())
        #assert (keypoints_target[idx_B, 0] - XB_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_t[1]) * float(wB)), 2).sum())
        #assert (keypoints_target[idx_B, 1] - YB_match.astype(np.float32) > 0).sum() == 0 #/ float(size_of_keypoints_t[0]) * float(hB)), 2).sum())

        matches_list = np.concatenate((idx_A.reshape(-1, 1), idx_B.reshape(-1, 1)), axis=1)
        if confidence_map is not None:
            scores = confidence_map.squeeze()[pB[:, 1], pB[:, 0]].cpu().numpy()
            matches_list = np.concatenate((matches_list, scores.reshape(-1, 1)), axis=1)
        matches = np.asarray(matches_list)

    # matches here are Nx2 or Nx3
    assert matches.shape[1] == 2 or matches.shape[1] == 3
    return matches




# ------------------ FINAL FUNCTIONS FOR GETTING MATCHES FROM IMAGES AND KEYPOINTS ------------------------
def retrieve_matches_at_keypoints_locations_from_pair_list(args: Dict[str, Any], cfg: Dict[str, Any], 
                                                           pair_names: List[str], images_dir: str, 
                                                           name_to_pair_function: Callable[[Any], Any],
                                                           name_to_keypoint_function: str=None, 
                                                           kp_dict: Dict[str, Any]=None,
                                                           path_to_h5_keypoints: str=None, 
                                                           key_for_keypoints: str=None,
                                                           matches_dict: Dict[str, Any]={}):
    """
    Retrieves matches between each image pair specificied in a list of image pairs, with prior keypoints extracted
    densely in a grid for each image.
    Each match has shape 1x2, it contains the index of the corresponding keypoints in source and target
    images respectively. It can also contain the confidence of the match, in that case the match is 1x3.

    Args:
        args:
        cfg: config, check default_cfg
        pair_names: list of pair names
        images_dir:
        name_to_pair_function: function to convert image pair names to key for matches h5 file
        name_to_keypoint_function: function to convert image image names to key for keypoint h5 file
        kp_dict: dictionary containing keypoints for each image
        path_to_h5_keypoints: path to h5 file containing keypoints for each imahe
        key_for_keypoints: additional keys to access keypoint in kp_dict, when applicable
        matches_dict: dictionary containing matches
        save_flow_dir:
        save_plots_dir:

    Returns:
        matches_dict: dictionary containing matches, where there is a key for each image pair,
                      defined by the name_to_pair_function.
                      for each pair, Nx2 for N matches, contains the index of the corresponding keypoints
                      in source and target image respectively.
                      If a confidence value is available, Nx3, where the third value is the confidence of the match.
    """
    if not args.local_optim_iter:
        local_optim_iter = args.optim_iter
    else:
        local_optim_iter = int(args.local_optim_iter)

    if path_to_h5_keypoints is not None:
        assert os.path.exists(path_to_h5_keypoints)
        kp_dict = h5py.File(path_to_h5_keypoints, 'r')

    segNet = None
    print(args)
    if args.homo:
        args.network_type = 'PDCNet'
        inference_parameters = {'confidence_map_R': 1.0, 'ransac_thresh': 1.0, 'multi_stage_type': 'h',
                                'mask_type': 'proba_interval_1_above_5',
                                # for multi-scale
                                'homography_visibility_mask': True,
                                'scaling_factors': [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
                                # [1, 0.5, 0.88, 1.0 / 0.5, 1.0 / 0.88],
                                'min_inlier_threshold_for_multi_scale': 0.2,
                                'min_nbr_points_for_multi_scale': 70,
                                'compute_cyclic_consistency_error': False}
        args.update(inference_parameters)
    network, estimate_uncertainty = select_model(
        args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
        path_to_pre_trained_models=args.path_to_pre_trained_models)

    pbar = tqdm(enumerate(pair_names), total=len(pair_names))
    for i, pair in pbar:
        img_fname0, img_fname1 = pair
        # read the images and feed them to the model
        src_fn = os.path.join(images_dir, img_fname0)
        tgt_fn = os.path.join(images_dir, img_fname1)
        name_of_pair = names_to_pair(img_fname0, img_fname1)
        name_of_pair_for_flow = name_of_pair.replace('/', '-').replace(' ', '--')

        name0 = img_fname0
        name1 = img_fname1

        if name_of_pair in list(matches_dict.keys()):
            continue

        image0_original = imageio.imread(src_fn)[:, :, :3]
        image1_original = imageio.imread(tgt_fn)[:, :, :3]

        # get the keypoints
        if key_for_keypoints is None:
            keypoints_A = kp_dict[name0].__array__()
            keypoints_B = kp_dict[name1].__array__()
        else:
            keypoints_A = kp_dict[name0][key_for_keypoints].__array__()
            keypoints_B = kp_dict[name1][key_for_keypoints].__array__()
        data = get_image_pair_info_(source=image0_original, target=image1_original, cfg=cfg)

        # do the matching
        confidence_map_from_1_to_0 = None
        flow_from_1_to_0, confidence_map_from_1_to_0, mask_from_1_to_0 = network.perform_matching(
            data['source'], data['target'], cfg, segNet=segNet)


        if cfg.keypoint_extractor == 'dense_grid':
            # get keypoints and matches
            matches_0_1 = get_matches_at_keypoints_dense_grid(
                flow_from_1_to_0, mask_from_1_to_0, data_source=data['source'],
                data_target=data['target'], cfg=cfg, 
                name_image='{}_forward'.format(name_of_pair_for_flow), 
                confidence_map=confidence_map_from_1_to_0)
        else:
            raise ValueError

        matches = matches_0_1
        matches = remove_duplicates(matches)  # .astype(np.int32), now there might be confidence value too
        matches_dict[name_of_pair] = matches
    return matches_dict

# cfg
default_pdcnet_cfg = {
    'model': 'PDCNet', 
    'homo': False, 
    'optim_iter': 3, 
    'local_optim_iter': 3, 
    'path_to_pre_trained_models': '/home/jupyter/shared/pre_trained_models', 
    'pre_trained_model': 'megadepth', 
    'network_type': None, 
    
    # keypoints
    'keypoint_extractor': 'dense_grid', 

    # matching
    'keep_original_image_when_smaller_reso': True, 'min_size': 480,
    'compute_matching_both_ways': False,
    'estimate_at_quarter_resolution': True, 'use_segnet': False,
    'segnet_pretrained_dir': '',
    'mask_type_for_pose_estimation': 'proba_interval_1_above_10', 
    'min_nbr_matches': 500, 

    'scaling_kp': 2, 'keypoint_nms': 4.,  
    'final_matches_type': 'reunion',  
    'overwrite': False
}
