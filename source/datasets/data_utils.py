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
import math
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import imageio
from typing import Union, Any, List, Callable, Dict

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def is_img_file(file_path):
    ext = file_path.split('.')[-1].lower()
    return ext == 'png' or ext == 'jpg' or ext == 'jpeg'

def imread_numpy(path):
    return imageio.imread(path).astype(np.float32) / 255.0

# COLOR AUGMENTATIONS AND CROPPING
# different parametrization of the camera 
def crop(image: Union[torch.Tensor, np.ndarray], size: List[int], random: bool=True, 
         intr: Union[torch.Tensor, np.ndarray]=None, 
         camera: Union[torch.Tensor, np.ndarray]=None, 
         other: Union[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]=None, 
         return_bbox: bool=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics.
    """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    if w_new >= w or h_new >= h:
        ret = [image]
        if intr is not None:
            ret += intr
        if camera is not None:
            ret += camera
        if other is not None:
            ret += other
        return ret
    top = np.random.randint(0, h - h_new + 1) if random else (h - h_new) // 2
    left = np.random.randint(0, w - w_new + 1) if random else (w - w_new) // 2
    image = image[top:top+h_new, left:left+w_new]
    ret = [image]

    if intr is not None:
        intr = intr.clone() if isinstance(intr, torch.Tensor) else intr.copy()
        intr[0, 2] -= left
        intr[1, 2] -= top
        ret += intr
    if camera is not None:
        camera = camera.clone() if isinstance(camera, torch.Tensor) else camera.copy()
        camera[0] = h_new
        camera[1] = w_new
        camera[4] -= left
        camera[8] -= top
    if other is not None:
        if isinstance(other, list):
            for other_ in other:
                if other_ is not None:
                    ret += [other_[top:top+h_new, left:left+w_new]]
        else:
            ret += [other[top:top+h_new, left:left+w_new]]
    if return_bbox:
        ret += [(top, top+h_new, left, left+w_new)]
    return ret

def resize(image: np.ndarray, size: List[int], fn: Callable[[List], float]=None, 
           interp: str='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        # pil
        w, h = image.size[:2]
        
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
    else:
        raise ValueError(f'Incorrect new size: {size}')
    
    # wants size to be dividable by 2!
    # h_new = h_new + 1 if h_new % 2 == 1 else h_new
    # w_new = w_new + 1 if w_new % 2 == 1 else w_new
    scale = (w_new / w, h_new / h)
    
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)
    else:
        image = image.resize((w_new, h_new))
    return image, scale

def resize_image_w_intrinsics(image: np.ndarray, size: List[int], resize_factor: float, 
                              intr: Union[torch.Tensor, np.ndarray]=None, 
                              camera: Union[torch.Tensor, np.ndarray]=None, 
                              fn: Callable[[List], float]=None, interp: str='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    assert intr is not None or camera is not None
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        # pil
        w, h = image.size
        
    if resize_factor is not None:
        h_new, w_new = int(round(h*resize_factor)), int(round(w*resize_factor))
    else:
        if isinstance(size, int):
            scale = size / fn(h, w)
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
        elif isinstance(size, (tuple, list)):
            h_new, w_new = size
        else:
            raise ValueError(f'Incorrect new size: {size}')
    
    # wants size to be dividable by 2!
    h_new = h_new + 1 if h_new % 2 == 1 else h_new
    w_new = w_new + 1 if w_new % 2 == 1 else w_new
    scale = (float(w_new) / float(w), float(h_new) / float(h))
    
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)
    else:
        image = image.resize((w_new, h_new))
    
    if camera is not None:
        camera = camera.clone() if isinstance(camera, torch.Tensor) else camera.copy()
        camera[0] = h_new
        camera[1] = w_new
        camera[2:6] *= scale[0] # first line of intrinsics in shape 4x4
        camera[6:10] *= scale[1] # second line of intrinsics in shape 4x4
        return image, camera
    else:
        intr = intr.clone() if isinstance(intr, torch.Tensor) else intr.copy()
        intr[0] *= scale[0]
        intr[1] *= scale[1]
        return image, intr

def numpy_image_to_torch(image: np.ndarray):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy((image / 255.).astype(np.float32, copy=False))


# ------ FOR VIEW SELECTION --------------
def vector_norm(data: np.ndarray, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle: np.ndarray, axis):
    """Return quaternion for rotation about axis.
    """
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def quaternion_matrix(quaternion: np.ndarray):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def rectify_inplane_rotation(src_pose: np.ndarray, tar_pose: np.ndarray, src_img: np.ndarray, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255*src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated


def angular_dist_between_2_vectors(vec1: np.ndarray, vec2: np.ndarray):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists

def batched_angular_dist_rot_matrix(R1: np.ndarray, R2: np.ndarray):
    '''
    calculate the angular distance between two rotation matrices (batched)
    Args:
        R1: the first rotation matrix [N, 3, 3]
        R2: the second rotation matrix [N, 3, 3]
    Retugns: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose_c2w: np.ndarray, ref_poses_c2w: np.ndarray, num_select: int, 
                         tar_id=-1, angular_dist_method: str='vector',
                         scene_center=(0, 0, 0)) -> List[int]:
    '''
    Args:
        tar_pose_c2w: target pose [3, 3]
        ref_poses_c2w: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
        angular_dist_method: matrix, vector, dist, random
    Returns: the selected indices
    '''
    num_cams = len(ref_poses_c2w)
    if tar_id > 0:
        # the image to render is one of the training images
        num_select = min(num_select, num_cams-1)
    else:
        num_select = min(num_select, num_cams)
    batched_tar_pose = tar_pose_c2w[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses_c2w[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses_c2w[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses_c2w[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    elif angular_dist_method == 'random':
        dists = np.random.rand(num_cams)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself
    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]

    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids

