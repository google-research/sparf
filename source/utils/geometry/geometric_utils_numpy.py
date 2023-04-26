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
import torch
from typing import Any, List, Union, Tuple

def get_absolute_coordinates(h_scale: int, w_scale: int, 
                             is_torch_tensor: bool =False) -> Union[np.ndarray, torch.Tensor]:
    """Get pixels coordinates

    Args:
        h_scale (int)
        w_scale (int)
        is_torch_tensor (bool, optional): Defaults to False.

    Returns:
        grid (torch.Tensor): Pixels coordinates, (H, W, 2)
    """
    if is_torch_tensor:
        xx = torch.arange(start=0, end=w_scale).view(1, -1).repeat(h_scale, 1)
        yy = torch.arange(start=0, end=h_scale).view(-1, 1).repeat(1, w_scale)
        xx = xx.view(h_scale, w_scale, 1)
        yy = yy.view(h_scale, w_scale, 1)
        grid = torch.cat((xx, yy), -1).float()  # H, W, 2
    else:
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        grid = np.dstack((X, Y))
    return grid


def angles2rotation_matrix(angles: Union[List[float], np.ndarray]):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

    
def dummy_projection_matrix() -> np.ndarray:
    eye = np.eye(3)
    matrix = np.zeros((4, 4), np.float32)
    matrix[-1, -1] = 1.0
    matrix[:3, :3] = eye
    return matrix


def scale_intrinsics(K: np.ndarray, scales: np.ndarray, 
                     invert_scale=True) -> np.ndarray:
    """
    Args:
        K: a batch of N 3x3 intrinsic matrix, (N, 3, 3)
        scales: a tensor of the shape (N, 1, 1), first horizontal
        invert_scale: whether to invert the scale?
    """
    if invert_scale:
        scales = np.diag([1./scales[0], 1./scales[1], 1.])
    else:
        scales = np.diag([scales[0], scales[1], 1.])
    return np.dot(scales, K)  # scales @ K


def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points: np.ndarray) -> np.ndarray:
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]


def backproject_to_3d(kpi: np.ndarray, di: np.ndarray, 
                      Ki: np.ndarray, T_itoj: np.ndarray) -> np.ndarray:
    """
    Backprojects pixels to 3D space 
    Args:
        kpi: BxNx2 coordinates in pixels
        di: BxN, corresponding depths
        Ki: camera intrinsics, Bx3x3
        T_itoj: Bx4x4
    Returns:
        kpi_3d_j: 3D points in coordinate system j, BxNx3
    """

    kpi_3d_i = to_homogeneous(kpi) @ np.linalg.inv(Ki).T
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
        to_homogeneous(kpi_3d_i) @ T_itoj.T)
    return kpi_3d_j  # Nx3


def project(kpi_3d: np.ndarray, T_itoj: np.ndarray, Kj: np.ndarray) -> np.ndarray:
    """
    Projects 3D points to image pixels coordinates. 
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        T_itoj: Bx4x4
        Kj: camera intrinsics Bx3x3

    Returns:
        pixels projections in image j, BxNx2
    """
    kpi_3d_in_j = from_homogeneous(to_homogeneous(kpi_3d) @ T_itoj.T)
    kpi_2d_in_j = kpi_3d_in_j @ Kj.T
    return from_homogeneous(kpi_2d_in_j)


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2.
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

