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

"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""
from typing import Any, List, Union, Tuple
import torch
import numpy as np


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


def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)


def batched_eye_like(x: torch.Tensor, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def create_norm_matrix(shift, scale):
    """Create a normalization matrix that shifts and scales points."""
    T = batched_eye_like(shift, 3)
    T[:, 0, 0] = T[:, 1, 1] = scale
    T[:, :2, 2] = shift
    return T


def normalize_keypoints_with_intrinsics(kpts: torch.Tensor, K0: torch.Tensor):
    """
    Normalizes a set of 2D keypoints
    Args:
        kpts: a batch of N 2-dimensional keypoints: (B, N, 2).
        K0: a batch of 3x3 intrinsic matrices: (B, 3, 3)
    """
    kpts[..., 0] -= K0[..., 0, 2].unsqueeze(1)
    kpts[..., 1] -= K0[..., 1, 2].unsqueeze(1)

    kpts[..., 0] /= K0[..., 0, 0].unsqueeze(1)
    kpts[..., 1] /= K0[..., 1, 1].unsqueeze(1)
    return kpts


def normalize_keypoints(kpts: torch.Tensor, size=None, shape=None):
    """Normalize a set of 2D keypoints for input to a neural network.
    Perform the normalization according to the size of the corresponding
    image: shift by half and scales by the longest edge.
    Use either the image size or its tensor shape.
    Args:
        kpts: a batch of N D-dimensional keypoints: (B, N, D).
        size: a tensor of the size the image `[W, H]`.
        shape: a tuple of the image tensor shape `(B, C, H, W)`.
    """
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*w, one*h])[None]

    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]

    T_norm = create_norm_matrix(shift, scale)
    T_norm_inv = create_norm_matrix(-shift/scale[:, None], 1./scale)
    return kpts, T_norm, T_norm_inv


def batched_scale_intrinsics(K: torch.Tensor, scales: torch.Tensor, 
                             invert_scales=True) -> torch.Tensor:
    """
    Args:
        K: a batch of N 3x3 intrinsic matrix, (N, 3, 3)
        scales: a tensor of the shape (N, 1, 1), first horizontal
    """
    scaling_mat = batched_eye_like(K, 3)
    if invert_scales:
        scaling_mat[:, 0, 0] = 1./scales[0]
        scaling_mat[:, 1, 1] = 1./scales[1]
    else:
        scaling_mat[:, 0, 0] = scales[0]
        scaling_mat[:, 1, 1] = scales[1]
    return scaling_mat @ K  # scales @ K


def sample_depth(pts: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """sample depth at points. 

    Args:
        pts (torch.Tensor): (N, 2)
        depth (torch.Tensor): (B, 1, H, W)
    """
    h, w = depth.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    batched = len(depth.shape) == 3
    if not batched:
        pts, depth = pts[None], depth[None]

    pts = (pts / pts.new_tensor([[w-1, h-1]]) * 2 - 1)[:, None]
    depth = torch.where(depth > 0, depth, depth.new_tensor(float('nan')))
    depth = depth[:, None]
    interp_lin = grid_sample(
            depth, pts, align_corners=True, mode='bilinear')[:, 0, 0]
    interp_nn = grid_sample(
            depth, pts, align_corners=True, mode='nearest')[:, 0, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = (~torch.isnan(interp)) & (interp > 0)
    # will exclude out of view matches here, except if the depth was dense and the points falls right at the border,
    # then nearest can get the depth value.
    if not batched:
        interp, valid = interp[0], valid[0]
    return interp, valid


def batch_project_to_other_img_and_check_depth(kpi: torch.Tensor, di: torch.Tensor, 
                                               depthj: torch.Tensor, 
                                               Ki: torch.Tensor, Kj: torch.Tensor, 
                                               T_itoj: torch.Tensor, 
                                               validi: torch.Tensor, 
                                               rth: float=0.1, 
                                               return_repro_error: bool=False
                                               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project pixels of one image to the other, and run depth-check. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        depthj: depth map of image j, BxHxW
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        validi: BxN, Bool mask
        rth: percentage of acceptable depth reprojection error. 
        return_repro_error: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        visible: Bool mask, visible pixels that have a valid reprojection error, BxN
    """

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)
    repro_error = torch.abs(di_j - dj) / dj
    consistent = repro_error < rth
    visible = validi & consistent & validj
    if return_repro_error:
        return kpi_j, visible, repro_error
    return kpi_j, visible


def batch_project_to_other_img(kpi: torch.Tensor, di: torch.Tensor, 
                               Ki: torch.Tensor, Kj: torch.Tensor, 
                               T_itoj: torch.Tensor, 
                               return_depth=False) -> torch.Tensor:
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j


def batch_backproject_to_3d(kpi: torch.Tensor, di: torch.Tensor, 
                            Ki: torch.Tensor, T_itoj: torch.Tensor) -> torch.Tensor:
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

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(
        to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    return kpi_3d_j  # Nx3


def batch_project(kpi_3d_i: torch.Tensor, T_itoj: torch.Tensor, Kj: torch.Tensor, return_depth=False):
    """
    Projects 3D points to image pixels coordinates. 
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        T_itoj: Bx4x4
        Kj: camera intrinsics Bx3x3

    Returns:
        pixels projections in image j, BxNx2
    """
    kpi_3d_in_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_2d_in_j = kpi_3d_in_j @ Kj.transpose(-1, -2)
    if return_depth:
        return from_homogeneous(kpi_2d_in_j), kpi_3d_in_j[..., -1]
    return from_homogeneous(kpi_2d_in_j)


def batch_check_depth_reprojection_error(kpi_3d_i: torch.Tensor, depthj: torch.Tensor, 
                                         Kj: torch.Tensor, T_itoj: torch.Tensor, 
                                         validi: torch.Tensor=None, rth=0.1) -> torch.Tensor:
    """
    Run depth consistency check. Project the 3D point to image j and compare depth of the projection
    to available depth measured at image j. 
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        depthj: depth map of image j, BxHxW
        Kj: Bx3x3
        T_itoj: Bx4x4
        validi: valid 3d poitn mask, BxN, bool
        rth: percentage of depth error acceptable

    Returns:
        visible: torch.Bool BxN, indicates if the 3D point has a valib depth reprojection error. 
    """
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)

    consistent = (torch.abs(di_j - dj) / dj) < rth
    visible = consistent & validj
    if validi is not None:
        visible = visible & validi
    return visible


def batch_project_and_check_depth(kpi_3d_i: torch.Tensor, depthj: torch.Tensor, 
                                  Kj: torch.Tensor, T_itoj: torch.Tensor, 
                                  validi: torch.Tensor, rth=0.1
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points to image pixels coordinates, and run depth reprojection erorr check. 
    Args:
        kpi: BxNx2 coordinates in pixels
        di: BxN, corresponding depths
        depthj: depth map of image j, BxHxW
        Kj: Bx3x3
        T_itoj: Bx4x4
        validi: BxN, bool
        rth: percentage of depth error acceptable

    Returns:
        pixels projections in image j, BxNx2
        visible: torch.Bool BxN, indicates if the point has a valib depth reprojection error. 
    """
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)

    consistent = (torch.abs(di_j - dj) / dj) < rth
    visible = validi & consistent & validj
    return kpi_j, visible


def batch_transform(kpi_3d_i: torch.Tensor, T_itoj: torch.Tensor) -> torch.Tensor:
    """
    Args:
        kpi_3d_i: BxNx3 coordinates in frame i
        T_itoj: Bx4x4
    """
    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    return kpi_3d_j
