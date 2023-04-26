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
import imageio
from pathlib import Path
from typing import Union, Any, Dict, Tuple


def pad_poses(p: np.ndarray):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses: np.ndarray):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def shift_origins(origins: np.ndarray, directions: np.ndarray, near=0.0):
  """Shift ray origins to near plane, such that oz = near."""
  t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions
  return origins


def poses_avg(poses: np.ndarray):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray, 
               subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def focus_pt_fn(poses: np.ndarray):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def normalize(x: np.ndarray):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def generate_spiral_path(poses_c2w: Union[torch.Tensor, np.ndarray], bounds, 
                         n_frames=240, n_rots=2, zrate=.5):
    """Calculates a forward facing spiral path for rendering.
    poses are in opencv format. """
    is_torch = False
    if isinstance(poses_c2w, torch.Tensor):
        is_torch = True
        poses_c2w = poses_c2w.detach().cpu().numpy()

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses_c2w[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses_c2w)
    up = poses_c2w[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    if is_torch:
        render_poses = torch.from_numpy(render_poses)[:, :3]
    return render_poses


def generate_spiral_path_dtu(poses_c2w: Union[torch.Tensor, np.ndarray], 
                             n_frames=240, n_rots=2, zrate=.5, perc=60):
    """Calculates a forward facing spiral path for rendering for DTU.
    poses are in opencv format. """
    is_torch = False
    if isinstance(poses_c2w, torch.Tensor):
        is_torch = True
        poses_c2w = poses_c2w.detach().cpu().numpy()
    # Get radii for spiral path using 60th percentile of camera positions.
    positions = poses_c2w[:, :3, 3]
    radii = np.percentile(np.abs(positions), perc, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = poses_avg(poses_c2w)
    up = poses_c2w[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses_c2w)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
        position = cam2world @ t
        render_poses.append(viewmatrix(z_axis, up, position, True))
    render_poses = np.stack(render_poses, axis=0)
    if is_torch:
        render_poses = torch.from_numpy(render_poses)[:, :3]
    return render_poses