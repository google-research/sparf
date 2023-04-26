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

import torch


def lossfun_distortion(t: torch.Tensor, w: torch.Tensor, 
                       normalize: bool=False) -> torch.Tensor:
  """Compute distortion loss of mipnerf360. 

  pushes the weights along  a ray to sum to 0 and to be localized in space (have a single peak at the surface)
  https://github.com/kakaobrain/NeRF-Factory/blob/ce06663abe385c5cbe85fddbca8b9b5ace80dbee/src/model/mipnerf360/model.py

  Args:
    w: the weights of each sample along the ray. [B, n_rays, n_samples, 1]
    t: the depth of each sample along the ray. [B, n_rays, n_samples, 1]
  """

  if normalize:
    w += 1e-6 
    w = w / torch.sum(w, axis=-2, keepdims=True)

  w = w[..., 0]  # [B, n_rays, n_samples]
  t = t[..., 0]  # [B, n_rays, n_samples]

  # The loss incurred between all pairs of intervals.
  ut = (t[..., 1:] + t[..., :-1]) / 2  # [B, n_rays, n_samples-1]
  w = w[..., 1:] # [B, n_rays, n_samples-1]
  dut = torch.abs(ut[..., :, None] - ut[..., None, :])
  loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = torch.sum(w**2 * torch.diff(t), axis=-1) / 3

  return (loss_inter + loss_intra).mean()


def depth_patch_loss(depths: torch.Tensor, patch_size: int, 
                     charbonnier_padding: float = 0.001) -> torch.Tensor:
  """Computes a smoothing regularizer over output depth patches.

  Args:
    depths: the computed depths to perform the regularization on.
    charbonnier_padding: the padding parameter of the Charbonnier loss.

  Returns:
    The loss value.
  """
  # predicted depth is B, -1, 1, need to convert to B, -1, patch**2
  B = depths.shape[0]
  depths = depths.reshape(B, -1, patch_size**2)
  resid_sq = (depths[..., None] - depths[..., None, :])**2
  return torch.sqrt(resid_sq + charbonnier_padding**2).mean()