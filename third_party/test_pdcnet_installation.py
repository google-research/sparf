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

""" Make sure PDC-net is installed and running properly, by running it on a pair of images. """
import torch
from easydict import EasyDict as edict
import os
import sys
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
torch.set_grad_enabled(False)

from pathlib import Path

env_path = str(Path(__file__).parent /  '../')
if env_path not in sys.path:
    sys.path.append(env_path)

from train_settings.default_config import get_joint_pose_nerf_default_config_360_data

env_path = str(Path(__file__).parent /  '../third_party/DenseMatching')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
from model_selection import select_model
from utils_flow.img_processing_utils import pad_to_same_shape
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask


def test_model_on_image_pair(args, query_image, reference_image):
    network, estimate_uncertainty = select_model(
        args.model, args.pre_trained_model, args, args.optim_iter, args.local_optim_iter,
        path_to_pre_trained_models=args.path_to_pre_trained_models)

    # save original ref image shape
    ref_image_shape = reference_image.shape[:2]

    # pad both images to the same size, to be processed by network
    query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
    # convert numpy to torch tensor and put it in right format
    query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
    reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

    # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
    # specific pre-processing (/255 and rescaling) are done within the function.

    # pass both images to the network, it will pre-process the images and ouput the estimated flow
    # in dimension 1x2xHxW
    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                        reference_image_,
                                                                                        mode='channel_first')
    confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
    confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]

    estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
    estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
    # removes the padding

    warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                    estimated_flow_numpy[:, :, 1]).astype(np.uint8)

    color = [255, 102, 51]
    fig, axis = plt.subplots(1, 5, figsize=(30, 30))

    confident_mask = (confidence_map > 0.50).astype(np.uint8)
    confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
    axis[2].imshow(confident_warped)
    axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
                        .format(args.model, args.pre_trained_model))
    axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
    axis[4].set_title('Confident regions')
    axis[0].imshow(query_image)
    axis[0].set_title('Query image')
    axis[1].imshow(reference_image)
    axis[1].set_title('Reference image')

    axis[3].imshow(flow_to_image(estimated_flow_numpy))
    axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
    fig.savefig(
        os.path.join(args.save_dir, 'test_pdcnet.png'),
        bbox_inches='tight')
    plt.close(fig)
    print('Saved image!')
    return 

if __name__ == "__main__":
    
    # get the path to pdcnet checkpoint
    default_config = get_joint_pose_nerf_default_config_360_data()
    ckpt_path = default_config.flow_ckpt_path
    default_pdcnet_cfg = {
        'model': 'PDCNet', 
        'homo': False, 
        'optim_iter': 3, 
        'local_optim_iter': 3, 
        'path_to_pre_trained_models': os.path.dirname(ckpt_path), 
        'pre_trained_model': 'megadepth', 
        'network_type': None, 
        'save_dir': 'third_party/'
    }

    args = edict(default_pdcnet_cfg)

    print('Applying PDC-Net to two images of DTU...')
    path_query_image = 'docs/000035.png'
    path_reference_image = 'docs/000045.png'
    query_image = cv2.imread(path_query_image, 1)[:, :, ::- 1]
    reference_image = cv2.imread(path_reference_image, 1)[:, :, ::- 1]

    test_model_on_image_pair(args, query_image, reference_image)
