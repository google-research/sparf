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
import tqdm
import threading,queue
from easydict import EasyDict as edict
from typing import List, Mapping, Any, Callable, Dict, Tuple
from typing import Any

from source.utils.config_utils import override_options
from source.datasets.data_utils import crop, resize_image_w_intrinsics, numpy_image_to_torch, resize
from source.utils.geometry.geometric_utils_numpy import backproject_to_3d, project, get_absolute_coordinates

default_conf = {'copy_data': False,  
                # specific to cluster, if needs to copy from zip
                
                'resize': None,
                'resize_factor': None, 
                'resize_by': 'max',
                'crop_ratio': None, 
                'crop': None,
                'apply_augmentation': False,
                'train_sub': None, 
                'val_sub': None,
                'mask_img': False, 
                
                'increase_depth_range_by_x_percent': 0.,
                
                # llff
                'llffhold': 8, 

                # dtu stuff
                'dtu_split_type': 'pixelnerf', 
                'dtuhold': 8, 
                'dtu_light_cond': 3,   # Light condition. Used only by DTU.
                'dtu_max_images': 49,   # Whether to restrict the max number of images.'
                }

class Dataset(torch.utils.data.Dataset):
    """Base for all datasets. """
    def __init__(self, args: Dict[str, Any], split: str):
        super().__init__()

        self.args = edict(override_options(default_conf, args))

        self.split = split

    def get_all_camera_poses(self, args: Dict[str, Any]):
        raise NotImplementedError

    def prefetch_all_data(self):
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
        return 
    
    def prefetch_at_indices(self, ind_list: List[int]):
        return torch.utils.data._utils.collate.default_collate(ind_list)

    def setup_loader(self, shuffle: bool = False, drop_last: bool = False):
        loader = torch.utils.data.DataLoader(self,
            batch_size=1,
            num_workers=self.args.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False, # spews warnings in PyTorch 1.9 but should be True in general
        )
        print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self):
        raise NotImplementedError

    def preload_worker(self, data_list: List[Mapping], load_func: Callable, q,lock,idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()
        return 

    def preload_threading(self, load_func: Callable,data_str: str="images"):
        data_list = [None]*len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(self.args.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

    def __getitem__(self,idx: int):
        raise NotImplementedError

    def get_image(self, idx: int):
        raise NotImplementedError

    def get_correspondences(self, idx_target: int, idx_source: int, 
                            ret_target: Dict[str, Any]=None, ret_source: Dict[str, Any]=None
                            ) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """If ground-truth depth and poses are available, can generate the ground-truth correspondences
        relating a source and target images. """
        if ret_target is None:
            ret_target = self.__getitem__(idx_target)
        
        if ret_source is None:
            ret_source = self.__getitem__(idx_source)
        
        H, W = ret_target['image'].shape[-2:]
        pixels_target = get_absolute_coordinates(H, W).reshape(-1, 2)  # (H, W, 2)
        depth = ret_target['depth_gt'].reshape(-1)
        K_target = ret_target['intr']
        K_source = ret_source['intr']
        w2c_target_ = ret_target['pose']
        w2c_target = np.eye(4)
        w2c_target[:3] = w2c_target_

        w2c_source_ = ret_source['pose']
        w2c_source = np.eye(4)
        w2c_source[:3] = w2c_source_

        target2source = w2c_source @ np.linalg.inv(w2c_target)
        kp3d = backproject_to_3d(kpi=pixels_target, di=depth, Ki=K_target, T_itoj=target2source)
        repr_in_source = project(kp3d, T_itoj=np.eye(4), Kj=K_source)
        corres_target_to_source = repr_in_source.reshape(H, W, 2)
        return ret_target, ret_source, corres_target_to_source

    def preprocess_image_and_intrinsics(self, image: np.ndarray, intr: np.ndarray, depth: np.ndarray=None, 
                                        mask_depth: np.ndarray=None, mask: np.ndarray=None, 
                                        new_size: List[int]=None, channel_first: bool=True) -> List[Any]:
        """Resizes images to the desired size. Adjust intrinsics accordingly. If provided, resize the depth map
        and masks as well. 
        Return the image as a torch.Tensor, normalized to [0, 1]. The others are numpy arrays. 

        Args:
            image (np.array): (H, W, 3)
            intr (np.array): (3, 3)
            depth (np.array, optional): (H, W). Defaults to None.
            mask_depth (np.array, optional): (H, W). Defaults to None.
            mask (np.array, optional): (H, W). Defaults to None.
            new_size (list, optional): (H_new, W_new). Defaults to None.
            channel_first (bool, optional): _description_. Defaults to True.
        """
        image = np.array(image).astype(np.float32)

        # crop image and modify intrinsics accordingly
        if self.args.crop_ratio or self.args.crop:
            if self.args.crop_ratio is not None:
                crop_H = int(self.raw_H*self.args.crop_ratio)
                crop_W = int(self.raw_W*self.args.crop_ratio)
            elif self.args.crop is not None:
                crop_H, crop_W = self.args.crop
            # will make sure the size is dividable by 2
            crop_H = crop_H + 1 if crop_H % 2 == 1 else crop_H
            crop_W = crop_W + 1 if crop_W % 2 == 1 else crop_W

            image, intr, depth = crop(image, size=(crop_H, crop_W), intr=intr, 
                                    other=[depth, mask_depth, mask], random=self.split == 'train')
 
        # resize
        fn = max if self.args.resize_by == 'max' else min
        # will make sure the size is dividable by 2, embedded in the functions
        if new_size is not None:
            image, intr = resize_image_w_intrinsics(image, new_size, None, intr=intr, fn=fn)
        elif self.args.resize or self.args.resize_factor:
            new_size = self.args.resize
            resize_factor = self.args.resize_factor
            image, intr = resize_image_w_intrinsics(image, new_size, resize_factor, intr=intr, fn=fn)
        
        # convert image to torch.Tensor and normalize to [0, 1]
        image = numpy_image_to_torch(image) # shape (3, H, W) and [0, 1] 
        img_shape = image.shape[-2:]  # (H, W)

        if not channel_first:
            image = image.permute(1, 2, 0)  # shape is (H, W, 3) and [0, 1]

        ret = [image, intr]

        if depth is not None:
            depth, _ = resize(depth, img_shape, interp='nearest')
            ret.append(depth)
        if mask_depth is not None:
            mask_depth, _ = resize(mask_depth.astype(np.float32), img_shape)
            mask_depth = np.floor(mask_depth).astype(np.bool)
            ret.append(mask_depth)
        if mask is not None:
            mask, _ = resize(mask.astype(np.float32), img_shape)
            mask = np.floor(mask).astype(np.bool)
            ret.append(mask)
        return ret

    def __len__(self):
        return len(self.list)
