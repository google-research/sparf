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
from source.datasets import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler
from typing import Optional
from operator import itemgetter
from typing import Any, Dict

from source.utils.euler_wrapper import prepare_data
    
    
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def create_dataset(args: Dict[str, Any], mode: str='train') -> Dataset:
    """
    Args:
        args (dict): settings. Must contain keys 'dataset' and 'scene'
        mode (str, optional): Defaults to 'train'.

    Returns:
        Dataset and optional sampler. 
    """
    if mode == 'train':
        print('training dataset: {}'.format(args.dataset))
        
        if hasattr(args, 'copy_data') and args.copy_data:
            print('Copying training data and unzipping...')
            prepare_data(args.env.__dict__[args.dataset + '_tar'], args.env.untar_dir)
            
        # a single dataset
        train_dataset = dataset_dict[args.dataset](args, mode, scenes=args.scene)
        print('Train dataset {} has {} samples'.format(args.dataset, len(train_dataset)))
        train_sampler = DistributedSampler(train_dataset) if args.distributed else None

        return train_dataset, train_sampler

    elif mode in ['val', 'test']:

        print('eval dataset: {}'.format(args.dataset))
        print('eval scenes: {}'.format(args.scene))
        
        if hasattr(args, 'copy_data') and args.copy_data:
            print('Copying validation data and unzipping...')
            prepare_data(args.env.__dict__[args.dataset + '_tar'], args.env.untar_dir)

        val_dataset = dataset_dict[args.dataset](
            args, mode,
            scenes=args.scene)
        print('Val dataset {} has {} samples'.format(args.dataset, len(val_dataset)))

        return val_dataset

    else:
        raise ValueError



