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

import os

class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        if debug:
            experiment_dir = 'experiments_sparf_debug'
        else:
            experiment_dir = 'experiments_sparf'
        self.workspace_dir = os.path.join(data_root, f'{experiment_dir}/snapshots')    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(data_root, f'{experiment_dir}/tensorboard')    # Directory for tensorboard files.
        self.log_dir = os.path.join(data_root, f'{experiment_dir}/logs')
        
        self.pretrained_networks = self.workspace_dir
        self.eval_dir = os.path.join(data_root, 'test_results_sparf')

        # actual dataset
        self.llff = os.path.join('data', 'nerf_llff_data')
        self.replica = os.path.join('data', 'Replica')
        self.dtu = os.path.join('data', 'rs_dtu_4/DTU')
        self.dtu_depth = os.path.join('data')
        self.dtu_mask = os.path.join('data', 'submission_data/idrmasks')
