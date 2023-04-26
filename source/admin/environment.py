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

import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    """ Contains the path to all_no_processing_of_pts necessary datasets or useful folders (like workspace, pretrained models..)"""
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': empty_str,
        'pretrained_networks': 'self.workspace_dir',
        'eval_dir': empty_str, 
        
        'llff': empty_str, 
        'dtu': empty_str, 
        'dtu_depth': empty_str, 
        'dtu_mask': empty_str, 
        'replica': empty_str, 
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.', 
               'pretrained_networks': 'Directory for saving other models pre-trained networks', 
               'eval_dir': 'Base directory for saving the evaluations. '}

    with open(path, 'w') as f:
        f.write('import os\n\n')
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self, data_root=\'\', debug=False):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings(data_root='', debug=False):
    env_module_name = 'source.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings(data_root, debug)
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        # create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all_no_processing_of_pts the paths you need. '
                           'Then try to run again.'.format(env_file))
