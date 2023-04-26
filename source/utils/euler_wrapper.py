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

import subprocess
import os

os.listdir()
def prepare_data(root_path, untar_path, mode='euler'):
    """Untar data at the specified root_path. Only useful for euler cluster"""
    if mode == 'euler':
        tar_file_path = root_path

        tar_file_name = os.path.split(tar_file_path)[-1].split('.')[0]
        out_path = '{}/{}'.format(untar_path, tar_file_name)

        marker_path = '{}_done.txt'.format(out_path)
        if not os.path.exists(marker_path):
            # not os.path.isdir(out_path) or
            if tar_file_path.endswith('.tar.gz'):
                cmd = 'tar -I pigz -xf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('tar'):
                cmd = 'tar -xf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('tar.xz'):
                cmd = 'tar -xvf {} -C {}'.format(tar_file_path, untar_path)
            elif tar_file_path.endswith('.zip'):
                # os.system('mkdir -p {}'.format(untar_path))
                # os.system('gsutil -m cp {} {}'.format(tar_file_path, untar_path))
                # os.system('unzip -q {} -d {}'.format(out_path, untar_path))
                # cmd = ' '
                cmd = 'unzip {} -d {}'.format(tar_file_path, untar_path)
            else:
                raise ValueError('Untaring file selected not valid : {}'.format(root_path))
            print('Copying data: {}'.format(cmd))
            out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            stdout, stderr = out.communicate()
            print(stdout)
            print(stderr)
            with open(marker_path, mode='a'):
                pass
        else:
            print('already downloaded {}'.format(out_path))
