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

from source.admin.environment import env_settings
from typing import Any, Dict


class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self, data_root: str = '', debug: bool=False):
        self.set_default(data_root, debug=debug)

    def update(self, dict_: Dict[str, Any]):
        for name, val in dict_.items():
            setattr(self, name, val)

    def set_default(self, data_root: str = '', debug: bool = False):
        self.env = env_settings(data_root, debug)
        self.use_gpu = True
        

def boolean_string(s: bool) -> bool:
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
