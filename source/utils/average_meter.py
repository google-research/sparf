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


class AverageMeter:
    """Keeps track of the metrics to log. """
    def __init__(self, last_n=None):
        self._records = []
        self.last_n = last_n

    def update(self, result):
        if isinstance(result, (list, tuple)):
            self._records += result
        else:
            self._records.append(result)

    def reset(self):
        self._records.clear()

    # @property
    def records(self, take_subset=True):
        if self.last_n is not None and take_subset:
            # only compute within the last recorded elements
            return self._records[-self.last_n:]
        else:
            return self._records

    def sum(self):
        return np.sum(self.records())

    def mean(self, take_subset=True):
        return np.mean(self.records(take_subset))

    def std(self):
        return np.std(self.records())

    def median(self):
        return np.median(self.records())

    def max(self, take_subset=False):
        return np.max(self.records(take_subset))
    
    def last(self):
        return self.records()[-1]
