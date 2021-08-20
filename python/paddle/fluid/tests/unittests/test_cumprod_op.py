# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest
import random
import paddle


class TestCumprod(OpTest):
    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        x = np.random.random(self.shape).astype('float64') + 0.5
        if zero_num > 0:
            zero_num = min(zero_num, x.size)
            shape = x.shape
            x = x.flatten()
            indices = random.sample(range(x.size), zero_num)
            for i in indices:
                x[i] = 0
            x = np.reshape(x, self.shape)
        self.inputs = {'X': x}
        self.outputs = {'Out': np.cumprod(x, axis=dim)}
        self.attrs = {'dim': dim}

    def init_params(self):
        self.shape = [4, 5, 6]
        self.zero_nums = [0, 10, 20, 30, int(np.prod(self.shape))]

    def setUp(self):
        self.init_params()
        self.op_type = "cumprod"
        self.inputs = {'X': None}
        self.outputs = {'Out': None}
        self.attrs = {'dim': None}

    def _get_places(self):
        return [paddle.CUDAPlace(0)]

    def test_check_output(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_output()

    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
