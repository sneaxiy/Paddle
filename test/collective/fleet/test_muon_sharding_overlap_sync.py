# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#
# Launcher: runs hybrid_parallel_sharding_muon_overlap_sync_model.py on 8 GPUs
# to verify that overlapping MuonShardingOptimizer's parameter update with the
# sharding parameter sync is bit-for-bit identical to the non-overlapped path.

import time
import unittest

from legacy_test.test_parallel_dygraph_dataparallel import (
    get_cluster_from_args,
    get_devices,
    start_local_trainers,
)

from paddle import base
from paddle.distributed.utils.launch_utils import watch_local_trainers

NUM_DEVICES = 8


class TestMuonShardingOverlapSync(unittest.TestCase):
    def test_overlap_sync_matches_baseline(self):
        if (
            not base.core.is_compiled_with_cuda()
            or base.core.get_cuda_device_count() < NUM_DEVICES
        ):
            self.skipTest(f"this test needs {NUM_DEVICES} GPUs")

        selected_devices = get_devices(
            ",".join(str(i) for i in range(NUM_DEVICES))
        )
        cluster, pod = get_cluster_from_args(selected_devices)

        procs = start_local_trainers(
            cluster,
            pod,
            training_script="hybrid_parallel_sharding_muon_overlap_sync_model.py",
            training_script_args=[],
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())
            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)


if __name__ == "__main__":
    unittest.main()
