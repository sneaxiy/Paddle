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
# Real single-machine 8-card test for MuonShardingOptimizer's overlap of the
# optimizer update with the sharding parameter sync
# (FLAGS_muon_sharding_overlap_optimize_and_sync).
#
# Baseline: _apply_optimize() updates every local param, then
# _sharding_sync_parameters() broadcasts / all-gathers everything.
# Overlap: gradient clipping runs once up-front (it needs the norm over all
# gradients and therefore cannot be overlapped), then the parameter-sync groups
# are walked one by one - each group's params are updated on the calc stream and
# the group's sync is immediately launched on the comm stream, so it overlaps the
# next group's update.
#
# Only *when* the sync collectives are launched changes, never the math, so the
# two runs must agree to the last bit.

import os
import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.muon import (
    MuonParamInfo,
    _default_should_use_muon,
)

SHARDING_DEGREE = 8
# Three distinct weight shapes => three distinct Muon group keys => the 2D
# params are cut into three broadcast stages (a group key is never split across
# stages, so the batched Newton-Schulz calls stay identical).
DIMS = [512, 384, 640]
NUM_BLOCKS = 6
# 1D params big enough to land in more than one fused all-gather buffer.
EXTRA_1D_NUMEL = 300000
NUM_EXTRA_1D = 4
# Small buffers => several sync groups on both the 2D and the 1D side.
COMM_BUFFER_SIZE_MB = 1
BATCH_SIZE = 16  # divisible by SHARDING_DEGREE
STEPS = 5

os.environ["MUON_DEBUG"] = "0"


class MuonMLP(paddle.nn.Layer):
    """Chained Linears cycling through DIMS, plus a few large 1D params.

    The Linear weights are 2D => Muon => whole-tensor sharding + broadcast.
    Biases and the extra 1D params => AdamW => element-wise sharding +
    all-gather. Cycling through three widths yields three Muon group keys.
    """

    def __init__(self, np_weights, np_extras):
        super().__init__()
        self.linears = paddle.nn.LayerList()
        for i, w in enumerate(np_weights):
            in_dim, out_dim = w.shape
            self.linears.append(
                paddle.nn.Linear(
                    in_dim,
                    out_dim,
                    weight_attr=paddle.framework.ParamAttr(
                        initializer=paddle.nn.initializer.Assign(w)
                    ),
                    bias_attr=paddle.framework.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(0.01 * i)
                    ),
                )
            )
        self.extras = paddle.nn.ParameterList(
            [
                self.create_parameter(
                    shape=[EXTRA_1D_NUMEL],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Assign(e),
                )
                for e in np_extras
            ]
        )

    def forward(self, x):
        h = x
        for linear in self.linears:
            h = paddle.tanh(linear(h))
        loss = h.mean() * 100
        for extra in self.extras:
            loss = loss + 0.01 * extra.mean()
        return loss


def _weight_shapes():
    shapes = []
    for _ in range(NUM_BLOCKS):
        for i in range(len(DIMS)):
            shapes.append((DIMS[i], DIMS[(i + 1) % len(DIMS)]))
    return shapes


def _init_weights():
    """Create the initial weights shared by both runs."""
    weights = [
        np.random.random_sample(shape).astype("float32") / 32
        for shape in _weight_shapes()
    ]
    extras = [
        np.random.random_sample((EXTRA_1D_NUMEL,)).astype("float32")
        for _ in range(NUM_EXTRA_1D)
    ]
    return weights, extras


def _build_model(weights, extras):
    model = MuonMLP(weights, extras)
    model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")
    return model


def _build_optimizer(model):
    muon_param_info_map = {}
    for name, param in model.named_parameters():
        use_muon = _default_should_use_muon(name, param.shape, [])
        muon_param_info_map[param.name] = MuonParamInfo(
            use_muon=use_muon, split_concat_func=None
        )
    return paddle.optimizer.Muon(
        parameters=model.parameters(),
        learning_rate=0.001,
        weight_decay=0.00001,
        muon_param_info_map=muon_param_info_map,
        ns_steps=5,
        ns_coeff_type="simple",
        multi_precision=True,
        # Global-norm clipping must happen before any parameter is updated, so
        # it is the part of _apply_optimize that must NOT be overlapped. The
        # threshold is deliberately tiny so that clipping is active on every
        # step and a per-group (instead of global) clip would show up as a diff.
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1e-3),
    )


class TestMuonOverlapSyncVsBaseline(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": SHARDING_DEGREE,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.use_muon_sharding = True
        sharding_configs = self.strategy.hybrid_configs["sharding_configs"]
        sharding_configs.accumulate_steps = 1
        sharding_configs.comm_buffer_size_MB = COMM_BUFFER_SIZE_MB
        sharding_configs.comm_overlap = False

        fleet.init(is_collective=True, strategy=self.strategy)

        # Identical full dataset on every rank; each rank slices its own shard.
        self.data = [
            np.random.random_sample((BATCH_SIZE, DIMS[0])).astype("float32")
            for _ in range(STEPS)
        ]

    def _check_stages(self, inner_opt):
        """The plan must really exercise multi-group pipelining."""
        stages = inner_opt._sync_stages
        assert stages is not None, "the sync-stage plan was not built"
        stages_2d = [s for s in stages if s['kind'] == '2d']
        stages_1d = [s for s in stages if s['kind'] == '1d']
        assert len(stages_2d) >= 2, (
            f"expected several 2D broadcast groups, got {len(stages_2d)}"
        )
        assert len(stages_1d) >= 2, (
            f"expected several 1D all-gather groups, got {len(stages_1d)}"
        )
        # At least one 2D group must be owned by more than one rank, otherwise
        # its broadcast could not overlap another group's update.
        assert any(len(s['rank2params']) > 1 for s in stages_2d), (
            "no 2D group spans multiple owner ranks"
        )
        if paddle.distributed.get_rank() == 0:
            print(
                f"[overlap plan] {len(stages_2d)} 2D broadcast groups, "
                f"{len(stages_1d)} 1D all-gather groups"
            )

    def _run(self, weights, extras, overlap):
        """Train on 8 cards for STEPS and return the final params as numpy."""
        os.environ["FLAGS_muon_sharding_overlap_optimize_and_sync"] = (
            "1" if overlap else "0"
        )

        model = _build_model(weights, extras)
        optimizer = _build_optimizer(model)
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        inner_opt = optimizer._inner_opt
        assert isinstance(inner_opt, MuonShardingOptimizer)
        # Guard against the comparison passing trivially because the flag never
        # took effect.
        assert inner_opt.overlap_optimize_and_sync is overlap, (
            f"overlap_optimize_and_sync is "
            f"{inner_opt.overlap_optimize_and_sync}, expected {overlap}"
        )

        hcg = fleet.get_hybrid_communicate_group()
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_bs = BATCH_SIZE // SHARDING_DEGREE

        losses = []
        for idx in range(STEPS):
            start = sharding_rank * local_bs
            batch = paddle.to_tensor(self.data[idx][start : start + local_bs])
            with paddle.amp.auto_cast(dtype="bfloat16"):
                loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(loss.cast("float32").numpy())
            if overlap and idx == 0:
                self._check_stages(inner_opt)

        params = {
            name: p.cast("float32").numpy()
            for name, p in model.named_parameters()
        }
        return losses, params

    def test_overlap_sync_matches_baseline_bitwise(self):
        weights, extras = _init_weights()

        baseline_losses, baseline = self._run(weights, extras, overlap=False)
        overlap_losses, overlap = self._run(weights, extras, overlap=True)

        for idx, (base_loss, ovlp_loss) in enumerate(
            zip(baseline_losses, overlap_losses)
        ):
            np.testing.assert_array_equal(
                ovlp_loss,
                base_loss,
                err_msg=f"loss of step {idx} differs from the baseline",
            )

        assert set(baseline) == set(overlap)
        for name in baseline:
            np.testing.assert_array_equal(
                overlap[name],
                baseline[name],
                err_msg=(
                    f"Param {name!r} differs between the overlapped and the "
                    f"baseline step; overlapping the update with the parameter "
                    f"sync is not bit-for-bit identical."
                ),
            )
        if paddle.distributed.get_rank() == 0:
            print(
                "[PASS] overlap_optimize_and_sync == baseline bit-for-bit "
                f"across {len(baseline)} params and {STEPS} steps"
            )


if __name__ == "__main__":
    unittest.main()
