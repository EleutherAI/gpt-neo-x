# Copyright (c) 2024, EleutherAI
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

"""
Instantiate models, run for a small number of iterations, and check that training loss improves.

Performs testing using a linear grid search over important parameter values, so that each setting that differs from the base is tested in isolation.

Potentially use fuzzing to test parameters in combination.
"""
import pytest
import train
from copy import deepcopy
from unittest.mock import patch
from megatron.neox_arguments import NeoXArgs
from tests.common import simulate_deepy_env, BASE_CONFIG

PARAMS_TO_TEST = {
    "gpt_j_residual": [True, False],
    "pos_emb": ["learned", "rotary", "sinusoidal", "rpe", "alibi", "none"],
    "attention_config": [
        "global",
        "local",
        "sparse_fixed",
        "sparse_variable",
        "bigbird",
        "bslongformer",
        "gmlp",
        "flash",
    ],
    "hidden_dropout": [0, 0.1],
    "weight_decay": [0, 0.1],
    "use_bias_in_attn_linear": [True, False],
    "use_bias_in_norms": [True, False],
    "precision": ["fp16", "fp32", "bfloat16"],
}

keys_to_test = PARAMS_TO_TEST.keys()

@pytest.mark.parametrize(
    "key, value",
    [(key, value) for key in keys_to_test for value in PARAMS_TO_TEST[key]],
)
def test_model_training_options(monkeypatch, key, value):
    # TODO: Possibly add testing over world_size=2 back in
    neox_args = NeoXArgs.from_dict(BASE_CONFIG)
    if getattr(neox_args, key) == value:
        pytest.skip("Skipping to avoid redundancy as no change in base config")
    if key == "precision" and value == "bfloat16":
        pytest.xfail(
            reason="Assumes that ZeRO optimization stage has been set in the YAML"
        )
    param_dict = {key: value}
    run_train_test(monkeypatch, overwrite_values=param_dict)


def run_train_test(monkeypatch, overwrite_values: dict):
    max_train_iters = 32
    checkpoint_args = {"train_iters": max_train_iters}
    overwrite_values = checkpoint_args
    input_args = ["train.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)

    # Train model, whilst patching collect_loss_for_unit_test to track model loss at each step
    loss_per_iteration = []
    with patch(
        "megatron.training.collect_loss_for_unit_test",
        side_effect=lambda x: loss_per_iteration.append(x),
    ):
        train.main(input_args=deepspeed_main_args, overwrite_values=overwrite_values)
        assert (
            len(loss_per_iteration) == max_train_iters
        ), "patching should have collected loss values from each train step"

        # loss should have decreased by now (otherwise increasing the max_steps parameter could have the testcase pass)
        assert min(loss_per_iteration) < loss_per_iteration[0], (
            "training loss should improve within " + str(max_train_iters) + " steps"
        )
