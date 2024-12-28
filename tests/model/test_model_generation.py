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
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""


import os
import pytest
from tests.common import DistributedTest, model_setup, parametrize

PARAMS_TO_TEST = {
    "include":["localhost:0,1"],
    "pipe_parallel_size,model_parallel_size,world_size": [
        [0, 1, 1],
        [0, 1, 2],
        [1, 2, 2],
        [0, 2, 2],
        [2, 1, 2],
    ],
    "top_p,temperature,top_k": [[0.0, 0.5, 0], [0.5, 0.0, 100], [0.5, 0.5, 0]],
    "prompt": ["", "hello world"],
    "fp16,fp32_allreduce": [
        [
            {
                "enabled": True,
                "type": "bfloat16",
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            True,
        ],
        [
            {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            False,
        ],
    ],
}

parameters, names = parametrize(
    PARAMS_TO_TEST, max_tests=int(os.getenv("MAX_TESTCASES", 50)), seed=None
)

class TestModelGeneration(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("param_dict", parameters, ids=names)
    def test_generate(self, param_dict, tmpdir):
        from megatron.text_generation_utils import generate_samples_from_prompt
        from megatron.utils import is_mp_rank_0

        fixed_params = {
            "num_samples": 3,
            "maximum_tokens": 50,
            "make_vocab_size_divisible_by": 2,
            "sample_output_file": "test_sample_output.txt",
            "checkpoint_activations": False,
            "partition_activations": False,
            "no_load_optim": True,
        }

        param_dict.update(fixed_params)
        model, _, _, _, args_loaded = model_setup(None, param_dict, clear_data=True)
        model.eval()

        prompt = param_dict.pop("prompt")
        prompts = [prompt for _ in range(args_loaded.num_samples)]
        output = generate_samples_from_prompt(
            neox_args=args_loaded,
            model=model,
            text=prompts,
            maximum_tokens=args_loaded.maximum_tokens,
            recompute=False,
            temperature=args_loaded.temperature,
            top_k=args_loaded.top_k,
            top_p=args_loaded.top_p,
        )

        # outputs only get generated on mp rank 0
        if is_mp_rank_0():
            assert len(output) == len(prompts)
            for prompt, out in zip(prompts, output):
                assert prompt == out["context"]
                assert len(out["text"]) > 0

        # Clean up
        del model
