# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy

from .data import broadcast_data

from .initialize import is_unitialized
from .initialize import destroy_model_parallel
from .initialize import get_data_parallel_group
from .initialize import get_data_parallel_rank
from .initialize import get_data_parallel_world_size
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank, set_model_parallel_rank
from .initialize import get_model_parallel_src_rank, get_data_parallel_src_rank
from .initialize import get_model_parallel_world_size, set_model_parallel_world_size
from .initialize import get_topology
from .initialize import get_pipe_parallel_group
from .initialize import get_pipe_parallel_rank
from .initialize import get_pipe_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_io_parallel_group
from .initialize import get_expert_tokens_for_rank
from .initialize import get_expert_token_counts_for_rank
from .initialize import initialize_model_parallel
from .initialize import model_parallel_is_initialized
from .initialize import get_fp32_allreduce

from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .layers import VocabParallelEmbedding
from .layers import ParallelRelativePositionBias

from .mappings import copy_to_model_parallel_region
from .mappings import copy_to_expert_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import gather_from_expert_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region
from .mappings import gather_from_sequence_parallel_region
from .mappings import scatter_to_sequence_parallel_region

from .random import checkpoint
from .random import get_cuda_rng_tracker
from .random import model_parallel_cuda_manual_seed

from .utils import divide
from .utils import split_tensor_along_last_dim
