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

import subprocess
from dataclasses import dataclass

try:
    from .template import NeoXArgsTemplate
except ImportError:
    from template import NeoXArgsTemplate

try:
    from typing import List, Literal, Union, Optional, Any
except ImportError:
    from typing_extensions import List, Literal, Union, Optional


ATTENTION_TYPE_CHOICES = [
    "global",
    "local",
    "sparse_fixed",
    "sparse_variable",
    "bigbird",
    "bslongformer",
    "gmlp",
    "amlp",
    "flash",
    "rwkv",
    "mamba",
]


def get_git_commit_hash():
    """Gets the git commit hash of your current repo (if it exists)"""
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = None
    return git_hash


@dataclass
class NeoXArgsParallelism(NeoXArgsTemplate):
    """
    Parallelism Arguments
    """

    pipe_parallel_size: int = 0
    """
    Number of pipeline parallel stages. Disable with 0.
    """

    model_parallel_size: int = 1
    """
    Size of the model parallelism.
    """

    pipe_partition_method: str = "type:transformer|mlp"
    """
    method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number
    of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or
    "type:[regex]", which balances layers whose class names match [regex]
    """

    world_size: int = None
    """
    Total world size (i.e number of gpus in cluster). Configured post-launch using distributed launcher
    """

    is_pipe_parallel: bool = False
    """
    flag to determine whether pipeline parallelism is on - shouldn't be set by user, is automatically determined
    according to pipeline parallel size.
    """

    sequence_parallel: bool = False
    """
    flag to determine whether Megatron-style Sequence Parallelism (https://arxiv.org/abs/2205.05198)
    (Layernorm inputs and activations are sharded across model parallel group) will be used. Has no effect when model_parallel_size is 1.
    **Set by user, in contrast to neox_args.is_pipe_parallel.**
    """

    expert_interval: int = 2
    """
    Have one MoE layer every expert_interval layers
    """


@dataclass
class NeoXArgsModel(NeoXArgsTemplate):
    """
    Model Arguments
    """

    precision: Literal["fp16", "fp32", "bfloat16"] = None
    """
    description of the used precision, either one of fp16 or fp32 (and in the future bf16).
    """

    num_layers: int = None
    """
    Number of transformer layers.
    """

    hidden_size: int = None
    """
    Transformer hidden size.
    """

    intermediate_size: int = None
    """
    Transformer intermediate size. Default = 4h
    """

    mlp_multiple_of: int = 1
    """
    force mlp size to be a multiple of this value
    """

    expansion_factor: float = None
    """
    Transformer intermediate size. Default = 4
    """

    num_attention_heads: int = None
    """
    Number of transformer attention heads.

    If num_kv_heads is set, will control only number of query heads.
    """

    num_kv_heads: int = None
    """
    Number of transformer key/value attention heads.

    If set to None or the same value as num_attention_heads, will perform multi-head attention (MHA).
    If set to < num_attention_heads but > 1, will perform grouped-query attention (GQA) (https://arxiv.org/pdf/2305.13245.pdf)
    If set to 1, will perform multi-query attention.

    Must be < num_attention_heads and divide num_attention_heads evenly.
    """

    seq_length: int = None
    """
    Maximum sequence length to process.
    """

    sliding_window_width: int = None
    """
    Width of the attention sliding window. Only supported with Flash Attention 2.
    """

    max_position_embeddings: int = None
    """
    Maximum number of position embeddings to use. This is the size of position embedding.
    """

    norm: Literal[
        "layernorm", "rmsnorm", "scalenorm", "te_rmsnorm", "te_layernorm"
    ] = "layernorm"
    """
    Normalization layer to use. Choose from "layernorm", "rmsnorm", "scalenorm", "te_rmsnorm", "te_layernorm".
    """

    layernorm_fusion: bool = False
    """
    Use fused layer norm kernel (if `norm` is `layernorm`).
    """

    rmsnorm_fusion: bool = False
    """
    Use fused RMS norm kernel (if `norm` is `rmsnorm`).
    """

    use_qk_layernorm: bool = False
    """
    Use QK Normalization
    """

    layernorm_epsilon: float = 1.0e-5
    """
    Layer norm epsilon.
    """

    rms_norm_epsilon: float = 1.0e-8
    """
    Root mean squared norm epsilon
    """

    scalenorm_epsilon: float = 1.0e-8
    """
    Scalenorm epsilon
    """

    pos_emb: Literal[
        "learned", "rotary", "sinusoidal", "rpe", "alibi", "none"
    ] = "learned"
    """
    Type of positional embedding to use - choose from 'learned', 'rotary', 'sinusoidal', 'rpe', 'none'
    """

    rpe_num_buckets: int = 32
    """
    T5 relative positional encoding number of buckets, default 32.
    """

    rpe_max_distance: int = 128
    """
    T5 relative positional encoding max distance, default 128.
    """

    opt_pos_emb_offset: int = 0
    """
    Learned position embedding offset (only used by OPT, where it should be set to 2).
    """

    no_weight_tying: bool = False
    """
    Disables weight tying between embedding weights and final Linear layer
    """

    attention_config: list = None

    """
    Attention configuration for gpt-neox

    The first item in the list specifies the attention type(s), and should be a list of strings. The second item
    specifies the number of times to repeat those attention types in the full list.

    attention type choices:  [global, local, sparse_fixed, sparse_variable, bslongformer, bigbird, "gmlp", "amlp", "flash", "mamba", "rwkv"]

    So a 12 layer network with only global attention could be specified like:
        [[[`global`], 12]]

    or a 12 layer network with alternating global / local like:
        [[[`global`, `local`], 6]]

    If none is specified, this defaults to
        [[[`global`], n_layers]]
    """

    sparsity_config: dict = None

    """
    Sparsity configuration dict as defined in https://www.deepspeed.ai/docs/config-json/#sparse-attention

    Note that since neox is autoregressive, attention is always "unidirectional" and `horizontal_global_attention` is
    always false.

    The main difference between our sparsity config and deepspeed's is that `mode` is ignored - since it is instead
    specified in attention_config defining each layer.

    An example config is given below:
          "sparse_attention": {
            "block": 16,
            "different_layout_per_head": true,
            "num_local_blocks": 4,
            "num_global_blocks": 1,
            "num_different_global_patterns": 4,
            "num_random_blocks": 0,
            "local_window_blocks": [4],
            "global_block_indices": [0],
            "global_block_end_indices": None,
            "num_sliding_window_blocks": 3
          }
    """

    rwkv_fla: bool = False
    """
    Whether to use the Flash Linear Attention implementation of the RWKV kernel, or the CUDA kernel version.
    """

    num_unique_layers: int = None
    """
    Number of unique transformer layers. num-layers should be divisible by this value. Currently only has an effect when pipe_parallel_size=0.
    """

    param_sharing_style: str = "grouped"
    """
    Ordering of the shared parameters. For example, for a num-layers=4 and --num-unique-layers=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].
    """

    make_vocab_size_divisible_by: int = 128
    """
    Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.
    """

    activation: Literal[
        "gelu",
        "geglu",
        "relu",
        "softsign",
        "swish",
        "mish",
        "silu",
        "reglu",
        "swiglu",
        "bilinear",
        "glu",
    ] = "gelu"
    """
    Activation function to use - choose from ["gelu", "geglu", "relu", "softsign", "swish", "mish", "silu", "reglu", "swiglu", "bilinear", "glu"]
    """

    use_flashattn_swiglu: bool = False
    """
    Use flash attention's version of swiglu
    """

    scaled_upper_triang_masked_softmax_fusion: bool = False
    """
    Enable fusion of query_key_value_scaling time (upper diagonal) masking and softmax.
    """

    scaled_masked_softmax_fusion: bool = False
    """
    Enable fusion of query_key_value_scaling general masking and softmax.
    """

    bias_gelu_fusion: bool = False
    """
    Enable bias and gelu fusion.
    """

    bias_dropout_fusion: bool = False
    """
    Enable bias and dropout fusion.
    """

    rope_fusion: bool = False
    """
    Enable rotary embedding fusion.
    """

    fp16_lm_cross_entropy: bool = False
    """
    Move the cross entropy unreduced loss calculation for lm head to fp16.
    """

    init_method_std: float = 0.02
    """
    Standard deviation of the zero mean normal distribution used for weight initialization.
    """

    apply_query_key_layer_scaling: bool = False
    """
    Scale Q * K^T by 1 / layer-number. If this flag is set, then it will automatically set attention-softmax-in-fp32 to true
    """

    use_cpu_initialization: bool = False
    """
    If set, affine parallel weights initialization uses CPU
    """

    attention_softmax_in_fp32: bool = False
    """
    Run attention masking and softmax in fp32.
    """

    rotary_pct: float = 1.0
    """
    pct of hidden dims to apply rotary positional embedding to
    """

    rotary_emb_base: int = 10000
    """
    Base for rotary positional embedding
    """

    rotary_save_freqs_buffer: bool = False
    """
    Used to control whether the `inv_freqs` buffer in rotary embeddings
    will be stored in checkpoints (persistent=True) or not.

    Defaults to false, but is left configurable to maintain backward-compatibility
    with GPT-NeoX checkpoints that were trained with this flag.
    """

    init_method: Literal[
        "normal",
        "scaled_normal",
        "orthogonal",
        "scaled_orthogonal",
        "xavier_uniform",
        "xavier_normal",
        "wang_init",
        "small_init",
        "single_residual_scaled_normal",
    ] = "normal"
    """
    Init function used on all layers except ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]
    """

    output_layer_init_method: Literal[
        "normal",
        "scaled_normal",
        "orthogonal",
        "scaled_orthogonal",
        "xavier_uniform",
        "xavier_normal",
        "wang_init",
        "small_init",
        "single_residual_scaled_normal",
    ] = "scaled_normal"
    """
    Init function used for ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]
    """

    gmlp_attn_dim: int = 64
    """
    the dimension of the single head self attention in gmlp model (not used in gpt models).
    If None - gmlp model doesn't use attention.
    """

    gpt_j_residual: bool = False
    """
    If false, we use the conventional residual path:
      x = x + attn(ln1(x))
      x = x + mlp(ln2(x))
    Otherwise, we use the residual path from GPT-J, which offers a slight speedup:
      x = ln(x)
      x = x + attn(x) + mlp(x)
    """

    gpt_j_tied: bool = False
    """
    If false, we use
      x = x + attn(ln1(x)) + mlp(ln2(x))
    Otherwise, we tie the layer norms
      y = ln(x)
      x = x + attn(y) + mlp(y)
    """

    use_bias_in_norms: bool = True
    """
    If false, norms (e.g. LayerNorm) will not have bias terms
    """
    use_bias_in_attn_linear: bool = True
    """
    If false, attn_linear (e.g. QKVO) will not have bias terms
    """
    use_bias_in_mlp: bool = True
    """
    If false, mlps will not have bias terms
    """

    soft_prompt_tuning: dict = None
    """
    Dictionary configuring the soft prompt tuning parameters.
    If enabled, will train *only* the soft prompt, and freezes the rest of the model.
    parameters in the dict are:
        'enabled': bool = True # enables soft prompting
        'num_tokens': int = 10 # length of the soft prompt in tokens
        'init_string': str = '' # if provided, initialize the soft prompt with the word embeddings of this string
        'init_range': float = 0.5 # if no init string is provided, initialize the soft prompt with a uniform distribution between -init_range and init_rang
    """

    mamba_selective_scan_fusion: bool = False
    """
    Enable fused kernels for Mamba selective scan.
    """

    mamba_causal_conv_fusion: bool = False
    """
    Enable fused kernels for Mamba causal Conv1d.
    """

    mamba_inner_func_fusion: bool = False
    """
    Enable fused inner operator for Mamba. (Supersedes conv. and selective scan fusion flags, requires each of those kernels to be installed.)
    """

    mamba_selective_fp32_params: bool = True
    """
    Keep selected parameters in fp32 for Mamba (A and D).
    Requires https://github.com/EleutherAI/DeeperSpeed/pull/61 .
    """

    mamba_use_bias_in_conv: bool = True
    """
    If false, conv1d in mamba block will not have bias term
    """

    mamba_use_bias_in_linears: bool = False
    """
    Enable bias terms in mamba block up- and down- projections (in_proj and out_proj).
    """

    # Output layer parallelism over the hidden dim is currently broken (https://github.com/EleutherAI/gpt-neox/issues/905)
    output_layer_parallelism: Literal["column"] = "column"
    """
    Parameter controlling whether the output layer is parallelized over the hidden dim (row) or the vocab dim (column)
    """

    serve_model_weights: bool = False
    """
    If true, serve model weight pointers over a socket connection
    """

    weight_server_port: Union[int, List[int]] = 6000
    """
    Port(s) to serve model weights over
    If an integer is provided, the port for each GPU will be 6000 + global rank
    If a list is provided, the ports will be used in order, e.g. rank0 will be weight_server_port[0]
    """

    online_dataserver_ips: Union[str, List[str]] = "localhost"
    """
    ip addresses to connect to for online data serving, defaults to localhost
    """

    online_dataserver_ports: Union[int, List[int]] = 10000
    """
    Port(s) to connect to for online data serving, defaults to 10000
    """

    te_columnparallel: bool = False
    """
    Use TransformerEngine for RowParallelLinear layer.
    """

    te_rowparallel: bool = False
    """
    Use TransformerEngine for ColumnParallelLinear layer.
    """

    te_layernorm_mlp: bool = False
    """
    Use TransformerEngine for LayerNormMLP layer.
    """

    te_mha: bool = False
    """
    Use TransformerEngine for MultiheadAttention layer.
    """

    te_fp8_format: Literal["e4m3", "hybrid"] = "hybrid"
    """
    Controls the FP8 data format used during forward and backward pass by TransformerEngine.
    Hybrid uses E4M3 during forward pass, E5M2 during backward pass.
    """

    te_fp8_wgrad: bool = True
    """
    When set to False, override FP8 config options and do the wgrad computation
    in higher precision.
    """

    te_fp8_amax_history_len: int = 1
    """
    The length of the amax history window used for scaling factor computation.
    """

    te_fp8_amax_compute_algo: str = "most_recent"
    """
    Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.
    """

    te_fp8_margin: int = 0
    """
    Margin for the scaling factor computation.
    """

    te_fp8_mha: bool = False
    """
    When set to True, use the FP8 implementation of Multi Head Attention.
    """

    dim_att: int = None
    """
    Total dimension of the attention mechanism for RWKV. If not set, defaults to hidden_size.
    """

    head_size: int = None
    """
    Size of each attention head for RWKV. Calculated as dim_att // num_attention_heads.
    """

    ffn_dim: int = None
    """
    Dimension of the feed-forward network for RWKV. If not set, calculated based on hidden_size and expansion_factor.
    """


@dataclass
class NeoXArgsOptimizer(NeoXArgsTemplate):
    """
    Optimizer Arguments
    """

    optimizer_type: Literal[
        "adam",
        "onebitadam",
        "cpu_adam",
        "cpu_torch_adam",
        "sm3",
        "madgrad_wd",
        "sgd",
        "lion",
    ] = "adam"
    """
    Type of optimizer to use. Choose from ['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'madgrad_wd', 'sgd', 'lion']
    NOTE: sgd will use MuSGD from Mup. Mup must be enabled for this optimizer.
    """

    use_bnb_optimizer: bool = False
    """
    Whether to enable the bitsandbytes optimizers
    """

    zero_stage: Union[int, List[int], Literal["all"]] = None
    """
    Zero Optimizer stage
    """

    zero_reduce_scatter: bool = None
    """
    Zero: Uses reduce or reduce scatter instead of allreduce to average gradients
    """

    zero_contiguous_gradients: bool = None
    """
    Zero: Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models.
    """

    zero_reduce_bucket_size: int = None
    """
    Zero: Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes
    """

    zero_allgather_bucket_size: int = None
    """
    Zero: Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes
    """

    lr: float = None
    """
    Max Learning rate during training
    """


@dataclass
class NeoXArgsLRScheduler(NeoXArgsTemplate):
    """
    LR Scheduler Arguments
    """

    lr_decay_style: Literal["constant", "linear", "cosine", "exponential"] = "linear"
    """
    Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.
    """

    lr_decay_iters: int = None
    """
    Number of iterations to decay learning rate over, If None defaults to
    --train-iters or the equivalent inferred valued from train_epochs.
    """

    lr_decay_fraction: float = None
    """
    Effective fraction of training over which to decay lr, overrides lr_decay_iters, useful when specifying train_epochs
    """

    min_lr: float = 0.0
    """
    Minimum value for learning rate. The scheduler clips values below this threshold.
    """

    warmup: float = 0.01
    """
    Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).
    """

    override_lr_scheduler: bool = False
    """
    Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.
    """

    use_checkpoint_lr_scheduler: bool = False
    """
    Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.
    """


@dataclass
class NeoXArgsLogging(NeoXArgsTemplate):
    """
    Logging Arguments
    """

    ### BEGIN WANDB ARGS ###
    use_wandb: bool = None
    """Flag indicating if wandb is to be used."""

    wandb_group: str = None
    """Weights and Biases group name - used to group together "runs"."""

    wandb_run_name: str = None
    """Weights and Biases run name for the current experiment"""

    wandb_team: str = None
    """Team name for Weights and Biases."""

    wandb_project: str = "neox"
    """wandb project name"""

    wandb_host: str = "https://api.wandb.ai"
    """url of the wandb host"""

    wandb_init_all_ranks: bool = False
    """Initialize wandb on all ranks."""
    ### END WANDB ARGS ###

    git_hash: str = get_git_commit_hash()
    """current git hash of repository"""

    log_dir: str = None
    """
    Directory to save logs to.
    """

    ### BEGIN TENSORBOARD ARGS ###
    tensorboard_writer = None
    """
    initialized tensorboard writer
    """

    tensorboard_dir: str = None
    """
    Write TensorBoard logs to this directory.
    """
    ### END TENSORBOARD ARGS ###

    ### BEGIN COMET ARGS ###
    use_comet: bool = None
    """Flag indicating if comet is to be used."""

    comet_workspace: Optional[str] = None
    """
    Comet workspace name, if not configured Comet Experiments will be created in the user configured default workspace.
    """

    comet_project: Optional[str] = None
    """
    Comet project name, if not configured Comet Experiments will be created in the Uncategorized Experiments project.
    """

    comet_experiment_name: Optional[str] = None
    """
    Custom name for the Comet experiment. If not provided, a random name is used.
    """

    comet_tags: Optional[list] = None
    """
    List of tags to attach to the created Comet Experiment.
    """

    comet_others: Optional[dict] = None
    """
    Custom metadata to attach to the created Comet Experiment.
    """

    comet_experiment: Any = None
    """
    Initialized comet experiment object used to log data
    """
    ### END COMET ARGS ###

    peak_theoretical_tflops: float = None
    """
    The peak hardware flops with which to compute MFU and HFU, in units of teraflops. Automatic detection is more trouble than it's worth, so this is left to the user. Helpful table listed at https://github.com/stas00/ml-engineering/tree/master/compute/accelerator#tflops-comparison-table
    """

    log_interval: int = 100
    """
    Interval between logging.
    """

    log_grad_pct_zeros: bool = False
    """
    Log the percentage of zeros for the gradient of each parameter to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.
    """

    log_param_norm: bool = False
    """
    Log the frob norm of the parameters to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.
    """

    log_grad_norm: bool = False
    """
    Log the frob norm of the gradients to wandb / tensorboard (useful for debugging).
    (N.B - this will only work with pp = 0 for now, as we don't have access to the gradients of the model because deepspeed.)
    """

    log_optimizer_states: bool = False
    """
    Log the frob norm of the optimizer states to wandb / tensorboard (useful for debugging).
    """

    log_gradient_noise_scale: bool = False
    """
    Whether to log the gradient noise scale when training (cf. https://arxiv.org/abs/1812.06162 for explanation)
    """

    gradient_noise_scale_n_batches: int = 5
    """
    Number of batches to accumulate gradients for in the gradient noise scale logger.
    """

    gradient_noise_scale_cpu_offload: bool = False
    """
    Whether to offload the buffered gradients to cpu when measuring gradient noise scale.
    """

    ### BEGIN PROFILING ARGS
    memory_profiling: bool = False
    """
    Whether to take a memory snapshot of the model. Useful for debugging memory issues.
    """

    memory_profiling_path: str = None
    """
    Path to save memory snapshot to.
    """

    profile: bool = False
    """
    Enable nsys and pytorch profiling. When using this option with nsys,
    nsys options should be directly specified in commandline.
    An example nsys commandline is
    ```
    nsys profile -s none -t nvtx,cuda -o <path/to/output_file>
    --force-overwrite true
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    ```
    """

    profile_step_start: int = 10
    """
    Step to start profiling at.
    """

    profile_step_stop: int = 12
    """
    Step to stop profiling at.
    """
    ### END PROFILING ARGS ###


@dataclass
class NeoXArgsOther(NeoXArgsTemplate):
    """
    Misc. Arguments
    """

    distributed_backend: str = "nccl"
    """
    Which backend to use for distributed training.
    """

    local_rank: int = None
    """
    local rank passed from distributed launcher.
    """

    rank: int = None
    """
    global rank of process being run (passed in via distributed launcher)
    """

    lazy_mpu_init: bool = False
    """
    If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on use-cpu-initialization flag. This is for external DDP manager.
    """

    short_seq_prob: float = 0.1
    """
    Probability of producing a short sequence.
    """

    eod_mask_loss: bool = False
    """
    Mask loss for the end of document tokens.
    """

    adlr_autoresume: bool = False
    """
    Enable auto-resume on adlr cluster.
    """

    adlr_autoresume_object = None
    """
    imported autoresume
    """

    adlr_autoresume_interval: int = 1000
    """
    Intervals over which check for auto-resume termination signal
    """

    seed: int = 1234
    """
    Random seed used for python, numpy, pytorch, and cuda.
    """

    onnx_safe: bool = False
    """
    Use workarounds for known problems with Torch ONNX exporter
    """

    deepscale: bool = False
    """
    (Deprecated) enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    """

    deepscale_config: str = None
    """(Deprecated) deepscale json configuration file."""

    deepspeed_mpi: bool = False
    """
    Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment
    """

    deepspeed_slurm: bool = False
    """
    Run via SLURM, this will attempt to discover the necessary variables to initialize torch distributed from the SLURM environment
    """

    user_script: str = None
    """
    user script to be run
    """

    iteration: int = None
    """
    Set during training
    """

    do_train: bool = None
    """
    Set during training
    """

    do_valid: bool = None
    """
    Set during training
    """

    do_test: bool = None
    """
    Set during training
    """

    global_num_gpus: int = None
    """
    Set during launching
    """


@dataclass
class NeoXArgsTokenizer(NeoXArgsTemplate):
    """
    Tokenizer Arguments
    """

    tokenizer_type: Literal[
        "GPT2BPETokenizer",
        "HFTokenizer",
        "HFGPT2Tokenizer",
        "SPMTokenizer",
        "CharLevelTokenizer",
        "TiktokenTokenizer",
    ] = "GPT2BPETokenizer"
    """
    Type of tokenizer to use - should be one of ["GPT2BPETokenizer", "HFTokenizer", "HFGPT2Tokenizer", "SPMTokenizer", "CharLevelTokenizer", "TiktokenTokenizer"]
    """

    padded_vocab_size: int = None
    """
    Total (padded) vocabulary size of tokenizer. Configured after launching of training,
    as it's dependent on the parallelism size.
    """

    tokenizer = None
    """
    tokenizer object loaded into memory and accessible by other functions
    """


@dataclass
class NeoXArgsTraining(NeoXArgsTemplate):
    """
    Training Arguments
    """

    data_path: str = None
    """
    Path to combined dataset to split.
    """

    use_shared_fs: bool = True
    """
    Whether to use a shared filesystem for data loading. If False, local rank 0 on all nodes will preprocess the data,
    otherwise only global rank 0 will preprocess the data. This is implemented in megatron/data/gpt2_dataset.py::_build_index_mappings.
    """

    train_data_paths: list = None
    """
    List of paths to train datasets.
    """

    train_label_data_paths: list = None
    """
    List of paths to train label datasets (not shifted by 1 yet!).
    """

    train_reward_data_paths: list = None
    """
    List of paths to train reward datasets
    """

    test_data_paths: list = None
    """
    List of paths to test datasets.
    """

    test_label_data_paths: list = None
    """
    List of paths to test label datasets (not shifted by 1 yet!).
    """

    test_reward_data_paths: list = None
    """
    List of paths to test reward datasets
    """

    valid_data_paths: list = None
    """
    List of paths to validation datasets.
    """

    valid_label_data_paths: list = None
    """
    List of paths to validation label datasets (not shifted by 1 yet!).
    """

    valid_reward_data_paths: list = None
    """
    List of paths to validation reward datasets
    """

    pos_train_data_paths: list = None
    neg_train_data_paths: list = None
    """
    List of paths to positive and negative training datasets.
    """

    pos_train_label_data_paths: list = None
    neg_train_label_data_paths: list = None
    """
    List of paths to positive and negative training label datasets (not shifted by 1 yet!).
    """

    pos_valid_data_paths: list = None
    neg_valid_data_paths: list = None
    """
    List of paths to positive and negative validation datasets.
    """

    pos_valid_label_data_paths: list = None
    neg_valid_label_data_paths: list = None
    """
    List of paths to positive and negative validation label datasets (not shifted by 1 yet!).
    """

    pos_test_data_paths: list = None
    neg_test_data_paths: list = None
    """
    List of paths to positive and negative test datasets.
    """

    pos_test_label_data_paths: list = None
    neg_test_label_data_paths: list = None
    """
    List of paths to positive and negative test label datasets (not shifted by 1 yet!).
    """

    train_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each training dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `train_data_paths`
    """

    valid_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each validation dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `valid_data_paths`
    """

    test_data_weights: list = None
    """
    List of 'weights' that decide how often to sample from each test dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `test_data_paths`
    """

    weight_by_num_documents: bool = False
    """
    If True, Builds dataset weights from a multinomial distribution over groups of data according to the number of
    documents in each group.

    WARNING: setting this to True will override any user provided weights

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """

    weighted_sampler_alpha: float = 1.0
    """
    Alpha value for `weight_by_num_documents`. Only has an effect if `weight_by_num_documents` = True.

    when alpha = 1, the probability of sampling from a given group = n_samples / total_samples
    as alpha -> 0, the probability of sampling from all groups becomes equal, and number of documents has no effect
    as alpha -> inf, the probability of sampling from the groups with *the most samples* -> 1
    """

    data_impl: Literal["infer", "mmap", "cached"] = "infer"
    """
    Implementation of indexed datasets, can be one of "infer", "cached", or "mmap"
    """

    pack_impl: Literal["packed", "pack_until_overflow", "unpacked"] = "packed"
    """
    Packing implementation, can be one of "packed", "pack_until_overflow", or "unpacked".

    warning: pack_until_overflow is very naive and will likely have issues with pretraining scale datasets
    """

    dataset_impl: Literal["gpt2", "pairwise", "online"] = "gpt2"
    """
    Dataset implementation, can be one of "gpt2", "pairwise", or "online"
    """

    train_impl: Literal["normal", "dpo", "rm", "kto", "reinforce"] = "normal"
    """
    Training implementation, can be one of "normal", "dpo", "kto", "reinforce", or "rm"
    """

    dpo_fp32: bool = True
    """
    Whether to cast logits to fp32 for DPO loss calculation.
    """

    dpo_reference_free: bool = False
    """
    Whether to use reference-free DPO.
    """

    dpo_beta: float = 0.1
    """
    Beta value for DPO
    """

    kto_fp32: bool = True
    """
    Whether to cast logits to fp32 for KTO loss calculation.
    """

    kto_desirable_weight: float = 1.0
    """
    Weight for desirable loss in KTO. Might help if you have unbalanced desirable and undesirable classes.
    """

    kto_undesirable_weight: float = 1.0
    """
    Weight for undesirable loss in KTO. Might help if you have unbalanced desirable and undesirable classes.
    """

    z_loss: float = 0.0
    """
    Z-loss parameter, only implemented for RM training currently.
    https://arxiv.org/pdf/2204.02311
    https://arxiv.org/pdf/2309.10305
    """

    kto_beta: float = 0.1
    """
    Beta value for KTO
    """

    fp32_reinforce: bool = True
    """
    Whether to cast logits to fp32 for Reinforce loss calculation.
    """

    kl_impl: Literal["abs", "mse", "kl", "full"] = "mse"
    """
    KL divergence implementation, can be one of "abs", "mse", "kl", or "full"
    """

    kl_div_beta: float = 0.1
    """
    Beta value for KL divergence in Reinforce loss calculation.
    """

    reinforce_leave_one_out: bool = False
    """
    Whether to use reinforce leave one out for training
    (from https://arxiv.org/abs/2402.14740 and https://api.semanticscholar.org/CorpusID:198489118)
    """

    allow_chopped: bool = True
    """
    WARNING: if your packing impl is packed, this is ignored.

    Allow chopped samples in the dataset.
    (e.g if your sequence length is 1024 and you have a sample of length 1026, it will be chopped to 1024)
    """

    mmap_warmup: bool = False
    """
    Warm up mmap files.
    """

    save: str = None
    """
    Output directory to save checkpoints to.
    """

    s3_path: str = None
    """
    Path to s3 bucket for saving checkpoints.
    """

    s3_chunk_size: int = 104_857_600
    """
    The number of bytes in each file chunk when uploading to s3. Defaults to 100MiB.
    """

    config_files: dict = None
    """
    Store of original config files mapping config filename to file contents
    """

    load: str = None
    """
    Directory containing a model checkpoint.
    """

    checkpoint_validation_with_forward_pass: bool = False
    """
    save input and output of a forward pass with the checkpoint and validate after load
    """

    checkpoint_scale: Literal["linear", "log"] = "linear"
    """
    How step at which checkpoints are saved should scale. "linear" implies 1 checkpoint will be saved at every multiple of `checkpoint-factor`,
    while "log" implies that the number of steps between each checkpoint will be multiplied by `checkpoint-factor` at each step, starting from step 1.
    """

    checkpoint_factor: Union[int, float] = None
    """
    Acts as a multiplier on either the "log" or "linear" checkpoint spacing.

    With `checkpoint-scale="linear"`, `checkpoint-factor=20`, and `train-iters=100`, checkpoints will be saved at
    steps [20, 40, 60, 80, 100].

    With `checkpoint-scale="log"`, `checkpoint-factor=2`, and `train-iters=100`, checkpoints will be saved at
    steps [1, 2, 4, 8, 16, 32, 64, 100].

    Note that the last checkpoint step is always saved.
    """

    extra_save_iters: list = None
    """
    Additional iterations when a checkpoint should be saved.
    Must be a list of ints or `None`.
    """

    no_save_optim: bool = False
    """
    Do not save current optimizer.
    """

    no_save_rng: bool = False
    """
    Do not save current rng state.
    """

    no_load_optim: bool = False
    """
    Do not load optimizer when loading checkpoint.
    """

    no_load_rng: bool = False
    """
    Do not load rng state when loading checkpoint.
    """

    finetune: bool = False
    """
    Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
    """

    batch_size: int = None
    """
    training microbatch size per gpu
    """

    train_iters: int = None
    """
    Number of iterations to run for training.
    """

    train_epochs: int = None
    """
    Number of epochs to run for training. Do not specify both train_epochs and train_iters.
    Not currently compatible with data reweighing, pairwise datasets, and packing other than 'packed'
    """

    eval_iters: int = 100
    """
    Number of iterations to run for evaluation validation/test for.
    """

    keep_last_n_checkpoints: int = None
    """
    Number of last checkpoints to keep
    """

    eval_interval: int = 1000
    """
    Interval between running evaluation on validation set.
    """

    split: str = "969, 30, 1"
    """
    Comma_separated list of proportions for training, validation, and test split. For example the split 90,5,5 will use 90% of data for training, 5% for validation and 5% for test.
    """

    vocab_file: str = None
    """
    Path to the vocab file.
    """

    merge_file: str = None
    """
    Path to the BPE merge file.
    """

    num_workers: int = 2
    """
    Dataloader number of workers.
    """

    exit_interval: int = None
    """
    Exit the program after the iteration is divisible by this value.
    """

    attention_dropout: float = 0.0
    """
    Post attention dropout probability.
    """

    hidden_dropout: float = 0.0
    """
    Dropout probability for hidden state transformer.
    """

    weight_decay: float = 0.1
    """
    Weight decay coefficient for L2 regularization.
    """

    checkpoint_activations: bool = False
    """
    Checkpoint activation to allow for training with larger models, sequences, and batch sizes.
    """

    checkpoint_num_layers: int = 1
    """
    Chunk size (number of layers) for checkpointing.
    """

    deepspeed_activation_checkpointing: bool = True
    """
    DEPRECATED - TODO: remove
    Uses activation checkpointing from deepspeed
    """

    contiguous_checkpointing: bool = False
    """
    Contiguous memory checkpointing for activations.
    """

    checkpoint_in_cpu: bool = False
    """
    Move the activation checkpoints to CPU.
    """

    synchronize_each_layer: bool = False
    """
    does a synchronize at the beginning and end of each checkpointed layer.
    """

    profile_backward: bool = False
    """
    Enables backward pass profiling for checkpointed layers.
    """

    partition_activations: bool = False
    """
    Partition Activations across GPUs before checkpointing.
    """

    clip_grad: float = 1.0
    """
    Gradient clipping based on global L2 norm.
    """

    hysteresis: int = 2
    """
    hysteresis for dynamic loss scaling
    """

    dynamic_loss_scale: bool = None
    """
    flag indicating whether dynamic loss scale is used
    """

    loss_scale: float = None
    """
    Static loss scaling, positive power of 2
    values can improve fp16 convergence. If None, dynamic loss scaling is used.
    """

    loss_scale_window: float = 1000.0
    """
    Window over which to raise/lower dynamic scale.
    """

    min_scale: float = 1.0
    """
    Minimum loss scale for dynamic loss scale.
    """

    char_level_ppl: bool = False
    """
    Whether to calculate character level perplexity as well as token level perplexity. (may incur a time cost)
    """

    use_mup: bool = False
    """
    Whether to use Microsoft's Mup https://github.com/microsoft/mup
    """

    coord_check: bool = False
    """
    Whether to generate a "coord check" plot to verify mup's implementation in neox
    """

    save_base_shapes: bool = False
    """
    Whether to save base shapes for mup. This will save the shapes to the path specified in base-shapes-file.
    """

    base_shapes_file: str = None
    """
    Path to the base shapes to save to/load from
    """

    mup_init_scale: float = 1.0
    """
    Initialization scale: All the parameters are multiplied by this value
    """

    mup_attn_temp: float = 1.0
    """
    Attention temperature: Reciprocal of the multiplier applied to the input to attention softmax
    """

    mup_output_temp: float = 1.0
    """
    Output temperature: Reciprocal of the multiplier applied to the input to softmax that
    produces the distribution over output tokens.
    """

    mup_embedding_mult: float = 1.0
    """
    Scalar by which we multiply the output of the embedding layer
    """

    mup_rp_embedding_mult: float = 1.0
    """
    Scalar by which we multiply vectors representing relative position
    """

    mup_width_scale: int = 2
    """
    What to scale width by when creating the delta model for mup
    """


@dataclass
class NeoXArgsTextgen(NeoXArgsTemplate):
    """
    Text Generation arguments
    """

    text_gen_type: str = None
    """
    How to generate text/sample the model.
    Options: `unconditional`, `input-file`, `interactive`, `precompute`
    """

    precompute_model_name: str = None
    """
    Model name to use for saving precomputed logprobs
    """

    temperature: float = 0.0
    """
    exponential scaling output distribution ("higher == more risk")
    """

    top_p: float = 0.0
    """
    Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    """

    top_k: int = 0
    """
    integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    """

    return_logits: bool = False
    """
    Boolean for whether to return the logits for generated tokens
    """

    maximum_tokens: int = 64
    """
    maximum number of tokens to be generated
    """

    prompt_end: str = "\n"
    """
    a single prompt's end. Defaults to newline
    """

    sample_input_file: str = None
    """
    Get input from file instead of interactive mode, each line is an input.
    """

    sample_output_file: str = "samples.txt"
    """
    Output file
    """

    num_samples: int = 1
    """
    Number of samples to generate unconditionally, defaults to 1 and interactive conditional sampling
    """

    recompute: bool = False
    """
    During generation recompute all attention instead of using previously computed keys/values.
    Should be set to true for sparse attention models
    """

    eval_results_prefix: str = ""
    """
    prefix to which to save evaluation results - final fp will be {eval_results_prefix}_eval_results_yy-mm-dd-HH-MM.json
    """

    eval_tasks: list = None
    """
    Tasks to evaluate on using lm_eval_harness

    NOTE: Requires internet connection
    """

    moe_top_k: int = 1
    """
    Activate top K experts in MoE
    """

    use_tutel: bool = False
    """
    Use Tutel optimizations in MoE
    """

    moe_num_experts: int = 1
    """
    Number of MoE experts
    """

    moe_loss_coeff: float = 0.1
    """
    Coefficient for MoE loss
    """

    moe_train_capacity_factor: float = 1.0
    """
    The capacity of the expert at train time
    """

    moe_eval_capacity_factor: float = 1.0
    """
    The capacity of the expert at eval time
    """

    moe_min_capacity: int = 4
    """
    The minimum capacity per expert regardless of the capacity_factor
    """

    moe_token_dropping: bool = False
    """
    Whether to drop tokens when exceeding capacity
    """

    create_moe_param_group: bool = True
    """
    Whether to create a separate parameter group for MoE parameters
    """

    moe_use_residual: bool = True
    """
    Whether to use residual in MoE
    """

    moe_expert_parallel_size: int = 1
    """
    Number of parallel experts in MoE
    """

    moe_type: str = "megablocks"
    """
    Either `deepspeed` or `megablocks`
    """

    moe_glu: bool = False
    """
    Use gated linear units in MoE
    """

    moe_lbl_in_fp32: bool = False
    """
    Whether to compute the load balancing loss in fp32.
    """

    moe_jitter_eps: float = None
    """
    Coefficient for MoE routing jitter. Jitter is
    not used if set to None
    """

    enable_expert_tensor_parallelism: bool = False
    """
    Enable expert tensor parallelism
    """
