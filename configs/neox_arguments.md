Arguments for gpt-neox. All of the following can be specified in your .yml config file(s):


## NeoXArgsLRScheduler

LR Scheduler Arguments



- **lr_decay_style**: typing.Literal['constant', 'linear', 'cosine', 'exponential']

    Default = linear

    Learning rate decay function. Choose from 'constant', 'linear', 'cosine', 'exponential'.


- **lr_decay_iters**: int

    Default = None

    Number of iterations to decay learning rate over. If None, defaults to 
    --train-iters or the equivalent inferred value from train_epochs.

- **lr_decay_fraction**: float

    Default = None

    Effective fraction of training over which to decay lr. Overrides lr_decay_iters. 
    Useful when specifying train_epochs.

- **min_lr**: float

    Default = 0.0

    Minimum value for learning rate. The scheduler clips values below this threshold.



- **warmup**: float

    Default = 0.01

    Percentage of total iterations to warmup on (.01 = 1 percent of all training iters).



- **override_lr_scheduler**: bool

    Default = False

    Reset the values of the scheduler (learning rate,warmup iterations, minimum learning rate, maximum number of iterations, and decay style from input arguments and ignore values from checkpoints. Note that all the above values will be reset.



- **use_checkpoint_lr_scheduler**: bool

    Default = False

    Use checkpoint to set the values of the scheduler (learning rate, warmup iterations, minimum learning rate, maximum number of iterations, and decay style from checkpoint and ignore input arguments.



## NeoXArgsLogging

Logging Arguments



- **use_wandb**: bool

    Default = None

    Flag indicating if wandb is to be used.



- **wandb_group**: str

    Default = None

    Weights and Biases group name - used to group together "runs".



- **wandb_team**: str

    Default = None

    Team name for Weights and Biases.



- **wandb_project**: str

    Default = neox

    wandb project name



- **wandb_host**: str

    Default = https://api.wandb.ai

    url of the wandb host



- **wandb_init_all_ranks**: bool

    Default = False

    Initialize wandb on all ranks.



- **git_hash**: str

    Default = 62c9738a

    current git hash of repository



- **log_dir**: str

    Default = None

    Directory to save logs to.



- **tensorboard_dir**: str

    Default = None

    Write TensorBoard logs to this directory.



- **use_comet**: bool

    Default = None

    Flag indicating if comet is to be used.



- **comet_workspace**: Optional

    Default = None

    Comet workspace name, if not configured Comet Experiments will be created in the user configured default workspace.



- **comet_project**: Optional

    Default = None

    Comet project name, if not configured Comet Experiments will be created in the Uncategorized Experiments project.



- **comet_experiment_name**: Optional

    Default = None

    Custom name for the Comet experiment. If not provided, a random name is used.



- **comet_tags**: Optional

    Default = None

    List of tags to attach to the created Comet Experiment.



- **comet_others**: Optional

    Default = None

    Custom metadata to attach to the created Comet Experiment.



- **log_interval**: int

    Default = 100

    Interval between logging.



- **log_grad_pct_zeros**: bool

    Default = False

    Log the percentage of zeros for the gradient of each parameter to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.



- **log_param_norm**: bool

    Default = False

    Log the frob norm of the parameters to wandb / tensorboard (useful for debugging). Needs wandb_init_all_ranks set to True if using pipeline parallelism to log all ranks.



- **log_grad_norm**: bool

    Default = False

    Log the frob norm of the gradients to wandb / tensorboard (useful for debugging).
    (N.B - this will only work with pp = 0 for now, as we don't have access to the gradients of the model because
    deepspeed.)



- **log_optimizer_states**: bool

    Default = False

    Log the frob norm of the optimizer states to wandb / tensorboard (useful for debugging).



- **log_gradient_noise_scale**: bool

    Default = False

    Whether to log the gradient noise scale when training (cf. https://arxiv.org/abs/1812.06162 for explanation)



- **gradient_noise_scale_n_batches**: int

    Default = 5

    Number of batches to accumulate gradients for in the gradient noise scale logger.



- **gradient_noise_scale_cpu_offload**: bool

    Default = False

    Whether to offload the buffered gradients to cpu when measuring gradient noise scale.



- **memory_profiling**: bool

    Default = False

    Whether to take a memory snapshot of the model. Useful for debugging memory issues.



- **memory_profiling_path**: str

    Default = None

    Path to save memory snapshot to.



- **profile**: bool

    Default = False

    Enable nsys profiling. When using this option,
    nsys options should be specified in commandline.
    An example nsys commandline is
    ```
    nsys profile -s none -t nvtx,cuda -o <path/to/output_file>
    --force-overwrite true
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    ```



- **profile_step_start**: int

    Default = 10

    Step to start profiling at.



- **profile_step_stop**: int

    Default = 12

    Step to stop profiling at.



## NeoXArgsModel

Model Arguments



- **precision**: typing.Literal['fp16', 'fp32', 'bfloat16']

    Default = None

    description of the used precision, either one of fp16 or fp32 (and in the future bf16).



- **num_layers**: int

    Default = None

    Number of transformer layers.



- **hidden_size**: int

    Default = None

    Transformer hidden size.
    When using muP, this is d_model



- **intermediate_size**: int

    Default = None

    Transformer intermediate size. Default = 4h



- **mlp_multiple_of**: int

    Default = 1

    force mlp size to be a multiple of this value



- **expansion_factor**: float

    Default = None

    Transformer intermediate size. Default = 4



- **num_attention_heads**: int

    Default = None

    Number of transformer attention heads.

    If num_kv_heads is set, will control only number of query heads.



- **num_kv_heads**: int

    Default = None

    Number of transformer key/value attention heads.

    If set to None or the same value as num_attention_heads, will perform multi-head attention (MHA).
    If set to < num_attention_heads but > 1, will perform grouped-query attention (GQA) (https://arxiv.org/pdf/2305.13245.pdf)
    If set to 1, will perform multi-query attention.

    Must be < num_attention_heads and divide num_attention_heads evenly.



- **seq_length**: int

    Default = None

    Maximum sequence length to process.



- **sliding_window_width**: int

    Default = None

    Width of the attention sliding window. Only supported with Flash Attention 2.



- **max_position_embeddings**: int

    Default = None

    Maximum number of position embeddings to use. This is the size of position embedding.



- **norm**: typing.Literal['layernorm', 'rmsnorm', 'scalenorm', 'te_rmsnorm', 'te_layernorm']

    Default = layernorm

    Normalization layer to use. Choose from "layernorm", "rmsnorm", "scalenorm", "te_rmsnorm", "te_layernorm".



- **layernorm_fusion**: bool

    Default = False

    Use fused layer norm kernel (if `norm` is `layernorm`).



- **rmsnorm_fusion**: bool

    Default = False

    Use fused RMS norm kernel (if `norm` is `rmsnorm`).



- **use_qk_layernorm**: bool

    Default = False

    Use QK Normalization



- **layernorm_epsilon**: float

    Default = 1e-05

    Layer norm epsilon.



- **rms_norm_epsilon**: float

    Default = 1e-08

    Root mean squared norm epsilon



- **scalenorm_epsilon**: float

    Default = 1e-08

    Scalenorm epsilon



- **pos_emb**: typing.Literal['learned', 'rotary', 'sinusoidal', 'rpe', 'alibi', 'none']

    Default = learned

    Type of positional embedding to use - choose from 'learned', 'rotary', 'sinusoidal', 'rpe', 'none'



- **rpe_num_buckets**: int

    Default = 32

    T5 relative positional encoding number of buckets, default 32.



- **rpe_max_distance**: int

    Default = 128

    T5 relative positional encoding max distance, default 128.



- **opt_pos_emb_offset**: int

    Default = 0

    Learned position embedding offset (only used by OPT, where it should be set to 2).



- **no_weight_tying**: bool

    Default = False

    Disables weight tying between embedding weights and final Linear layer



- **attention_config**: list

    Default = None

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



- **sparsity_config**: dict

    Default = None

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



- **num_unique_layers**: int

    Default = None

    Number of unique transformer layers. num-layers should be divisible by this value. Currently only has an effect when pipe_parallel_size=0.



- **param_sharing_style**: str

    Default = grouped

    Ordering of the shared parameters. For example, for a num-layers=4 and --num-unique-layers=2, we will have the following ordering for two unique layers 1 and 2-: grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].



- **make_vocab_size_divisible_by**: int

    Default = 128

    Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.



- **activation**: typing.Literal['gelu', 'geglu', 'relu', 'softsign', 'swish', 'mish', 'silu', 'reglu', 'swiglu', 'bilinear', 'glu']

    Default = gelu

    Activation function to use - choose from ["gelu", "geglu", "relu", "softsign", "swish", "mish", "silu", "reglu", "swiglu", "bilinear", "glu"]



- **use_flashattn_swiglu**: bool

    Default = False

    Use flash attention's version of swiglu



- **scaled_upper_triang_masked_softmax_fusion**: bool

    Default = False

    Enable fusion of query_key_value_scaling time (upper diagonal) masking and softmax.



- **scaled_masked_softmax_fusion**: bool

    Default = False

    Enable fusion of query_key_value_scaling general masking and softmax.



- **bias_gelu_fusion**: bool

    Default = False

    Enable bias and gelu fusion.



- **bias_dropout_fusion**: bool

    Default = False

    Enable bias and dropout fusion.



- **rope_fusion**: bool

    Default = False

    Enable rotary embedding fusion.



- **fp16_lm_cross_entropy**: bool

    Default = False

    Move the cross entropy unreduced loss calculation for lm head to fp16.



- **init_method_std**: float

    Default = 0.02

    Standard deviation of the zero mean normal distribution used for weight initialization.
    When using muP this is the base std



- **apply_query_key_layer_scaling**: bool

    Default = False

    Scale Q * K^T by 1 / layer-number. If this flag is set, then it will automatically set attention-softmax-in-fp32 to true



- **use_cpu_initialization**: bool

    Default = False

    If set, affine parallel weights initialization uses CPU



- **attention_softmax_in_fp32**: bool

    Default = False

    Run attention masking and softmax in fp32.



- **rotary_pct**: float

    Default = 1.0

    pct of hidden dims to apply rotary positional embedding to



- **rotary_emb_base**: int

    Default = 10000

    Base for rotary positional embedding



- **rotary_save_freqs_buffer**: bool

    Default = False

    Used to control whether the `inv_freqs` buffer in rotary embeddings
    will be stored in checkpoints (persistent=True) or not.

    Defaults to false, but is left configurable to maintain backward-compatibility
    with GPT-NeoX checkpoints that were trained with this flag.



- **init_method**: typing.Literal['normal', 'scaled_normal', 'orthogonal', 'scaled_orthogonal', 'xavier_uniform', 'xavier_normal', 'wang_init', 'small_init', 'single_residual_scaled_normal']

    Default = normal

    Init function used on all layers except ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]



- **output_layer_init_method**: typing.Literal['normal', 'scaled_normal', 'orthogonal', 'scaled_orthogonal', 'xavier_uniform', 'xavier_normal', 'wang_init', 'small_init', 'single_residual_scaled_normal']

    Default = scaled_normal

    Init function used for ff residual outputs - choose from
    ["normal", "scaled_normal", "orthogonal", "scaled_orthogonal", "xavier_uniform", "xavier_normal", "wang_init", "small_init"]



- **gmlp_attn_dim**: int

    Default = 64

    the dimension of the single head self attention in gmlp model (not used in gpt models).
    If None - gmlp model doesn't use attention.



- **gpt_j_residual**: bool

    Default = False

    If false, we use the conventional residual path:
      x = x + attn(ln1(x))
      x = x + mlp(ln2(x))
    Otherwise, we use the residual path from GPT-J, which offers a slight speedup:
      x = ln(x)
      x = x + attn(x) + mlp(x)



- **gpt_j_tied**: bool

    Default = False

    If false, we use
      x = x + attn(ln1(x)) + mlp(ln2(x))
    Otherwise, we tie the layer norms
      y = ln(x)
      x = x + attn(y) + mlp(y)



- **use_bias_in_norms**: bool

    Default = True

    If false, norms (e.g. LayerNorm) will not have bias terms



- **use_bias_in_attn_linear**: bool

    Default = True

    If false, attn_linear (e.g. QKVO) will not have bias terms



- **use_bias_in_mlp**: bool

    Default = True

    If false, mlps will not have bias terms



- **soft_prompt_tuning**: dict

    Default = None

    Dictionary configuring the soft prompt tuning parameters.
    If enabled, will train *only* the soft prompt, and freezes the rest of the model.
    parameters in the dict are:
        'enabled': bool = True # enables soft prompting
        'num_tokens': int = 10 # length of the soft prompt in tokens
        'init_string': str = '' # if provided, initialize the soft prompt with the word embeddings of this string
        'init_range': float = 0.5 # if no init string is provided, initialize the soft prompt with a uniform distribution between -init_range and init_rang



- **mamba_selective_scan_fusion**: bool

    Default = False

    Enable fused kernels for Mamba selective scan.



- **mamba_causal_conv_fusion**: bool

    Default = False

    Enable fused kernels for Mamba causal Conv1d.



- **mamba_inner_func_fusion**: bool

    Default = False

    Enable fused inner operator for Mamba. (Supersedes conv. and selective scan fusion flags, requires each of those kernels to be installed.)



- **mamba_selective_fp32_params**: bool

    Default = True

    Keep selected parameters in fp32 for Mamba (A and D).
    Requires https://github.com/EleutherAI/DeeperSpeed/pull/61 .



- **mamba_use_bias_in_conv**: bool

    Default = True

    If false, conv1d in mamba block will not have bias term



- **mamba_use_bias_in_linears**: bool

    Default = False

    Enable bias terms in mamba block up- and down- projections (in_proj and out_proj).



- **output_layer_parallelism**: typing.Literal['column']

    Default = column

    Parameter controlling whether the output layer is parallelized over the hidden dim (row) or the vocab dim (column)



- **dim_att**: int

    Default = None

    Total dimension of the attention mechanism for RWKV. If not set, defaults to hidden_size.



- **head_size**: int

    Default = None

    Size of each attention head for RWKV. Calculated as dim_att // num_attention_heads.



- **ffn_dim**: int

    Default = None

    Dimension of the feed-forward network for RWKV. If not set, calculated based on hidden_size and expansion_factor.


## NeoXArgsOptimizer

Optimizer Arguments



- **optimizer_type**: typing.Literal['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'madgrad_wd', 'sgd', 'lion']

    Default = adam

    Type of optimizer to use. Choose from ['adam', 'onebitadam', 'cpu_adam', 'cpu_torch_adam', 'sm3', 'madgrad_wd', 'sgd', 'lion']
    NOTE: sgd will use MuSGD from Mup. Mup must be enabled for this optimizer.



- **use_bnb_optimizer**: bool

    Default = False

    Whether to enable the bitsandbytes optimizers



- **zero_stage**: typing.Union[int, typing.List[int], typing.Literal['all']]

    Default = None

    Zero Optimizer stage



- **zero_reduce_scatter**: bool

    Default = None

    Zero: Uses reduce or reduce scatter instead of allreduce to average gradients



- **zero_contiguous_gradients**: bool

    Default = None

    Zero: Copies the gradients to a contiguous buffer as they are produced. Avoids memory fragmentation during backward pass. Only useful when running very large models.



- **zero_reduce_bucket_size**: int

    Default = None

    Zero: Number of elements reduced/allreduced at a time. Limits the memory required for the allgather for large model sizes



- **zero_allgather_bucket_size**: int

    Default = None

    Zero: Number of elements allgathered at a time. Limits the memory required for the allgather for large model sizes



- **lr**: float

    Default = None

    Max Learning rate during training
    When using muP, this is the base learning rate



## NeoXArgsOther

Misc. Arguments



- **distributed_backend**: str

    Default = nccl

    Which backend to use for distributed training.



- **local_rank**: int

    Default = None

    local rank passed from distributed launcher.



- **rank**: int

    Default = None

    global rank of process being run (passed in via distributed launcher)



- **lazy_mpu_init**: bool

    Default = False

    If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on use-cpu-initialization flag. This is for external DDP manager.



- **short_seq_prob**: float

    Default = 0.1

    Probability of producing a short sequence.



- **eod_mask_loss**: bool

    Default = False

    Mask loss for the end of document tokens.



- **adlr_autoresume**: bool

    Default = False

    Enable auto-resume on adlr cluster.



- **adlr_autoresume_interval**: int

    Default = 1000

    Intervals over which check for auto-resume termination signal



- **seed**: int

    Default = 1234

    Random seed used for python, numpy, pytorch, and cuda.



- **onnx_safe**: bool

    Default = False

    Use workarounds for known problems with Torch ONNX exporter



- **deepscale**: bool

    Default = False

    (Deprecated) enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'



- **deepscale_config**: str

    Default = None

    (Deprecated) deepscale json configuration file.



- **deepspeed_mpi**: bool

    Default = False

    Run via MPI, this will attempt to discover the necessary variables to initialize torch distributed from the MPI environment



- **deepspeed_slurm**: bool

    Default = False

    Run via SLURM, this will attempt to discover the necessary variables to initialize torch distributed from the SLURM environment



- **user_script**: str

    Default = None

    user script to be run



- **iteration**: int

    Default = None

    Set during training



- **do_train**: bool

    Default = None

    Set during training



- **do_valid**: bool

    Default = None

    Set during training



- **do_test**: bool

    Default = None

    Set during training



- **save_iters**: list

    Default = None

    Set during training



- **global_num_gpus**: int

    Default = None

    Set during launching



## NeoXArgsParallelism

Parallelism Arguments



- **pipe_parallel_size**: int

    Default = 0

    Number of pipeline parallel stages. Disable with 0.



- **model_parallel_size**: int

    Default = 1

    Size of the model parallelism.



- **pipe_partition_method**: str

    Default = type:transformer|mlp

    method used to distribute model layers across pipeline stages. Choose from "parameters", which balances the number
    of parameters on each pipeline stage, "uniform", which naively balances the number of layers per stage, or
    "type:[regex]", which balances layers whose class names match [regex]



- **world_size**: int

    Default = None

    Total world size (i.e number of gpus in cluster). Configured post-launch using distributed launcher



- **is_pipe_parallel**: bool

    Default = False

    flag to determine whether pipeline parallelism is on - shouldn't be set by user, is automatically determined
    according to pipeline parallel size.



- **sequence_parallel**: bool

    Default = False

    flag to determine whether Megatron-style Sequence Parallelism (https://arxiv.org/abs/2205.05198)
    (Layernorm inputs and activations are sharded across model parallel group) will be used. Has no effect when model_parallel_size is 1.
    **Set by user, in contrast to neox_args.is_pipe_parallel.**



- **expert_interval**: int

    Default = 2

    Have one MoE layer every expert_interval layers



## NeoXArgsTemplate

NeoXArgsTemplate()



## NeoXArgsTextgen

Text Generation arguments



- **text_gen_type**: str

    Default = None

    How to generate text/sample the model.
    Options: `unconditional`, `input-file`, `interactive`, `precompute`



- **precompute_model_name**: str

    Default = None

    Model name to use for saving precomputed logprobs



- **temperature**: float

    Default = 0.0

    exponential scaling output distribution ("higher == more risk")



- **top_p**: float

    Default = 0.0

    Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.



- **top_k**: int

    Default = 0

    integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.



- **return_logits**: bool

    Default = False

    Boolean for whether to return the logits for generated tokens



- **maximum_tokens**: int

    Default = 64

    maximum number of tokens to be generated



- **prompt_end**: str

    Default = 


    a single prompt's end. Defaults to newline



- **sample_input_file**: str

    Default = None

    Get input from file instead of interactive mode, each line is an input.



- **sample_output_file**: str

    Default = samples.txt

    Output file



- **num_samples**: int

    Default = 1

    Number of samples to generate unconditionally, defaults to 1 and interactive conditional sampling



- **recompute**: bool

    Default = False

    During generation recompute all attention instead of using previously computed keys/values.
    Should be set to true for sparse attention models



- **eval_results_prefix**: str

    Default = 

    prefix to which to save evaluation results - final fp will be {eval_results_prefix}_eval_results_yy-mm-dd-HH-MM.json



- **eval_tasks**: list

    Default = None

    Tasks to evaluate on using lm_eval_harness

    NOTE: Requires internet connection



- **moe_top_k**: int

    Default = 1

    Activate top K experts in MoE



- **use_tutel**: bool

    Default = False

    Use Tutel optimizations in MoE



- **moe_num_experts**: int

    Default = 1

    Number of MoE experts



- **moe_loss_coeff**: float

    Default = 0.1

    Coefficient for MoE loss



- **moe_train_capacity_factor**: float

    Default = 1.0

    The capacity of the expert at train time



- **moe_eval_capacity_factor**: float

    Default = 1.0

    The capacity of the expert at eval time



- **moe_min_capacity**: int

    Default = 4

    The minimum capacity per expert regardless of the capacity_factor



- **moe_token_dropping**: bool

    Default = False

    Whether to drop tokens when exceeding capacity



- **create_moe_param_group**: bool

    Default = True

    Whether to create a separate parameter group for MoE parameters



- **moe_use_residual**: bool

    Default = True

    Whether to use residual in MoE



- **moe_expert_parallel_size**: int

    Default = 1

    Number of parallel experts in MoE



- **moe_type**: str

    Default = megablocks

    Either `deepspeed` or `megablocks`



- **moe_glu**: bool

    Default = False

    Use gated linear units in MoE



- **moe_lbl_in_fp32**: bool

    Default = False

    Whether to compute the load balancing loss in fp32.



- **moe_jitter_eps**: float

    Default = None

    Coefficient for MoE routing jitter. Jitter is
    not used if set to None



- **enable_expert_tensor_parallelism**: bool

    Default = False

    Enable expert tensor parallelism



## NeoXArgsTokenizer

Tokenizer Arguments



- **tokenizer_type**: typing.Literal['GPT2BPETokenizer', 'HFTokenizer', 'HFGPT2Tokenizer', 'SPMTokenizer', 'CharLevelTokenizer', 'TiktokenTokenizer']

    Default = GPT2BPETokenizer

    Type of tokenizer to use - should be one of ["GPT2BPETokenizer", "HFTokenizer", "HFGPT2Tokenizer", "SPMTokenizer", "CharLevelTokenizer", "TiktokenTokenizer"]



- **padded_vocab_size**: int

    Default = None

    Total (padded) vocabulary size of tokenizer. Configured after launching of training,
    as it's dependent on the parallelism size.



## NeoXArgsTraining

Training Arguments



- **data_path**: str

    Default = None

    Path to combined dataset to split.



- **use_shared_fs**: bool

    Default = True

    Whether to use a shared filesystem for data loading. If False, local rank 0 on all nodes will preprocess the data,
    otherwise only global rank 0 will preprocess the data. This is implemented in megatron/data/gpt2_dataset.py::_build_index_mappings.



- **train_data_paths**: list

    Default = None

    List of paths to train datasets.



- **train_label_data_paths**: list

    Default = None

    List of paths to train label datasets (not shifted by 1 yet!).



- **train_reward_data_paths**: list

    Default = None

    List of paths to train reward datasets



- **test_data_paths**: list

    Default = None

    List of paths to test datasets.



- **test_label_data_paths**: list

    Default = None

    List of paths to test label datasets (not shifted by 1 yet!).



- **test_reward_data_paths**: list

    Default = None

    List of paths to test reward datasets



- **valid_data_paths**: list

    Default = None

    List of paths to validation datasets.



- **valid_label_data_paths**: list

    Default = None

    List of paths to validation label datasets (not shifted by 1 yet!).



- **valid_reward_data_paths**: list

    Default = None

    List of paths to validation reward datasets



- **pos_train_data_paths**: list

    Default = None

    



- **neg_train_data_paths**: list

    Default = None

    List of paths to positive and negative training datasets.



- **pos_train_label_data_paths**: list

    Default = None

    



- **neg_train_label_data_paths**: list

    Default = None

    List of paths to positive and negative training label datasets (not shifted by 1 yet!).



- **pos_valid_data_paths**: list

    Default = None

    



- **neg_valid_data_paths**: list

    Default = None

    List of paths to positive and negative validation datasets.



- **pos_valid_label_data_paths**: list

    Default = None

    



- **neg_valid_label_data_paths**: list

    Default = None

    List of paths to positive and negative validation label datasets (not shifted by 1 yet!).



- **pos_test_data_paths**: list

    Default = None

    



- **neg_test_data_paths**: list

    Default = None

    List of paths to positive and negative test datasets.



- **pos_test_label_data_paths**: list

    Default = None

    



- **neg_test_label_data_paths**: list

    Default = None

    List of paths to positive and negative test label datasets (not shifted by 1 yet!).



- **train_data_weights**: list

    Default = None

    List of 'weights' that decide how often to sample from each training dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `train_data_paths`



- **valid_data_weights**: list

    Default = None

    List of 'weights' that decide how often to sample from each validation dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `valid_data_paths`



- **test_data_weights**: list

    Default = None

    List of 'weights' that decide how often to sample from each test dataset when blending datasets. If None, defaults to equal weighting.
    Should be a list the same length as `test_data_paths`



- **weight_by_num_documents**: bool

    Default = False

    If True, Builds dataset weights from a multinomial distribution over groups of data according to the number of
    documents in each group.

    WARNING: setting this to True will override any user provided weights

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details



- **weighted_sampler_alpha**: float

    Default = 1.0

    Alpha value for `weight_by_num_documents`. Only has an effect if `weight_by_num_documents` = True.

    when alpha = 1, the probability of sampling from a given group = n_samples / total_samples
    as alpha -> 0, the probability of sampling from all groups becomes equal, and number of documents has no effect
    as alpha -> inf, the probability of sampling from the groups with *the most samples* -> 1



- **data_impl**: typing.Literal['infer', 'mmap', 'cached']

    Default = infer

    Implementation of indexed datasets, can be one of "infer", "cached", or "mmap"



- **pack_impl**: typing.Literal['packed', 'pack_until_overflow', 'unpacked']

    Default = packed

    Packing implementation, can be one of "packed", "pack_until_overflow", or "unpacked".

    warning: pack_until_overflow is very naive and will likely have issues with pretraining scale datasets



- **dataset_impl**: typing.Literal['gpt2', 'pairwise']

    Default = gpt2

    Dataset implementation, can be one of "gpt2" or "pairwise"



- **train_impl**: typing.Literal['normal', 'dpo', 'rm', 'kto']

    Default = normal

    Training implementation, can be one of "normal", "dpo", "kto", or "rm"



- **dpo_fp32**: bool

    Default = True

    Whether to cast logits to fp32 for DPO loss calculation.



- **dpo_reference_free**: bool

    Default = False

    Whether to use reference-free DPO.



- **dpo_beta**: float

    Default = 0.1

    Beta value for DPO



- **kto_fp32**: bool

    Default = True

    Whether to cast logits to fp32 for KTO loss calculation.



- **kto_desirable_weight**: float

    Default = 1.0

    Weight for desirable loss in KTO. Might help if you have unbalanced desirable and undesirable classes.



- **kto_undesirable_weight**: float

    Default = 1.0

    Weight for undesirable loss in KTO. Might help if you have unbalanced desirable and undesirable classes.



- **kto_beta**: float

    Default = 0.1

    Beta value for KTO



- **allow_chopped**: bool

    Default = True

    WARNING: if your packing impl is packed, this is ignored.

    Allow chopped samples in the dataset.
    (e.g if your sequence length is 1024 and you have a sample of length 1026, it will be chopped to 1024)



- **mmap_warmup**: bool

    Default = False

    Warm up mmap files.



- **save**: str

    Default = None

    Output directory to save checkpoints to.



- **s3_path**: str

    Default = None

    Path to s3 bucket for saving checkpoints.



- **s3_chunk_size**: int

    Default = 104857600

    The number of bytes in each file chunk when uploading to s3. Defaults to 100MiB.



- **config_files**: dict

    Default = None

    Store of original config files mapping config filename to file contents



- **load**: str

    Default = None

    Directory containing a model checkpoint.



- **checkpoint_validation_with_forward_pass**: bool

    Default = False

    save input and output of a forward pass with the checkpoint and validate after load



- **checkpoint_scale**: typing.Literal['linear', 'log']

    Default = linear

    How step at which checkpoints are saved should scale. "linear" implies 1 checkpoint will be saved at every multiple of `checkpoint-factor`,
    while "log" implies that the number of steps between each checkpoint will be multiplied by `checkpoint-factor` at each step, starting from step 1.



- **checkpoint_factor**: int

    Default = None

    Acts as a multiplier on either the "log" or "linear" checkpoint spacing.

    With `checkpoint-scale="linear"`, `checkpoint-factor=20`, and `train-iters=100`, checkpoints will be saved at
    steps [20, 40, 60, 80, 100].

    With `checkpoint-scale="log"`, `checkpoint-factor=2`, and `train-iters=100`, checkpoints will be saved at
    steps [1, 2, 4, 8, 16, 32, 64, 100].

    Note that the last checkpoint step is always saved.



- **extra_save_iters**: list

    Default = None

    Additional iterations when a checkpoint should be saved.
    Must be a list of ints or `None`.



- **no_save_optim**: bool

    Default = False

    Do not save current optimizer.



- **no_save_rng**: bool

    Default = False

    Do not save current rng state.



- **no_load_optim**: bool

    Default = False

    Do not load optimizer when loading checkpoint.



- **no_load_rng**: bool

    Default = False

    Do not load rng state when loading checkpoint.



- **finetune**: bool

    Default = False

    Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.



- **batch_size**: int

    Default = None

    training microbatch size per gpu



- **train_iters**: int

    Default = None

    Number of iterations to run for training.



- **train_epochs**: int

    Default = None

    Number of epochs to run for training. Do not specify both train_epochs and train_iters.
    Not currently compatible with data reweighing, pairwise datasets, and packing other than 'packed'



- **eval_iters**: int

    Default = 100

    Number of iterations to run for evaluation validation/test for.



- **keep_last_n_checkpoints**: int

    Default = None

    Number of last checkpoints to keep



- **eval_interval**: int

    Default = 1000

    Interval between running evaluation on validation set.



- **split**: str

    Default = 969, 30, 1

    Comma_separated list of proportions for training, validation, and test split. For example the split 90,5,5 will use 90% of data for training, 5% for validation and 5% for test.



- **vocab_file**: str

    Default = None

    Path to the vocab file.



- **merge_file**: str

    Default = None

    Path to the BPE merge file.



- **num_workers**: int

    Default = 2

    Dataloader number of workers.



- **exit_interval**: int

    Default = None

    Exit the program after the iteration is divisible by this value.



- **attention_dropout**: float

    Default = 0.0

    Post attention dropout probability.



- **hidden_dropout**: float

    Default = 0.0

    Dropout probability for hidden state transformer.



- **weight_decay**: float

    Default = 0.1

    Weight decay coefficient for L2 regularization.



- **checkpoint_activations**: bool

    Default = False

    Checkpoint activation to allow for training with larger models, sequences, and batch sizes.



- **checkpoint_num_layers**: int

    Default = 1

    Chunk size (number of layers) for checkpointing.



- **deepspeed_activation_checkpointing**: bool

    Default = True

    DEPRECATED - TODO: remove
    Uses activation checkpointing from deepspeed



- **contiguous_checkpointing**: bool

    Default = False

    Contiguous memory checkpointing for activations.



- **checkpoint_in_cpu**: bool

    Default = False

    Move the activation checkpoints to CPU.



- **synchronize_each_layer**: bool

    Default = False

    does a synchronize at the beginning and end of each checkpointed layer.



- **profile_backward**: bool

    Default = False

    Enables backward pass profiling for checkpointed layers.



- **partition_activations**: bool

    Default = False

    Partition Activations across GPUs before checkpointing.



- **clip_grad**: float

    Default = 1.0

    Gradient clipping based on global L2 norm.



- **hysteresis**: int

    Default = 2

    hysteresis for dynamic loss scaling



- **dynamic_loss_scale**: bool

    Default = None

    flag indicating whether dynamic loss scale is used



- **loss_scale**: float

    Default = None

    Static loss scaling, positive power of 2
    values can improve fp16 convergence. If None, dynamic loss scaling is used.



- **loss_scale_window**: float

    Default = 1000.0

    Window over which to raise/lower dynamic scale.



- **min_scale**: float

    Default = 1.0

    Minimum loss scale for dynamic loss scale.



- **char_level_ppl**: bool

    Default = False

    Whether to calculate character level perplexity as well as token level perplexity. (may incur a time cost)



- **use_mup**: bool

    Default = False

    Whether to use muP



- **mup_save**: str

    Default = None

    Path to save results when using muP



- **mup_lr**: float

    Default = None

    An alias parameter for lr,
    if not None will override lr



- **mup_std**: float

    Default = None

    An alias parameter for init_method_std,
    if not None will override init_method_std



- **mup_hidden_size**: int

    Default = None

    An alias parameter for hidden_size,
    if not None will override hidden_size



- **coord_check**: bool

    Default = False

    Whether to generate a "coord check" plot to verify mup's implementation in neox



- **coord_check_nsteps**: int

    Default = 10

    Number of steps to do for the coordinate check



- **coord_check_nseeds**: int

    Default = 5

    Number of repetition for each size in coordinate check



- **save_base_shapes**: bool

    Default = False

    Whether to save base shapes for mup. This will save the shapes to the path specified in base-shapes-file.



- **base_shapes_file**: str

    Default = None

    Path to the base shapes to save to/load from



- **mup_embedding_multiplier**: float

    Default = 1.0

    Embedding output multiplier



- **mup_output_multiplier**: float

    Default = 1.0

    Output logits multiplier



- **mup_width_multiplier**: float

    Default = None

    Manually set the layer width multiplier (d_model/d_model,base)



- **mup_d_model_base**: int

    Default = 256

    d_model,base
    Proxy (base) model's layer width



## NeoXArgsDeepspeedConfig

Args for deepspeed config
    Every argument included here will be included in deepspeed config json
    As of Mar 8 2023, up to date compared to https://www.deepspeed.ai/docs/config-json/



- **deepspeed**: bool

    Default = True

    boolean flag to enable DeepSpeed (Always True)



- **train_batch_size**: int

    Default = None

    The effective training batch size. This is the amount of data samples that leads to one step of model update. train_batch_size is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., train_step_batch_size), the gradient accumulation steps (a.k.a., gradient_accumulation_steps), and the number of GPUs.



- **train_micro_batch_size_per_gpu**: int

    Default = None

    Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, gradient_accumulation_steps is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with gradient_accumulation_steps in the configuration JSON.



- **gradient_accumulation_steps**: int

    Default = 1

    Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, train_step_batch_size is automatically calculated using train_batch_size and number of GPUs. Should not be concurrently specified with train_step_batch_size in the configuration JSON.



- **optimizer**: dict

    Default = None

    dict containing the keys type and params

    type: The optimizer name. DeepSpeed natively supports Adam, AdamW, OneBitAdam, Lamb, and OneBitLamb optimizers (See here for details) and will import other optimizers from torch.

    params: Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for Adam).



- **scheduler**: dict

    Default = None

    dict containing the keys type and params

    type: The scheduler name. See here (https://deepspeed.readthedocs.io/en/latest/schedulers.html) for list of support schedulers.

    params: Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature.



- **fp32_allreduce**: bool

    Default = False

    During gradient averaging perform allreduce with 32 bit values



- **prescale_gradients**: bool

    Default = False

    Scale gradients before doing allreduce



- **gradient_predivide_factor**: float

    Default = 1.0

    Before gradient averaging predivide gradients by a specified factor, can sometimes help with fp16 stability when scaling to large numbers of GPUs



- **sparse_gradients**: bool

    Default = False

    Enable sparse compression of torch.nn.Embedding gradients.



- **fp16**: dict

    Default = None

    Configuration for using mixed precision/FP16 training that leverages NVIDIA’s Apex package.

    Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#fp16-training-options



- **bf16**: dict

    Default = None

    Configuration for using bfloat16 floating-point format as an alternative to FP16. BFLOAT16 requires hardware support (e.g., NVIDIA A100).

    Dictionary options as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options



- **amp**: dict

    Default = None

    Configuration for using automatic mixed precision (AMP) training that leverages NVIDIA’s Apex AMP package.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options



- **gradient_clipping**: float

    Default = 1.0

    Enable gradient clipping with provided value



- **zero_optimization**: dict

    Default = None

    Configuration for using ZeRO optimization.

    Multi-level dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#zero-optimization-options



- **curriculum_learning**: dict

    Default = None

    



- **curriculum_seqlen**: int

    Default = 0

    Internal var for tracking the current seqlen



- **steps_per_print**: int

    Default = 10

    Print train loss every N steps.



- **wall_clock_breakdown**: bool

    Default = False

    Enable timing of the latency of forward/backward/update training phases.



- **dump_state**: bool

    Default = False

    Print out state information of DeepSpeed object after initialization.



- **flops_profiler**: dict

    Default = None

    Configuration for using FLOPS profiler.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#flops-profiler



- **communication_data_type**: bool

    Default = None

    During gradient averaging, perform communication with selected data type. By default it will be determined by selected regime



- **autotuning**: dict

    Default = None

    Configuration for using autotuning.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#autotuning



- **activation_checkpointing**: dict

    Default = None

    Configuration for using activation checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#activation-checkpointing



- **sparse_attention**: dict

    Default = None

    Configuration for using sparse attention.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#sparse-attention



- **data_efficiency**: dict

    Default = None

    Configuration for using data efficiency.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-efficiency



- **tensorboard**: dict

    Default = None

    Configuration for using tensorboard.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#monitoring-module-tensorboard-wandb-csv



- **wandb**: dict

    Default = None

    Configuration for using wandb.



- **csv_monitor**: dict

    Default = None

    Configuration for using csv_monitor.



- **elasticity**: dict

    Default = None

    Configuration for using elastic training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#elastic-training-config-v01-and-v02



- **comms_logger**: dict

    Default = None

    Configuration for using communication logger.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#communication-logging



- **compression_training**: dict

    Default = None

    Configuration for using compression training.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#compression



- **checkpoint**: dict

    Default = None

    Configuration for using checkpointing.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#checkpoint-options



- **data_types**: dict

    Default = None

    Configuration for using data types.

    Dictionary as described in Deepspeed documentation: https://www.deepspeed.ai/docs/config-json/#data-type-options



- **deepspeed_extra_args**: dict

    Default = None

    Dictionary of extra arguments to be included in the yaml config file. This can be used for any argument not included in the above list.



## NeoXArgsDeepspeedRunner

Args for deepspeed runner (deepspeed.launcher.runner).
    Every argument included here will be passed as command line argument to deepspeed.launcher.runner



- **hostfile**: str

    Default = None

    list of hostnames / ssh aliases and the number of GPUs per host

    example file contents:
    worker-1 slots=4
    worker-2 slots=4
    127.0.0 slots=4
    127.0.1 slots=4



- **include**: str

    Default = None

    Specify hardware resources to use during execution. String format is `NODE_SPEC[@NODE_SPEC ...]` where `NODE_SPEC=NAME[:SLOT[,SLOT ...]]`. If `:SLOT` is omitted, include all slots on that host. Example: `"worker-0@worker-1:0,2"` will use all slots. on `worker-0` and slots `[0, 2]` on `worker-1`.



- **exclude**: str

    Default = None

    Specify hardware resources to NOT use during execution. Same format as include



- **num_nodes**: int

    Default = -1

    Total number of worker nodes to run on, this will use the top N hosts from the given hostfile. -1 will use all.



- **num_gpus**: int

    Default = None

    Max number of GPUs to use on each node, will use [0:N) GPU ids on each node. None / not specifying a value will use all.



- **master_port**: int

    Default = 29500

    Port used by PyTorch distributed for communication during training.



- **master_addr**: str

    Default = None

    IP address of node 0, will be inferred via 'hostname -I' if not specified.



- **launcher**: typing.Literal['pdsh', 'openmpi', 'mvapich', 'slurm']

    Default = pdsh

    Launcher backend for multi-node training. Options currently include PDSH, OpenMPI, MVAPICH.



- **force_multi**: bool

    Default = False

    Force multi-node training even if only one node is specified.



- **detect_nvlink_pairs**: bool

    Default = False

    If true, autodetects nvlink pairs and remaps cuda visible devices to place them next to each other. This is an Eleuther addition to deepspeed, and should speed up model parallel training on setups with nvlink pairs when mp=2.



- **autotuning_run**: str

    Default = None

    Either "tune", "run", or `None`.



- **no_ssh_check**: bool

    Default = False

    If true, overrides the default check where DeepSpeed confirms that the headnode is accessible via ssh.



- **comment**: str

    Default = None

    Adds a `--comment` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.



- **account**: str

    Default = None

    Adds a `--account` to the DeepSpeed launch command. In DeeperSpeed this is passed on to the SlurmLauncher as well. Sometimes necessary for cluster rules, or so I've heard.

