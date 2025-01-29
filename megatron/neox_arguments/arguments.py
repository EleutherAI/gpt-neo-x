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

import base64
import os
from pathlib import Path
import yaml
import json
import logging
import copy
import torch
import argparse
from pkg_resources import packaging
from importlib.metadata import version

from dataclasses import dataclass
from typing import List, Dict
from socket import gethostname

try:
    from typing import Literal, Union
except ImportError:
    from typing_extensions import Literal, Union
from deepspeed.launcher.runner import DLTS_HOSTFILE
from megatron.logging import Tee
from megatron.tokenizer import build_tokenizer
from megatron.utils import obtain_resource_pool, expand_attention_types
from .deepspeed_args import NeoXArgsDeepspeedConfig, NeoXArgsDeepspeedRunner
from .neox_args import (
    NeoXArgsModel,
    NeoXArgsTokenizer,
    NeoXArgsTraining,
    NeoXArgsParallelism,
    NeoXArgsLogging,
    NeoXArgsOther,
    NeoXArgsTextgen,
    NeoXArgsOptimizer,
    NeoXArgsLRScheduler,
    ATTENTION_TYPE_CHOICES,
)

### ANSI escape codes ###
END = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

### Formatted logging prefixes ###
ERROR = f"{RED}[ERROR]{END} "
FAIL = f"{RED}[FAIL]{END}"
INFO = "[INFO]"
OKAY = f"{GREEN}[OKAY]{END}"
SUCCESS = f"{GREEN} [SUCCESS] {END}"
WARNING = f"{YELLOW}[WARNING]{END}"

# ZERO defaults by deespeed
# These values should not be changed unless defaults in deepspeed are changed
# for all zero_optimization options, see https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
ZERO_DEFAULTS = {
    "stage": 0,
    "allgather_partitions": True,
    "reduce_scatter": True,
    "allgather_bucket_size": int(5e8),
    "overlap_comm": False,
    "reduce_scatter": True,
    "reduce_bucket_size": int(5e8),
    "contiguous_gradients": False,
}

# NeoX optimizer defaults
OPT_DEFAULT = "Adam"
OPT_PARAMS_DEFAULTS = {
    "lr": 0.001,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 0,
    "freeze_step": 400,
    "momentum": 0.0,
    "cuda_aware": False,
}


AUTOTUNING_ARGS = (
    "train_batch_size",
    "train_micro_batch_size_per_gpu",
    "gradient_accumulation_steps",
    "zero_optimization",
    "autotuning",
)

BASE_CLASSES = [
    NeoXArgsDeepspeedRunner,
    NeoXArgsDeepspeedConfig,
    NeoXArgsModel,
    NeoXArgsLRScheduler,
    NeoXArgsOptimizer,
    NeoXArgsTokenizer,
    NeoXArgsTraining,
    NeoXArgsParallelism,
    NeoXArgsLogging,
    NeoXArgsTextgen,
    NeoXArgsOther,
]

DEEPSPEED_ARG_CLASSES = [NeoXArgsDeepspeedRunner, NeoXArgsDeepspeedConfig]
NEOX_ARG_CLASSES = [i for i in BASE_CLASSES if i not in DEEPSPEED_ARG_CLASSES]

if "DLTS_HOSTFILE" in os.environ:
    DLTS_HOSTFILE = os.environ["DLTS_HOSTFILE"]


@dataclass
class NeoXArgs(*BASE_CLASSES):
    """
    data class containing all configurations

    NeoXArgs inherits from a number of small configuration classes
    """

    ############################################################################################################################
    # start of instantiation

    def __post_init__(self):
        """
        after initialization of default or loaded values
        a number of functions are performed in order to
        calculate values, assert consistency and do typechecking.
        """
        if not NeoXArgs.validate_keys():
            raise ValueError(
                self.__class__.__name__
                + ".__post_init__() NeoXArgs keys cannot be validated"
            )

        self.enable_logging()

        self.calculate_derived()

        if not self.validate_types():
            raise ValueError(
                self.__class__.__name__
                + ".__post_init__() NeoXArgs types cannot be validated"
            )

        if not self.validate_values():
            raise ValueError(
                self.__class__.__name__
                + ".__post_init__() NeoXArgs values cannot be validated"
            )

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self)

    def initialize_tensorboard_writer(self):
        if self.tensorboard_dir and self.rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter

                print("> setting up tensorboard ...")
                self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_dir)
            except (ModuleNotFoundError, ImportError):
                print(
                    "WARNING: TensorBoard writing requested but is not "
                    "available (are you using PyTorch 1.1.0 or later and do you have tensorboard installed?), "
                    "no TensorBoard logs will be written.",
                    flush=True,
                )

    def initialize_comet(self):
        if self.use_comet and self.rank == 0:
            try:
                import comet_ml

                # Deactivate output logging to avoid any potential interference with Tee
                self.comet_experiment = comet_ml.start(
                    workspace=self.comet_workspace,
                    project=self.comet_project,
                    experiment_config=comet_ml.ExperimentConfig(
                        auto_output_logging=False
                    ),
                )
                self.comet_experiment.__internal_api__log_parameters__(
                    self.all_config,
                    framework="gpt-neox",
                    source="manual",
                    flatten_nested=True,
                )

                if self.comet_experiment_name:
                    self.comet_experiment.set_name(self.comet_experiment_name)

                if self.comet_tags:
                    self.comet_experiment.add_tags(self.comet_tags)

                if self.comet_others:
                    self.comet_experiment.log_others(self.comet_others)

                logging.info("> setting up comet ...")
            except ImportError as e:
                logging.error(
                    f'{FAIL} importing comet. Comet can be installed with "pip install comet_llm". See https://github.com/comet-ml/comet-llm for more info. Full error is:'
                )
                raise e
            except Exception as e:
                logging.error(
                    f'{FAIL} Error setting up Comet. Either set "use_comet: False" in your configuration file, or resolve the issue with Comet. Full error is:',
                )
                raise e

    @classmethod
    def from_ymls(cls, paths_to_yml_files: List[str], overwrite_values: Dict = None):
        """
        instantiates NeoXArgs while reading values from yml files

        paths_to_yml_files: list of paths to yml files

        overwrite_values: If provided, overwrite any values in the yamls with these values
        """

        print(cls.__name__ + ".from_ymls() " + str(paths_to_yml_files), flush=True)

        # initialize an empty config dictionary to be filled by yamls
        config = dict()
        config_files = dict()
        # iterate of all to be loaded yaml files
        for conf_file_name in paths_to_yml_files:
            # load file
            with open(conf_file_name) as conf_file:
                conf = yaml.load(conf_file, Loader=yaml.FullLoader)

            # check for key duplicates and load values
            for conf_key, conf_value in conf.items():
                if conf_key in config:
                    raise ValueError(
                        f"Conf file {conf_file_name} has the following duplicate keys with previously loaded file: {conf_key}"
                    )

                conf_key_converted = conf_key.replace(
                    "-", "_"
                )  # TODO remove replace and update configuration files?
                config[conf_key_converted] = conf_value

            # load original config files to save unchanged with checkpoint
            # saving the original config retains comments
            filename = os.path.basename(conf_file_name)
            assert (
                filename not in config_files
            ), "At least two config files have the same filename. This will result in conflicts when saving out configs with the checkpoint in one single directory. Please use unique names for configs."
            config_files[filename] = open(conf_file_name).read()

        # add config file content to neox args to make them accessible in code
        # this is used when saving checkpoints
        config["config_files"] = config_files

        # Configuration parameters not specified
        params_not_in_config = sorted(
            list(set(cls.__dataclass_fields__.keys()) - set(config.keys()))
        )
        if len(params_not_in_config) > 0:
            logging.debug(
                cls.__name__
                + ".from_ymls() Configuration parameters not specified (using defaults): "
                + ", ".join(params_not_in_config)
            )

        if overwrite_values is not None:
            for k, v in overwrite_values.items():
                config[k] = v

        # instantiate class and return
        # duplicate values and unrecognized keys are again checked upon instantiation
        return cls(**config)

    @classmethod
    def from_dict(cls, args_dict: Dict):
        """
        instantiates NeoXArgs while reading values from input dict
        """
        return cls(**args_dict)

    ############################################################################################################################
    # start of command line args interface

    @classmethod
    def consume_deepy_args(cls, input_args=None):
        """
        entry point for deepy.py configuring and consuming command line arguments.

        We can use `--wandb_group` / `--wandb_team` to overwrite those args from the command line, otherwise the value from the config is taken.
        """

        parser = argparse.ArgumentParser(
            description="GPT-NeoX Configuration", allow_abbrev=False
        )

        group = parser.add_argument_group(title="Training Configuration")

        group.add_argument(
            "user_script",
            type=str,
            help="User script to launch, followed by any required " "arguments.",
        )

        group.add_argument(
            "--conf_dir",
            "-d",
            type=str,
            default=None,
            help="Directory to prefix to all configuration file paths",
        )

        group.add_argument(
            "conf_file",
            type=str,
            nargs="+",
            help="Configuration file path. Multiple files can be provided and will be merged.",
        )

        group = parser.add_argument_group(title="Weights and Biases monitoring args")

        group.add_argument(
            "--wandb_group",
            type=str,
            default=None,
            help='Weights & Biases group name - used to group together "runs".',
        )
        group.add_argument(
            "--wandb_run_name",
            type=str,
            default=None,
            help="Weights & Biases run name for the current experiment.",
        )
        group.add_argument(
            "--wandb_team",
            type=str,
            default=None,
            help="Weights & Biases team name.",
        )

        group = parser.add_argument_group(title="Eval args")

        group.add_argument(
            "--eval_tasks",
            type=str,
            nargs="+",
            default=None,
            help="Optionally overwrite eval tasks to run for eval.py",
        )
        group.add_argument(
            "--iteration",
            type=int,
            default=None,
            help="Iteration to load checkpoint from in the eval.py and generate.py scripts. If None is provided, uses the latest iteration.",
        )
        group.add_argument(
            "--eval_results_prefix",
            type=str,
            default=None,
            help="prefix to append to eval results file",
        )
        parser.add_argument(
            "-H",
            "--hostfile",
            type=str,
            help="Hostfile path (in MPI style) that defines the "
            "resource pool available to the job (e.g., "
            "worker-0 slots=4)",
        )
        group = parser.add_argument_group(title="Generation args")
        group.add_argument(
            "-i",
            "--sample_input_file",
            type=str,
            default=None,
            help="Optionally overwrite `sample_input_file` for generate.py",
        )
        group.add_argument(
            "-o",
            "--sample_output_file",
            type=str,
            default=None,
            help="Optionally overwrite `sample_output_file` for generate.py",
        )

        tuning = parser.add_argument_group(title="DeepSpeed Autotuning")
        tuning.add_argument(
            "--autotuning",
            type=str,
            default=None,
            choices=("tune", "run"),
            help="Use DeepSpeed's autotuning feature to optimize certain hyperparameters. For more details refer to documentation here: https://www.deepspeed.ai/tutorials/autotuning/",
        )
        args_parsed = parser.parse_args(input_args)

        # Validate user_script exists
        assert os.path.exists(
            args_parsed.user_script
        ), f"User script could not be found: {args_parsed.user_script}"

        # load config files
        conf_files = args_parsed.conf_file
        if args_parsed.conf_dir:
            conf_files = [os.path.join(args_parsed.conf_dir, f) for f in conf_files]

        # enables us to pass in `125M` instead of `125M.yml`
        conf_files = [
            (cf if (cf.endswith(".yml") or cf.endswith(".json")) else cf + ".yml")
            for cf in conf_files
        ]

        # determine overwrite values
        overwrite_values = dict()
        for k, v in vars(args_parsed).items():
            if k == "autotuning" and v is not None:
                overwrite_values["autotuning_run"] = v
            elif k not in ["conf_dir", "conf_file"] and v is not None:
                overwrite_values[k] = v

        # load args
        neox_args = cls.from_ymls(
            paths_to_yml_files=conf_files, overwrite_values=overwrite_values
        )

        if neox_args.use_wandb:
            try:
                import wandb

                # Check if the W&B group name is configured
                if neox_args.wandb_group is None:
                    # Set a randomized string as group name if no group name is provided
                    neox_args.wandb_group = wandb.sdk.lib.runid.generate_id()
                else:
                    # Concatenate the W&B group name with a randomized string to ensure uniqueness.
                    neox_args.wandb_group += "_" + wandb.sdk.lib.runid.generate_id()
            except ModuleNotFoundError as e:
                if e.name == "wandb":
                    e.msg += "\nWeights & Biases monitoring was requested but `wandb` was not found. Install `wandb` to use Weights & Biases, or set the `use_wandb` configuration option to a boolean false to disable Weights & Biases logging."
                raise e

            neox_args.wandb_group += "_" + wandb.util.generate_id()

        neox_args.print()

        return neox_args

    @classmethod
    def consume_neox_args(cls, overwrite_values=None, input_args=None):
        """
        Deepspeed launcher needs to pass the arguments for `pretrain_gpt2.py` across to all machines.

        In order not to have any problems with different configs being mismatched across machines, we instead read the .yaml configuration file from the main rank,
        then serialize the arguments to a dictionary, which the deepspeed launcher broadcasts to all machines (`--megatron_config`).

        We then instantiate a new NeoXArgs from the dictionary (`.from_dict`). This should ensure args are never inconsistent across machines.
        """

        parser = argparse.ArgumentParser(
            description="GPT-NeoX Configuration", allow_abbrev=False
        )
        parser.add_argument(
            "--megatron_config",
            type=str,
            default=None,
            help="json dict dumped as string in NeoXArgs.get_deepspeed_main_args()",
        )
        parser.add_argument(
            "--deepspeed_config",
            type=str,
            default=None,
            help="Only need this (at this stage) for autotuning",
        )
        args_parsed, _ = parser.parse_known_args(input_args)
        megatron_config = json.loads(
            base64.urlsafe_b64decode(args_parsed.megatron_config).decode("utf-8")
        )
        if args_parsed.deepspeed_config is not None:
            overwrite_values = cls.set_up_autotuning(
                args_parsed.deepspeed_config, overwrite_values
            )
        if overwrite_values is not None:
            megatron_config.update(overwrite_values)
        return cls.from_dict(args_dict=megatron_config)

    @staticmethod
    def set_up_autotuning(encoded_config, overwrite_values):
        config = json.loads(base64.urlsafe_b64decode(encoded_config).decode("utf-8"))
        overwrite_values = overwrite_values if overwrite_values else {}
        for tuning_param in AUTOTUNING_ARGS:
            # TODO: This is for autotuning specifically, may cause surprises for someone with a weird setup
            if tuning_param in config:
                overwrite_values[tuning_param] = config[tuning_param]
        return overwrite_values

    @staticmethod
    def convert_key_value_to_command_line_arg(k, v):
        if isinstance(v, bool):
            if v:
                return [f"--{k}"]
            else:
                return []
        if v is None:
            return []
        return [f"--{k}", str(v)]

    def get_extra_deepspeed_args(self):
        """
        Sets up the extra arguments for deepspeed. This is done by reading in the `deepspeed_extra_args` dictionary from
            the configuration file, and then adding any arguments where values differ from those specified in the dataclass.
        """
        neox_args = self.get_parent_class_value_dict(
            *self.__class__.__bases__, only_non_defaults=True
        )

        extra_ds_args = dict()

        for key, value in self.deepspeed_extra_args.items():
            # Check to make sure the key is not already changed from defaults, and raise an exception if it is
            # This is to prevent users from accidentally writing arguments both in deepspeed_extra_args and in the base level
            # of the configuration file
            if hasattr(neox_args, key):
                raise ValueError(
                    f"Key {key} is already specified elsewhere. Reading in a different value from the 'deepspeed_extra_args' option in the configuration file will cause undefined behavior."
                )
            extra_ds_args[key] = value

        return extra_ds_args

    def get_deepspeed_main_args(self):
        args_list = list()

        if self.autotuning_run is not None:
            args_list.extend(
                self.convert_key_value_to_command_line_arg(
                    "autotuning", self.autotuning_run
                )
            )

        # get deepspeed runner args, and only pass them in to deepspeed launcher if they differ from defaults
        for key, default_value in NeoXArgsDeepspeedRunner().defaults():
            if key == "autotuning_run":
                continue
            configured_value = getattr(self, key)

            if key == "force_multi":
                if self.deepspeed_slurm or self.deepspeed_mpi:
                    configured_value = True
            if configured_value != default_value:
                args_list.extend(
                    self.convert_key_value_to_command_line_arg(key, configured_value)
                )

        if self.deepspeed_slurm:
            comment = getattr(self, "comment")
            if comment:
                args_list.extend(
                    self.convert_key_value_to_command_line_arg("comment", comment)
                )
            account = getattr(self, "account")
            if account:
                args_list.extend(
                    self.convert_key_value_to_command_line_arg("account", account)
                )

            # master_address = os.environ['SLURM_JOB_NODELIST'].split('\n')[0]
            # args_list.extend(
            #    self.convert_key_value_to_command_line_arg('master_addr', master_address)
            # )

        if "DLTS_HOSTFILE" in os.environ:
            args_list.extend(
                self.convert_key_value_to_command_line_arg(
                    "hostfile", os.environ["DLTS_HOSTFILE"]
                )
            )

        if "MASTER_ADDR" in os.environ:
            args_list.extend(
                self.convert_key_value_to_command_line_arg(
                    "master_addr", os.environ["MASTER_ADDR"]
                )
            )

        if (
            "--include" in args_list or "--exclude" in args_list
        ) and "--num_gpus" in args_list:
            print(
                "WARNING: both --include/--exclude and num_gpus were specified simultaneously - overriding num_gpus with --include/--exclude"
            )
            # cannot specify these both simultaneously, remove num_gpus from list
            idx = args_list.index("--num_gpus")
            # pop twice, once for the arg, once for its value
            args_list.pop(idx)
            args_list.pop(idx)

        # add user script
        args_list.append(self.user_script)

        self.configure_distributed_args()
        cwd = Path.cwd()

        # get deepspeed_config
        args_list.append("--deepspeed_config")

        if self.autotuning_run is not None:
            ds_fp = cwd / Path("ds_config.json")
            if self.rank == 0:
                with open(ds_fp, mode="w") as ds_file:
                    json.dump(self.deepspeed_config, ds_file)
            args_list.append(str(ds_fp))
        else:
            encoded_ds_config = base64.urlsafe_b64encode(
                json.dumps(self.deepspeed_config).encode("utf-8")
            ).decode("utf-8")
            args_list.append(encoded_ds_config)

        # get all config values
        args_list.append("--megatron_config")
        neox_args = self.get_parent_class_value_dict(
            *self.__class__.__bases__, only_non_defaults=True
        )
        encoded_mega_config = base64.urlsafe_b64encode(
            json.dumps(neox_args).encode("utf-8")
        ).decode("utf-8")
        args_list.append(str(encoded_mega_config))
        return args_list

    ############################################################################################################################
    # start of calculated properties

    @property
    def deepspeed_config(self) -> dict:
        """
        returns a dict containing variables within deepspeed config
        """
        config = self.get_parent_class_value_dict_extra_ds(
            NeoXArgsDeepspeedConfig, only_non_defaults=True
        )
        return config

    @property
    def deepspeed_runner(self) -> dict:
        """
        returns variables within deepspeed runner
        """
        return self.get_parent_class_value_dict(NeoXArgsDeepspeedRunner)

    @property
    def megatron_config(self) -> dict:
        """
        returns variables within megatron args
        """
        return self.get_parent_class_value_dict(*NEOX_ARG_CLASSES)

    @property
    def all_config(self) -> dict:
        """
        returns variables of all args
        """
        return self.get_parent_class_value_dict(*BASE_CLASSES)

    def get_parent_class_value_dict(
        self, *parent_classes, only_non_defaults=False
    ) -> dict:
        """
        takes a sequence of parent classes and returns corresponding values (with defaults set)
        """
        # TODO no Nones or non-defaults
        result = dict()
        for parent in parent_classes:
            for key, default_value in parent().defaults():
                if key in ["tokenizer", "tensorboard_writer", "adlr_autoresume_object"]:
                    continue
                if only_non_defaults:
                    value = getattr(self, key)
                    if value == default_value:
                        continue
                result[key] = getattr(self, key)
        return result

    def get_parent_class_value_dict_extra_ds(
        self, *parent_classes, only_non_defaults=False
    ) -> dict:
        """
        Takes a sequence of parent classes and returns corresponding values (with defaults set).
        Also adds in any extra deepspeed arguments that are specified in the configuration file.

        Args:
            parent_classes: sequence of parent classes
            only_non_defaults: if True, only returns values that differ from defaults

        Returns:
            dict of arguments and values

        """
        # TODO no Nones or non-defaults
        result = dict()
        for parent in parent_classes:
            for key, default_value in parent().defaults():
                if key in [
                    "tokenizer",
                    "tensorboard_writer",
                    "adlr_autoresume_object",
                    "deepspeed_extra_args",
                ]:
                    continue
                if only_non_defaults:
                    value = getattr(self, key)
                    if value == default_value:
                        continue
                result[key] = getattr(self, key)

        if self.deepspeed_extra_args is not None:
            extra_ds_args = self.get_extra_deepspeed_args()
            result.update(extra_ds_args)

        return result

    @property
    def params_dtype(self):
        """
        returns the datatype on the basis of configured precision
        """
        if self.precision == "fp16":
            return torch.half
        elif self.precision == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float

    ############################################################################################################################
    # start of logging and output

    def enable_logging(self):
        """
        enable Tee logs based on the configured logdir
        """
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            hostname = gethostname()
            file_prefix = os.path.join(self.log_dir, hostname)
            Tee(file_prefix + "_stdout.txt", err=False)
            Tee(file_prefix + "_stderr.txt", err=True)

    def print(self):
        """Print arguments."""
        if self.rank == 0 or self.rank is None:
            print("-------------------- arguments --------------------", flush=True)
            str_list = []
            for arg in vars(self):
                # add arg + value
                dots = "." * (32 - len(arg))
                value = getattr(self, arg)
                print_str = "  {} {} {}".format(arg, dots, value)

                # add info 'default or updated'
                field_def = self.__dataclass_fields__.get(arg)
                if field_def is not None:
                    default_info = (
                        "default" if value == field_def.default else "updated"
                    )
                else:
                    default_info = ""
                dots = "." * (64 - len(print_str))
                print_str += dots
                str_list.append({"print_str": print_str, "default_info": default_info})

            for arg in sorted(
                sorted(str_list, key=lambda x: x["print_str"].lower()),
                key=lambda x: x["default_info"],
                reverse=True,
            ):
                print(arg["print_str"] + arg["default_info"], flush=True)
            print("---------------- end of arguments ----------------", flush=True)

    ############################################################################################################################
    # start of calculations and derived values

    def configure_distributed_args(self):
        """
        Configures distributed training arguments from local variables set by deepspeed launcher.
        """
        if self.deepspeed_mpi:
            from deepspeed.comm import mpi_discovery

            mpi_discovery()

        if self.deepspeed_slurm:
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = (
                os.environ["SLURM_NTASKS"]
                if os.environ.get("SLURM_NTASKS") is not None
                else str(
                    int(os.environ["SLURM_NNODES"])
                    * int(os.environ["SLURM_NTASKS_PER_NODE"])
                )
            )

        self.update_value("local_rank", int(os.getenv("LOCAL_RANK", "0")))
        self.update_value("rank", int(os.getenv("RANK", "0")))
        self.update_value("world_size", int(os.getenv("WORLD_SIZE", "1")))

        if self.rank == 0:
            print(
                self.__class__.__name__
                + ".configure_distributed_args() using world size: {}, pipe-parallel size: {}, context-parallel size: {}, and model-parallel size: {} ".format(
                    self.world_size,
                    self.pipe_parallel_size,
                    self.context_parallel_size,
                    self.model_parallel_size,
                ),
                flush=True,
            )

    @staticmethod
    def calculate_batch_parameters(
        dp_world_size, train_batch=None, micro_batch=None, grad_acc=None
    ):
        # all values are provided nothing needs to be set
        if train_batch is not None and micro_batch is not None and grad_acc is not None:
            return train_batch, micro_batch, grad_acc

        # gradient_accumulation_steps needs to be set
        elif train_batch is not None and micro_batch is not None:
            grad_acc = train_batch // micro_batch
            grad_acc //= dp_world_size

        # micro_batch_per_gpu needs to be set
        elif train_batch is not None and grad_acc is not None:
            micro_batch = train_batch // dp_world_size
            micro_batch //= grad_acc

        # train_batch_size needs to be set
        elif micro_batch is not None and grad_acc is not None:
            train_batch = micro_batch * grad_acc
            train_batch *= dp_world_size

        # gradient_accumulation_steps and micro_batch_per_gpus is set
        elif train_batch is not None:
            grad_acc = 1
            micro_batch = train_batch // dp_world_size

        # train_batch_size and gradient_accumulation_step is set
        elif micro_batch is not None:
            train_batch = micro_batch * dp_world_size
            grad_acc = 1

        # either none of the three parameters are provided or just gradient_accumulation_step is provided
        else:
            assert (
                False
            ), "Either train_batch_size or train_micro_batch_size_per_gpu needs to be provided"
        return int(train_batch), int(micro_batch), int(grad_acc)

    @staticmethod
    def check_batch_parameters(dp_world_size, train_batch, micro_batch, grad_acc):
        assert (
            train_batch > 0
        ), f"Train batch size: {train_batch} has to be greater than 0"

        assert (
            micro_batch > 0
        ), f"Micro batch size per gpu: {micro_batch} has to be greater than 0"

        assert (
            grad_acc > 0
        ), f"Gradient accumulation steps: {grad_acc} has to be greater than 0"

        assert train_batch == micro_batch * grad_acc * dp_world_size, (
            f"Check batch related parameters. train_batch_size is not equal"
            " to micro_batch_per_gpu * gradient_acc_step * world_size \n"
            f"{train_batch} != {micro_batch} * {grad_acc} * {dp_world_size}"
        )

    def calculate_derived(self):
        """
        Derives additional configuration values necessary for training from the current config
        """

        # number of gpus
        # Get number of GPUs param or hostfile to determine train_batch_size
        global_num_gpus = getattr(self, "global_num_gpus", None)
        if global_num_gpus is None:
            if self.hostfile is not None or os.path.exists(DLTS_HOSTFILE):
                hostfile_path = self.hostfile or DLTS_HOSTFILE
                resources = obtain_resource_pool(
                    hostfile_path, self.include or "", self.exclude or ""
                )
                if self.num_nodes is not None and self.num_nodes > 0:
                    resources = {
                        k: resources[k]
                        for k in list(resources.keys())[: self.num_nodes]
                    }
                global_num_gpus = sum(map(len, resources.values()))
                if self.num_gpus is not None and self.num_gpus > 0:
                    global_num_gpus = self.num_gpus * len(resources)
            else:
                global_num_gpus = torch.cuda.device_count()
            self.update_value("global_num_gpus", global_num_gpus)

        logging.info(
            self.__class__.__name__
            + ".calculate_derived() "
            + f"Total number of GPUs determined to be: {global_num_gpus}"
        )

        # get world size in the model/pipe parallel case, the actual `world size` deepspeed uses is the size of the
        # data-parallel group, or (num_gpus / mp_size) / pp_size
        pp_size = self.pipe_parallel_size
        pp_size = pp_size if pp_size >= 1 else 1
        mp_size = self.model_parallel_size
        mp_size = mp_size if mp_size >= 1 else 1
        cp_size = self.context_parallel_size
        cp_size = cp_size if cp_size >= 1 else 1
        self.update_value("model_parallel_size", mp_size)
        self.update_value("context_parallel_size", cp_size)

        # pp_size, mp_size, and cp_size are only used here to compute dp world size and nowhere else.
        dp_world_size = (global_num_gpus / pp_size) / (mp_size * cp_size)
        if not (dp_world_size % 1 == 0):
            error_message = (
                "ERROR"
                + self.__class__.__name__
                + ".calculate_derived() "
                + f"(global_num_gpus / pp_size) / mp_size [({global_num_gpus} / {pp_size}) / {mp_size}] must be a whole number"
            )
            logging.error(error_message)
            raise AssertionError(error_message)

            # Automatically derive train_batch_size = train_micro_batch_size_per_gpu*global_num_gpus*gradient_accumulation_steps
        (
            train_batch_size,
            train_micro_batch_size_per_gpu,
            gradient_accumulation_steps,
        ) = self.calculate_batch_parameters(
            dp_world_size=dp_world_size,
            train_batch=self.train_batch_size,
            micro_batch=self.train_micro_batch_size_per_gpu,
            grad_acc=self.gradient_accumulation_steps,
        )
        self.check_batch_parameters(
            dp_world_size=dp_world_size,
            train_batch=train_batch_size,
            micro_batch=train_micro_batch_size_per_gpu,
            grad_acc=gradient_accumulation_steps,
        )
        self.update_values(
            {
                # batch size params
                "train_batch_size": train_batch_size,
                "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "batch_size": train_micro_batch_size_per_gpu,
                # duplicate items
                "clip_grad": self.gradient_clipping,
            }
        )

        # derive precision
        if self.fp16 and self.fp16.get("enabled", False):
            if self.precision is None:
                self.update_value("precision", "fp16")
            else:
                fp16_conflict = "DeepSpeed fp16 field was set but precision conflicts"
                assert self.precision == "fp16", fp16_conflict

        if self.bf16 and self.bf16.get("enabled", False):
            if self.precision is None:
                self.update_value("precision", "bfloat16")
            else:
                bf16_conflict = "DeepSpeed bf16 field was set but precision conflicts"
                assert self.precision == "bfloat16", bf16_conflict

        if self.precision == "fp16":
            if isinstance(self.fp16, dict) and len(self.fp16) > 0:
                fp16_args = copy.deepcopy(self.fp16)
                fp16_args["enabled"] = True
            else:
                fp16_args = {"type": "fp16", "enabled": True}
            self.update_value("fp16", fp16_args)
        elif self.precision == "bfloat16":
            if not self.bf16:
                bf_config = {"bf16": {"enabled": True}}
                # dt_config = {"grad_accum_dtype": "fp32"}
                if self.deepspeed_extra_args is None:
                    self.update_value("deepspeed_extra_args", bf_config)
                else:
                    extra_args = copy.deepcopy(self.deepspeed_extra_args)
                    extra_args.update(bf_config)
                    self.update_value("deepspeed_extra_args", extra_args)

            zero_stage = self.zero_optimization["stage"]
            if self.data_types is None:
                fp32_grad_accum = False
            else:
                fp32_grad_accum = self.data_types.get("grad_accum_dtype") == "fp32"
            if (zero_stage > 0) and (pp_size > 0) and not fp32_grad_accum:
                # Remove this code when this issue is resolved
                # https://github.com/microsoft/DeepSpeed/issues/1835
                logging.warn(
                    "Outstanding DeepSpeed issue means that pp>0, zero1, and bf16 will break without fp32 grads"
                )
        else:
            self.update_value("precision", "fp32")

        # zero optimization
        if self.zero_optimization is None:
            self.zero_optimization = copy.deepcopy(
                ZERO_DEFAULTS
            )  # a dict is overwritten and not updated key by key
        try:
            stage = self.zero_optimization["stage"]
            if stage in (0, 1, 2, 3):
                self.update_values(
                    {
                        "zero_stage": self.zero_optimization.get(
                            "stage", ZERO_DEFAULTS["stage"]
                        ),
                        "zero_reduce_scatter": self.zero_optimization.get(
                            "reduce_scatter", ZERO_DEFAULTS["reduce_scatter"]
                        ),
                        "zero_contiguous_gradients": self.zero_optimization.get(
                            "contiguous_gradients",
                            ZERO_DEFAULTS["contiguous_gradients"],
                        ),
                        "zero_reduce_bucket_size": self.zero_optimization.get(
                            "reduce_bucket_size", ZERO_DEFAULTS["reduce_bucket_size"]
                        ),
                        "zero_allgather_bucket_size": self.zero_optimization.get(
                            "allgather_bucket_size",
                            ZERO_DEFAULTS["allgather_bucket_size"],
                        ),
                    }
                )
            else:
                assert (
                    self.autotuning is not None
                ), f"Zero Stage must be an integer unless you are doing autotuning, not {stage}"
        except KeyError as ke:
            print(f"Zero Optimization config: {self.zero_optimization}")
            raise ke

        # optimizer and scheduler
        opt_params = self.optimizer or {
            "type": OPT_DEFAULT,
            "params": OPT_PARAMS_DEFAULTS,
        }
        self.update_values(
            {
                "optimizer_type": opt_params.get("type", OPT_DEFAULT),
                "lr": opt_params["params"].get("lr", OPT_PARAMS_DEFAULTS["lr"]),
            }
        )

        if self.optimizer_type.lower() == "onebitadam":
            assert (
                self.train_iters is not None
            ), "OneBitAdam requires train_iters to be specified"

            # onebitadam needs to instantiated by deepspeed, and so we need to pass deepspeed scheduler args
            # for all other optimizers, the scheduling is handled by megatron
            self.scheduler = {
                "type": "WarmupDecayLR",  # for now this is the only ds scheduler offering decay
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": int(self.train_iters * self.warmup),
                    "total_num_steps": self.lr_decay_iters or self.train_iters,
                },
            }

        # Fp16 loss scaling.
        self.update_value("dynamic_loss_scale", self.loss_scale is None)

        # Update 'is pipe parallel' flag
        # if we set pipe_parallel_size to 0, GPT2ModelPipe.to_sequential() is called, and we run training with
        # the sequential model without the PipelineModule wrapper to avoid the overhead it incurs
        self.update_value("is_pipe_parallel", self.pipe_parallel_size >= 1)
        # update 'is sequence parallel' flag
        self.update_value(
            "is_context_parallel",
            self.context_parallel_size > 1 and self.moe_num_experts == 1,
        )
        if self.moe_num_experts > 1:
            assert not (
                self.is_pipe_parallel or self.pipe_parallel_size > 1
            ), "MoE not supported with pipeline parallelism"
            assert self.zero_optimization["stage"] != 3, "MoE not compatible with zero3"

            assert (
                self.sequence_parallel is False
            ), "MoE not compatible with Sequence Parallel"

        # Attention config
        if self.attention_config is None:
            self.update_value("attention_config", [[["global"], self.num_layers]])
        self.update_value(
            "attention_config",
            expand_attention_types(self.attention_config, self.num_layers),
        )
        self.update_value(
            "requires_attention_mask",
            not all([item in ["ring", "flash"] for item in self.attention_config]),
        )
        assert all([item == "ring" for item in self.attention_config]) or (
            not self.is_context_parallel
        ), "Context parallel requires ring attention!"
        assert (
            len(self.attention_config) == self.num_layers
        ), "Length of attention config list must equal num_layers"
        for item in self.attention_config:
            assert (
                item in ATTENTION_TYPE_CHOICES
            ), f"Attention type {item} not recognized"
        if "gmlp" in self.attention_config or "amlp" in self.attention_config:
            assert (
                not self.partition_activations
            ), "GMLP Blocks are not compatible with partition activations"
        if "mamba" in self.attention_config:
            if isinstance(self.zero_stage, int):
                assert self.zero_stage <= 2, "Zero stage 3 not compatible with Mamba"
            assert (
                self.hidden_dropout == 0.0,
            ), "Mamba does not yet have dropout implemented"
        if "rwkv" in self.attention_config:
            assert (
                self.model_parallel_size == 1
            ), "RWKV not currently compatible with model parallelism"
            if isinstance(self.zero_stage, int):
                assert self.zero_stage <= 2, "Zero stage 3 not compatible with RWKV"
            assert (
                self.hidden_dropout == 0.0,
            ), "RWKV does not yet have dropout implemented"

        # Sparsity config
        if self.sparsity_config is None:
            # Can't have a default value as an empty dict so need to set it here
            self.update_value("sparsity_config", {})

        # Multi-query or grouped-query attention settings
        if self.num_kv_heads is not None:
            # need KV heads <= query heads, and KV heads dividing query heads evenly
            assert (
                self.num_attention_heads % self.num_kv_heads == 0
            ), "num_kv_heads must evenly divide num_attention_heads and be no greater than it"

            if self.num_kv_heads < self.num_attention_heads:
                # GQA / MQA not compatible with sparse attention configurations
                assert (
                    not self.sparsity_config
                ), "Sparse attention not compatible with GQA or MQA"
                assert all(
                    (attn_type == "flash")
                    or (attn_type == "global")
                    or (attn_type == "ring")
                    for attn_type in self.attention_config
                ), "GQA / MQA currently only compatible with Flash or standard global/sliding window Attention"
                assert (
                    self.num_kv_heads % self.model_parallel_size == 0
                ), "Number of KV heads must be at least model_parallel_size for now!"
        # Flash attention version >=2.3.0 required to combine Flash + Sliding Window Attention
        if "flash" in self.attention_config:
            _flash_version = packaging.version.Version(version("flash-attn"))
            if self.sliding_window_width is not None:
                assert _flash_version >= packaging.version.Version(
                    "2.3.0"
                ), f"Flash-Attention version ({str(_flash_version)}) must be >= 2.3.0 to support sliding window attention."
            if self.pos_emb == "alibi":
                if not _flash_version >= packaging.version.Version("2.4.0.post1"):
                    print(
                        f"Warning: Flash-Attention version ({str(_flash_version)}) must be >= 2.4.0.post1 to support AliBi. Falling back to flash-attn triton backend, but version 2.4.0.post1 or later will be required in future."
                    )

        # Adding equal dataset weights if none are provided
        if self.train_data_paths and (self.train_data_weights is None):
            self.train_data_weights = [1.0] * len(self.train_data_paths)
        elif self.pos_train_data_paths and (self.train_data_weights is None):
            self.train_data_weights = [1.0] * len(self.pos_train_data_paths)
        if self.valid_data_paths and (self.valid_data_weights is None):
            self.valid_data_weights = [1.0] * len(self.valid_data_paths)
        elif self.pos_valid_data_paths and (self.valid_data_weights is None):
            self.valid_data_weights = [1.0] * len(self.pos_valid_data_paths)
        if self.test_data_paths and (self.test_data_weights is None):
            self.test_data_weights = [1.0] * len(self.test_data_paths)
        elif self.pos_test_data_paths and (self.test_data_weights is None):
            self.test_data_weights = [1.0] * len(self.pos_test_data_paths)

        if self.train_label_data_paths:
            err_str = "Must use `train_label_data_paths` with `train_data_paths`, not `data_path`"
            assert self.train_data_paths and not self.data_path, err_str

        # if a sample input file is provided, default text_gen_type type to input-file
        if self.text_gen_type is None:
            if self.sample_input_file:
                self.update_value("text_gen_type", "input-file")
            else:
                self.update_value("text_gen_type", "unconditional")

    ############################################################################################################################
    # start of validation functions

    @classmethod
    def validate_keys(cls):
        """
        test that there are no duplicate arguments
        """
        source_classes = list(cls.__bases__)
        defined_properties = dict()

        for source_class in source_classes:
            source_vars = list(source_class.__dataclass_fields__)
            for item in source_vars:
                if item in defined_properties.keys():
                    logging.error(
                        f"({cls.__name__}) duplicate of item: {item}, in class {source_class.__name__} and {defined_properties[item]}"
                    )
                    return False
                else:
                    defined_properties[item] = source_class.__name__
        return True

    def validate_values(self):
        # the current codebase assumes running with deepspeed only
        if not self.deepspeed:
            return False

        # learning rate
        if self.lr is None:
            error_message = (
                f"{FAIL} " + self.__class__.__name__ + ".validate_values() lr is None"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # required arguments
        required_args = [
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "max_position_embeddings",
        ]
        for req_arg in required_args:
            if getattr(self, req_arg) is None:
                error_message = (
                    f"{FAIL}"
                    + self.__class__.__name__
                    + ".validate_values() "
                    + req_arg
                    + " is None."
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

        # Checks.
        if self.hidden_size % self.num_attention_heads != 0 and not (
            "mamba" in self.attention_config
        ):
            error_message = (
                f"{FAIL}"
                + self.__class__.__name__
                + ".validate_values() hidden_size must be divisible by num_attention_heads"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        if self.seq_length is not None:
            if not (self.max_position_embeddings >= self.seq_length):
                error_message = (
                    f"{FAIL}"
                    + self.__class__.__name__
                    + ".validate_values() max_position_embeddings must be bigger or equal seq_length"
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

        if not (self.min_lr <= self.lr):
            error_message = (
                "{FAIL}"
                + self.__class__.__name__
                + ".validate_values() min_lr must be smaller or equal lr"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        if (
            self.save is not None
            and self.checkpoint_factor is None
            and self.extra_save_iters is None
        ):
            error_message = (
                f"{FAIL}"
                + self.__class__.__name__
                + ".validate_values() checkpoint_factor or extra_save_iters must be defined if save is defined"
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # Parameters sharing does not work with torch DDP.
        if (self.num_unique_layers is not None) and (self.num_layers is not None):
            if not (self.num_unique_layers <= self.num_layers):
                error_message = (
                    f"{FAIL}"
                    + self.__class__.__name__
                    + ".validate_values() num-unique-layers must be smaller or equal num_layers"
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

            if not (self.num_layers % self.num_unique_layers == 0):
                error_message = (
                    f"{FAIL}"
                    + self.__class__.__name__
                    + ".validate_values() num-layers should be divisible by num-unique-layers"
                )
                logging.error(error_message)
                raise ValueError(error_message)
                return False

        if self.fp16_lm_cross_entropy and self.precision != "fp16":
            error_message = (
                f"{FAIL}"
                + self.__class__.__name__
                + ".validate_values() lm cross entropy in fp16 only support in fp16 mode."
            )
            logging.error(error_message)
            raise ValueError(error_message)
            return False

        # assert that if one of train/test/valid_data_path are provided, data_path should not be
        has_separate_path = [
            data_path is not None
            for data_path in [
                self.train_data_paths,
                self.valid_data_paths,
                self.test_data_paths,
            ]
        ]
        if all(has_separate_path):
            assert self.data_path is None, (
                f"{FAIL} Please provide *either* `data_path` or `train/valid/test_data_path` "
                "in args "
            )

        # assert that if one of train/test/valid_data_path are provided, all should be
        assert_error_mess = (
            f"{FAIL} One or more of train/valid/test data_path are not provided:\n\t"
        )
        assert_error_mess += "\n\t".join(
            [
                f"{name} data paths: {data_path},"
                for name, data_path in [
                    ["train", self.train_data_paths],
                    ["valid", self.valid_data_paths],
                    ["test", self.test_data_paths],
                ]
            ]
        )
        assert any(has_separate_path) == all(has_separate_path), assert_error_mess

        # assert that if train / valid / test data path(s) and weights are provided, that the paths and the weights should be equal length
        if self.train_data_paths is not None:
            assert len(self.train_data_paths) == len(self.train_data_weights)
        if self.valid_data_paths is not None:
            assert len(self.valid_data_paths) == len(self.valid_data_weights)
        if self.test_data_paths is not None:
            assert len(self.test_data_paths) == len(self.test_data_weights)

        return True

    def validate_types(self):
        """
        At runtime, checks types are actually the type specified.
        """
        for field_name, field_def in self.__dataclass_fields__.items():
            actual_value = getattr(self, field_name)
            if actual_value is None:
                continue  # we allow for some values not to be configured

            if self.autotuning is not None and actual_value == "auto":
                continue

            actual_type = type(actual_value)
            if actual_type != field_def.type:
                if (
                    actual_type == int and field_def.type == float
                ):  # floats should be able to be configured as ints
                    continue

                # for typing.Literal (i.e a list of choices) - checks that actual value is in accepted values
                elif field_def.type.__origin__ == Literal:
                    accepted_values = field_def.type.__args__
                    if actual_value in accepted_values:
                        continue
                    elif type(actual_value) == str:
                        # case insensitive checking
                        lowercase_accepted_values = [
                            i.lower() for i in accepted_values if isinstance(i, str)
                        ]
                        if actual_value.lower() in lowercase_accepted_values:
                            continue
                    logging.error(
                        f"{FAIL}"
                        + self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: '{actual_value}' Not in accepted values: '{accepted_values}'"
                    )
                    return False
                elif field_def.type.__origin__ == Union:
                    accepted_types = field_def.type.__args__
                    if actual_type in accepted_types:
                        continue
                    else:
                        logging.error(
                            f"{FAIL}"
                            + self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: '{actual_type}' not in {accepted_types}"
                        )
                        return False

                logging.error(
                    f"{FAIL}"
                    + self.__class__.__name__
                    + ".validate_types() "
                    + f"{field_name}: '{actual_type}' instead of '{field_def.type}'"
                )
                return False

        # validate deepspeed dicts
        for field_name in ["optimizer", "scheduler"]:
            value = getattr(self, field_name)
            if isinstance(
                value, dict
            ):  # dict is checked above, only fields are checked here
                if "type" in value:
                    if not isinstance(value["type"], str):
                        logging.error(
                            self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: key 'type' must be a string"
                        )
                        return False
                else:
                    logging.error(
                        f"{FAIL}"
                        + self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'type'"
                    )
                    return False
                if "params" in value:
                    if not isinstance(value["params"], dict):
                        logging.error(
                            f"{FAIL}"
                            + self.__class__.__name__
                            + ".validate_types() "
                            + f"{field_name}: key 'params' must be a dict"
                        )
                        return False
                else:
                    logging.error(
                        f"{FAIL}"
                        + self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'params'"
                    )
                    return False

        for field_name in ["fp16", "amp", "flops_profiler"]:
            value = getattr(self, field_name)
            if isinstance(value, dict):
                if not "enabled" in value:
                    error_message = (
                        f"{FAIL}"
                        + self.__class__.__name__
                        + ".validate_types() "
                        + f"{field_name}: must contain key 'enabled'"
                    )
                    logging.error(error_message)
                    return False

        return True
