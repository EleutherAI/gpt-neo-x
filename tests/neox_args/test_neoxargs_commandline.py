"""
verify parsing and handover of command line arguments
"""
import pytest
import sys
from unittest.mock import patch

from ..common import get_root_directory, get_config_directory, get_configs_with_path


@pytest.mark.cpu
def test_neoxargs_consume_deepy_args_with_config_dir():
    """
    verify consume_deepy_args processes command line arguments without config dir
    """

    from megatron.neox_arguments import NeoXArgs

    # load neox args with command line
    with patch(
        "sys.argv",
        [str(get_root_directory() / "deepy.py"), "train.py"]
        + get_configs_with_path(["small.yml", "local_setup.yml"]),
    ):
        args_loaded_consume = NeoXArgs.consume_deepy_args()

    # load neox args directly from yaml files
    args_loaded_yamls = NeoXArgs.from_ymls(
        get_configs_with_path(["small.yml", "local_setup.yml"])
    )

    # update values from yaml files that cannot otherwise be matched
    args_loaded_yamls.update_value("user_script", "train.py")
    args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

    assert args_loaded_yamls == args_loaded_consume


@pytest.mark.cpu
def test_neoxargs_consume_deepy_args_without_yml_suffix():
    """
    verify consume_deepy_args processes command line arguments without yaml suffix
    """

    from megatron.neox_arguments import NeoXArgs

    # load neox args with command line
    with patch(
        "sys.argv",
        [str(get_root_directory() / "deepy.py"), "train.py"]
        + get_configs_with_path(["small", "local_setup"]),
    ):
        args_loaded_consume = NeoXArgs.consume_deepy_args()

    # load neox args directly from yaml files
    args_loaded_yamls = NeoXArgs.from_ymls(
        get_configs_with_path(["small.yml", "local_setup.yml"])
    )

    # update values from yaml files that cannot otherwise be matched
    args_loaded_yamls.update_value("user_script", "train.py")
    args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

    assert args_loaded_yamls == args_loaded_consume

@pytest.mark.cpu
def test_neoxargs_consume_deepy_args_with_hostfile_param():
    """
    Verify consume_deepy_args processes command line arguments without yaml suffix.
    Also test the hostfile CLI arg
    """

    from megatron.neox_arguments import NeoXArgs

    # load neox args with command line
    with patch(
        "sys.argv",
        [str(get_root_directory() / "deepy.py"), "train.py"]
        + get_configs_with_path(["small", "local_setup"])
        + ["--hostfile=/mock_path"]
    ):
        args_loaded_consume = NeoXArgs.consume_deepy_args()

    # load neox args directly from yaml files
    args_loaded_yamls = NeoXArgs.from_ymls(
        get_configs_with_path(["small.yml", "local_setup.yml"])
    )

    # update values from yaml files that cannot otherwise be matched
    args_loaded_yamls.update_value("user_script", "train.py")
    args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

    assert args_loaded_yamls == args_loaded_consume

@pytest.mark.cpu
def test_neoxargs_consume_deepy_args_with_config_dir():
    """
    verify consume_deepy_args processes command line arguments including config dir
    """

    from megatron.neox_arguments import NeoXArgs

    # load neox args with command line
    with patch(
        "sys.argv",
        [
            str(get_root_directory() / "deepy.py"),
            "train.py",
            "-d",
            str(get_config_directory()),
        ]
        + ["small.yml", "local_setup.yml"],
    ):
        args_loaded_consume = NeoXArgs.consume_deepy_args()

    # load neox args directly from yaml files
    args_loaded_yamls = NeoXArgs.from_ymls(
        get_configs_with_path(["small.yml", "local_setup.yml"])
    )

    # update values from yaml files that cannot otherwise be matched
    args_loaded_yamls.update_value("user_script", "train.py")
    args_loaded_yamls.wandb_group = args_loaded_consume.wandb_group

    assert args_loaded_yamls == args_loaded_consume


@pytest.mark.cpu
def test_neoxargs_consume_neox_args():
    """
    verify megatron args are correctly consumed after sending via deepspeed
    """
    from megatron.neox_arguments import NeoXArgs

    # intitially load config from files as would be the case in deepy.py
    yaml_list = get_configs_with_path(["small.yml", "local_setup.yml"])
    args_baseline = NeoXArgs.from_ymls(yaml_list)
    args_baseline.update_value("user_script", str(get_root_directory() / "train.py"))
    deepspeed_main_args = args_baseline.get_deepspeed_main_args()

    # patch sys.argv so that args can be access by set_global_variables within initialize_megatron
    with patch("sys.argv", deepspeed_main_args):
        args_loaded = NeoXArgs.consume_neox_args()

    # TODO is the wandb group really to be changed?
    args_loaded.wandb_group = args_baseline.wandb_group
    assert args_baseline.megatron_config == args_loaded.megatron_config
