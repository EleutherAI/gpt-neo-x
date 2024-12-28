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

import os
import time
import shutil
import itertools
import inspect
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from deepspeed.accelerator import get_accelerator

import pytest
from _pytest.outcomes import Skipped
from _pytest.fixtures import FixtureLookupError, FixtureFunctionMarker
import random
import train

import torch

import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from copy import deepcopy
import deepspeed

TEST_CHECKPOINT_DIR = "test_checkpoint"
TEST_LOG_DIR = "test_logs"
TEST_TENSORBOARD_DIR = "test_tensorboard"

# Worker timeout *after* the first worker has completed.
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120
DEEPSPEED_TEST_TIMEOUT = 600

def is_rocm_pytorch():
    """
    Check if the current PyTorch installation is using ROCm.
    
    Returns:
        bool: True if PyTorch is using ROCm, False otherwise.
    """
    return hasattr(torch.version, 'hip') and torch.version.hip is not None

def get_xdist_worker_id():
    xdist_worker = os.environ.get("PYTEST_XDIST_WORKER", None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace("gw", "")
        return int(xdist_worker_id)
    return None


def get_master_port():
    master_port = os.environ.get("DS_TEST_PORT", "29503")
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is not None:
        master_port = str(int(master_port) + xdist_worker_id)
    return master_port


_num_gpus = None

def set_accelerator_visible():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it using accelerator specific command instead
        if get_accelerator().device_name() == "cuda":
            if is_rocm_pytorch():
                rocm_smi = subprocess.check_output(["rocm-smi", "--showid"])
                gpu_ids = filter(
                    lambda s: "GPU" in s, rocm_smi.decode("utf-8").strip().split("\n")
                )
                num_accelerators = len(list(gpu_ids))
            else:
                nvidia_smi = subprocess.check_output(["nvidia-smi", "--list-gpus"])
                num_accelerators = len(nvidia_smi.decode("utf-8").strip().split("\n"))
        elif get_accelerator().device_name() == "xpu":
            clinfo = subprocess.check_output(["clinfo"])
            lines = clinfo.decode("utf-8").strip().split("\n")
            num_accelerators = 0
            for line in lines:
                match = re.search("Device Type.*GPU", line)
                if match:
                    num_accelerators += 1
        elif get_accelerator().device_name() == "npu":
            npu_smi = subprocess.check_output(["npu-smi", "info", "-l"])
            num_accelerators = int(
                npu_smi.decode("utf-8").strip().split("\n")[0].split(":")[1].strip()
            )
        else:
            assert get_accelerator().device_name() == "cpu"
            cpu_sockets = int(
                subprocess.check_output(
                    'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l',
                    shell=True,
                )
            )
            num_accelerators = cpu_sockets

        cuda_visible = ",".join(map(str, range(num_accelerators)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)


def count_gpus():
    global _num_gpus
    if _num_gpus is None:
        nvidia_smi = subprocess.check_output(["nvidia-smi", "--list-gpus"])
        _num_gpus = len(nvidia_smi.decode("utf-8").strip().split("\n"))
    return _num_gpus


def set_cuda_visibile():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it from nvidia-smi instead
        nvidia_smi = subprocess.check_output(["nvidia-smi", "--list-gpus"])
        num_gpus = len(nvidia_smi.decode("utf-8").strip().split("\n"))
        cuda_visible = ",".join(map(str, range(num_gpus)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)


def get_root_directory():
    return Path(__file__).parents[1]


def get_config_directory():
    return get_root_directory() / "configs"


def get_configs_with_path(configs):
    return [str(get_config_directory() / cfg) for cfg in configs]


def clear_test_dirs():
    log_dir = os.path.join(get_root_directory(), TEST_LOG_DIR)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    checkpoint_dir = os.path.join(get_root_directory(), TEST_CHECKPOINT_DIR)
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    tensorboard_dir = os.path.join(get_root_directory(), TEST_TENSORBOARD_DIR)
    if os.path.isdir(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)


class DistributedExec(ABC):
    """
    Base class for distributed execution of functions/methods. Contains common
    methods needed for DistributedTest and DistributedFixture.
    """

    world_size = 2
    backend = get_accelerator().communication_backend_name()
    init_distributed = True
    set_dist_env = True
    requires_cuda_env = True
    reuse_dist_env = False
    _pool_cache = {}
    exec_timeout = DEEPSPEED_TEST_TIMEOUT

    @abstractmethod
    def run(self):
        ...

    def __call__(self, request=None):
        self._fixture_kwargs = self._get_fixture_kwargs(request, self.run)
        world_size = self.world_size
        if self.requires_cuda_env and not get_accelerator().is_available():
            pytest.skip("only supported in accelerator environments.")

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)

    def _get_fixture_kwargs(self, request, func):
        if not request:
            return {}
        # Grab fixture / parametrize kwargs from pytest request object
        fixture_kwargs = {}
        params = inspect.getfullargspec(func).args
        params.remove("self")
        for p in params:
            try:
                fixture_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return fixture_kwargs

    def _launch_procs(self, num_procs):
        # Verify we have enough accelerator devices to run this test
        if (
            get_accelerator().is_available()
            and get_accelerator().device_count() < num_procs
        ):
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {num_procs} required, {get_accelerator().device_count()} available"
            )

        mp.set_start_method("spawn", force=True)

        # Create process pool or use cached one
        master_port = None
        if self.reuse_dist_env:
            if num_procs not in self._pool_cache:
                self._pool_cache[num_procs] = mp.Pool(processes=num_procs)
                master_port = get_master_port()
            pool = self._pool_cache[num_procs]
        else:
            pool = mp.Pool(processes=num_procs)
            master_port = get_master_port()

        # Run the test
        args = [(local_rank, num_procs, master_port) for local_rank in range(num_procs)]
        skip_msgs_async = pool.starmap_async(self._dist_run, args)

        try:
            skip_msgs = skip_msgs_async.get(self.exec_timeout)
        except mp.TimeoutError:
            # Shortcut to exit pytest in the case of a hanged test. This
            # usually means an environment error and the rest of tests will
            # hang (causing super long unit test runtimes)
            pytest.exit("Test hanged, exiting", returncode=0)

        # Tear down distributed environment and close process pools
        self._close_pool(pool, num_procs)

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])

    def _dist_run(self, local_rank, num_procs, master_port):
        skip_msg = ""
        if not dist.is_initialized():
            """Initialize deepspeed.comm and execute the user function."""
            if self.set_dist_env:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(master_port)
                os.environ["LOCAL_RANK"] = str(local_rank)
                # NOTE: unit tests don't support multi-node so local_rank == global rank
                os.environ["RANK"] = str(local_rank)
                # In case of multiprocess launching LOCAL_SIZE should be same as WORLD_SIZE
                # DeepSpeed single node launcher would also set LOCAL_SIZE accordingly
                os.environ["LOCAL_SIZE"] = str(num_procs)
                os.environ["WORLD_SIZE"] = str(num_procs)

            # turn off NCCL logging if set
            os.environ.pop("NCCL_DEBUG", None)

            if get_accelerator().is_available():
                set_accelerator_visible()

            if get_accelerator().is_available():
                get_accelerator().set_device(local_rank)

            if self.init_distributed:
                deepspeed.init_distributed(dist_backend=self.backend)
                dist.barrier()

        try:
            self.run(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_msg = e.msg
            else:
                raise e

        return skip_msg

    def _dist_destroy(self):
        if (dist is not None) and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _close_pool(self, pool, num_procs, force=False):
        if force or not self.reuse_dist_env:
            msg = pool.starmap(self._dist_destroy, [() for _ in range(num_procs)])
            pool.close()
            pool.join()


class DistributedFixture(DistributedExec):
    """
    Implementation that extends @pytest.fixture to allow for distributed execution.
    This is primarily meant to be used when a test requires executing two pieces of
    code with different world sizes.

    There are 2 parameters that can be modified:
        - world_size: int = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside fixture
        - can be reused by multiple tests
        - can accept other fixtures as input

    Limitations:
        - cannot use @pytest.mark.parametrize
        - world_size cannot be modified after definition and only one world_size value is accepted
        - any fixtures used must also be used in the test that uses this fixture (see example below)
        - return values cannot be returned. Passing values to a DistributedTest
          object can be achieved using class_tmpdir and writing to file (see example below)

    Usage:
        - must implement a run(self, ...) method
        - fixture can be used by making the class name input to a test function

    Example:
        @pytest.fixture(params=[10,20])
        def regular_pytest_fixture(request):
            return request.param

        class distributed_fixture_example(DistributedFixture):
            world_size = 4

            def run(self, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                local_rank = os.environ["LOCAL_RANK"]
                print(f"Rank {local_rank} with value {regular_pytest_fixture}")
                with open(os.path.join(class_tmpdir, f"{local_rank}.txt"), "w") as f:
                    f.write(f"{local_rank},{regular_pytest_fixture}")

        class TestExample(DistributedTest):
            world_size = 1

            def test(self, distributed_fixture_example, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                for rank in range(4):
                    with open(os.path.join(class_tmpdir, f"{rank}.txt"), "r") as f:
                        assert f.read() == f"{rank},{regular_pytest_fixture}"
    """

    is_dist_fixture = True

    # These values are just placeholders so that pytest recognizes this as a fixture
    _pytestfixturefunction = FixtureFunctionMarker(scope="function", params=None)
    __name__ = ""

    def __init__(self):
        assert isinstance(
            self.world_size, int
        ), "Only one world size is allowed for distributed fixtures"
        self.__name__ = type(self).__name__
        _pytestfixturefunction = FixtureFunctionMarker(
            scope="function", params=None, name=self.__name__
        )


class DistributedTest(DistributedExec):
    """
    Implementation for running pytest with distributed execution.

    There are 2 parameters that can be modified:
        - world_size: Union[int,List[int]] = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside tests
        - works with pytest fixtures, parametrize, mark, etc.
        - can contain multiple tests (each of which can be parametrized separately)
        - class methods can be fixtures (usable by tests in this class only)
        - world_size can be changed for individual tests using @pytest.mark.world_size(world_size)
        - class_tmpdir is a fixture that can be used to get a tmpdir shared among
          all tests (including DistributedFixture)

    Usage:
        - class name must start with "Test"
        - must implement one or more test*(self, ...) methods

    Example:
        @pytest.fixture(params=[10,20])
        def val1(request):
            return request.param

        @pytest.mark.fast
        @pytest.mark.parametrize("val2", [30,40])
        class TestExample(DistributedTest):
            world_size = 2

            @pytest.fixture(params=[50,60])
            def val3(self, request):
                return request.param

            def test_1(self, val1, val2, str1="hello world"):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                assert all(val1, val2, str1)

            @pytest.mark.world_size(1)
            @pytest.mark.parametrize("val4", [70,80])
            def test_2(self, val1, val2, val3, val4):
                assert int(os.environ["WORLD_SIZE"]) == 1
                assert all(val1, val2, val3, val4)
    """
    is_dist_test = True

    # Temporary directory that is shared among test methods in a class
    @pytest.fixture(autouse=True, scope="class")
    def class_tmpdir(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp(self.__class__.__name__)
        return fn

    def run(self, **fixture_kwargs):
        self._current_test(**fixture_kwargs)

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_cuda_env and not get_accelerator().is_available():
            pytest.skip("only supported in accelerator environments.")

        # Catch world_size override pytest mark
        for mark in getattr(request.function, "pytestmark", []):
            if mark.name == "world_size":
                world_size = mark.args[0]
                break
        else:
            world_size = self.world_size

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)
            time.sleep(0.5)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)


def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))


def model_setup(yaml_list=None, param_dict=None, clear_data=True):
    from megatron.neox_arguments import NeoXArgs
    from megatron.mpu import destroy_model_parallel
    from megatron.initialize import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    destroy_model_parallel()  # mpu model parallel contains remaining global vars
    if clear_data and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
        or torch.distributed.get_rank() == 0
    ):
        clear_test_dirs()

    overwrite_values = {
        "user_script": str(get_root_directory() / "train.py"),
        "save": TEST_CHECKPOINT_DIR,
        "load": TEST_CHECKPOINT_DIR,
        "log_dir": TEST_LOG_DIR,
        "tensorboard_dir": TEST_TENSORBOARD_DIR,
    }

    # should not both be none
    assert yaml_list is not None or param_dict is not None

    # initially load config from files as would be the case in deepy.py
    if yaml_list is not None:
        args_loaded = NeoXArgs.from_ymls(yaml_list, overwrite_values=overwrite_values)
    else:
        p_dict = param_dict.copy()
        p_dict.update(overwrite_values)
        args_loaded = NeoXArgs.from_dict(p_dict)

    args_loaded.build_tokenizer()

    initialize_megatron(neox_args=args_loaded)
    model, optimizer, lr_scheduler, reference_model = setup_model_and_optimizer(
        neox_args=args_loaded, use_cache=True
    )
    return model, optimizer, lr_scheduler, reference_model, args_loaded


def simulate_deepy_env(monkeypatch, input_args):
    from megatron.neox_arguments import NeoXArgs

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    return deepspeed_main_args


def save_random_model(input_args, model_dir, train_iters=0):
    # Save randomly initialised model
    train_args = {
        "do_train": False,
        "train_iters": train_iters,
        "save": model_dir,
        "extra_save_iters": [train_iters],
    }
    train.main(input_args=input_args, overwrite_values=train_args)


def bounded_product(sequence, n=None, seed=None):
    """
    Returns a shuffled, bounded cartesian product of the input sequence.
    Designed to cover as wide a range of permutations as possible with a limited number of iterations.
    Will manifest the whole list in memory, so not suitable for super large sequences.

    :param sequence: iterable
    :param n: length of returned list
    :param seed: random seed for reproducibility
    :return: list
    """
    p = list(itertools.product(*sequence))
    if seed is not None:
        random.seed(seed)
    random.shuffle(p)
    return p if n is None else p[:n]


def model_setup_simple(deepspeed_main_args, overwrite_values, iteration=None):
    from megatron.neox_arguments import NeoXArgs
    from megatron import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    neox_args = NeoXArgs.consume_neox_args(
        input_args=deepspeed_main_args, overwrite_values=overwrite_values
    )
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()
    initialize_megatron(neox_args=neox_args)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        neox_args=neox_args, use_cache=False
    )
    return model, optimizer, lr_scheduler, neox_args


def parametrize(
    params_to_test: dict, max_tests: int = 50, seed: int = None, with_names=True
):
    """
    Generates a random sample of max_tests length of all possible combinations of values in
    `params_to_test`.

    In `params_to_test` you can either specify one value, and all possible settings of that value,
    or two values separated by a comma, and all possible combinations of those two values in tandem.
        i.e "hidden_size,num_heads": [[768,12], [1024,32], [2048, 64]]
    so the first item in each list is a value of `hidden_size` and the second a value of `num_heads`
    this is useful for reducing the size of possible tests for values we know are unlikely to interact beforehand,
    since the cartesian product can grow very large.

    :param params_to_test: dict of neox params
    :param max_tests: maximum number of tests to run
    :param seed: random seed
    :return: a list of neox param dicts to pass to a parametrized unit test
    """
    keys, values = zip(*params_to_test.items())
    ret = []
    if with_names:
        experiments = []
    for p in bounded_product(values, n=max_tests, seed=seed):
        experiment = dict(zip(keys, p))
        to_pop = []
        to_add = {}
        for k, v in experiment.items():
            if "," in k:
                keys_split = [i.strip() for i in k.split(",")]
                values_separated = experiment[k]
                to_pop.append(k)
                assert len(values_separated) == len(keys_split)
                new_dict = dict(zip(keys_split, values_separated))
                to_add.update(new_dict)
        experiment.update(to_add)
        for k in to_pop:
            experiment.pop(k)
        base = deepcopy(BASE_CONFIG)
        base.update(experiment)
        ret.append(base)
        if with_names:
            experiments.append(experiment)
    if with_names:
        return ret, [dict_repr(d) for d in experiments]
    return ret


def dict_repr(d):
    return " ".join([f"{str(k)} : {str(v)}" for k, v in d.items()])


binary = [True, False]

with open("tests/config/test_setup.yml", "r") as f:
    BASE_CONFIG = load(f, Loader=Loader)
    print(f"Base Config:\n{BASE_CONFIG}")
