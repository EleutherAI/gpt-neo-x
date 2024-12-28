# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest

import eval
import generate
import train
from megatron.neox_arguments import NeoXArgs
from tests.common import save_random_model, simulate_deepy_env
from tools.datasets import preprocess_data


@pytest.fixture(
    params=[
        "HFGPT2Tokenizer",
        "HFTokenizer",
        "GPT2BPETokenizer",
        "CharLevelTokenizer",
        "TiktokenTokenizer",
        "SPMTokenizer",
    ]
)
def tokenizer_type(request):
    return request.param


@pytest.fixture(params=[None, "tests/data/sample_prompt.txt"])
def sample_input_file(request):
    return request.param


@pytest.mark.cpu
def test_preprocess_data(tokenizer_type):
    if tokenizer_type == "SPMTokenizer":
        pytest.xfail(
            reason="Expected easy resolution: Need to provide a valid model file from somewhere"
        )
    vocab_file = {
        "HFTokenizer": "tests/data/hf_cache/tokenizer/gpt2.json",
        "TiktokenTokenizer": "cl100k_base",
        "HFGPT2Tokenizer": "gpt2",
    }
    input_args = [
        "--input",
        "./tests/data/enwik8_first100.txt",
        "--output-prefix",
        "./tests/data/enwik8_first100",
        "--vocab",
        vocab_file.get(tokenizer_type, "./data/gpt2-vocab.json"),
        "--tokenizer-type",
        tokenizer_type,
        "--merge-file",
        "./data/gpt2-merges.txt",
        "--append-eod",
    ]
    preprocess_data.main(input_args)


def test_generate(monkeypatch, tmpdir, tmp_path, sample_input_file):
    model_dir = str(tmpdir)
    sample_output_file = str(tmp_path) + ".txt"
    input_args = ["generate.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    save_random_model(deepspeed_main_args, model_dir)

    # Generate output
    generate_args = {
        "load": model_dir,
        "sample_input_file": sample_input_file,
        "sample_output_file": sample_output_file,
    }
    generate.main(input_args=deepspeed_main_args, overwrite_values=generate_args)


def test_evaluate(monkeypatch, tmpdir, tmp_path):
    model_dir = str(tmpdir)
    sample_output_file = str(tmp_path)
    input_args = ["generate.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    save_random_model(deepspeed_main_args, model_dir)

    # Generate output
    evaluate_args = {
        "load": model_dir,
        "eval_tasks": ["lambada"],  # ["lambada", "hellaswag", "piqa", "sciq"],
        "eval_results_prefix": sample_output_file,
    }
    eval.main(input_args=deepspeed_main_args, overwrite_values=evaluate_args)


#@pytest.mark.skip(
#    reason="All model tests are skipped until we fix the CUDA + torch multiprocessing issue."
#)
def test_finetuning(monkeypatch, tmpdir, tmp_path):
    # Save random model, load random model, keep training
    # TODO: add mocking to check that we're not ignoring the previously loaded model
    model_dir = str(tmpdir)
    sample_output_file = str(tmp_path)
    input_args = ["generate.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    save_random_model(deepspeed_main_args, model_dir)

    # Generate output
    finetune_args = {"load": model_dir, "finetune": True}
    train.main(input_args=deepspeed_main_args, overwrite_values=finetune_args)


#@pytest.mark.skip(
#    reason="All model tests are skipped until we fix the CUDA + torch multiprocessing issue."
#)
def test_train_launcher(monkeypatch):
    input_args = ["train.py", "tests/config/test_setup.yml"]
    deepspeed_main_args = simulate_deepy_env(monkeypatch, input_args)
    train.main(input_args=deepspeed_main_args)
