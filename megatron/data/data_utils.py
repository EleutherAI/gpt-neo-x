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

import math
import torch
import numpy as np
from typing import List, Tuple
from itertools import zip_longest, cycle
from functools import partial

from megatron import mpu, print_rank_0
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt2_dataset import GPT2Dataset
from megatron.data.pairwise_dataset import PairwiseDataset
from megatron.data.online_dataset import OnlineDataset
from megatron.data.samplers import DistributedBatchSampler


def make_data_loader(dataset, neox_args):
    """Build dataloader given an input dataset."""
    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )
    # Torch dataloader.
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
    )


def build_the_dataset(
    data_prefix,
    pos_data_prefix,
    neg_data_prefix,
    name,
    data_impl,
    pack_impl,
    dataset_impl,
    allow_chopped,
    num_samples,
    num_epochs,
    seq_length,
    seed,
    skip_warmup,
    build_index_mappings=True,
    label_prefix=None,
    pos_label_prefix=None,
    neg_label_prefix=None,
    precompute_model_name=None,
    reward_prefix=None,
):
    """Build train/valid/test datasets."""
    if dataset_impl == "gpt2":
        indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
        if label_prefix is None:
            label_dataset = None
        else:
            label_dataset = make_indexed_dataset(label_prefix, data_impl, skip_warmup)
        if precompute_model_name is not None:
            # If we have the name, assume it exists. If it doesn't, it will just be None which is fine.
            precompute_indexed_dataset = make_indexed_dataset(
                data_prefix + "_" + precompute_model_name, data_impl, skip_warmup
            )
            precompute_indexed_dataset = precompute_indexed_dataset
        else:
            precompute_indexed_dataset = None
        if reward_prefix is not None:
            reward_dataset = make_indexed_dataset(reward_prefix, data_impl, skip_warmup)
        else:
            reward_dataset = None
    elif dataset_impl == "pairwise":
        pos_indexed_dataset = make_indexed_dataset(
            pos_data_prefix, data_impl, skip_warmup
        )
        neg_indexed_dataset = make_indexed_dataset(
            neg_data_prefix, data_impl, skip_warmup
        )
        if pos_label_prefix is None:
            pos_label_dataset = None
            # Also do neg here since they both must be the same
            assert neg_label_prefix is None
            neg_label_dataset = None
        else:
            pos_label_dataset = make_indexed_dataset(
                pos_label_prefix, data_impl, skip_warmup
            )
            # Also do neg here since they both must be the same
            assert neg_label_prefix is not None
            neg_label_dataset = make_indexed_dataset(
                neg_label_prefix, data_impl, skip_warmup
            )
        if precompute_model_name is None:
            pos_ref_dataset = None
            neg_ref_dataset = None
        else:
            pos_ref_dataset = make_indexed_dataset(
                pos_data_prefix + "_" + precompute_model_name, data_impl, skip_warmup
            )
            neg_ref_dataset = make_indexed_dataset(
                neg_data_prefix + "_" + precompute_model_name, data_impl, skip_warmup
            )
    else:
        raise NotImplementedError(f"dataset_impl={dataset_impl} not implemented")

    total_num_of_documents = (
        indexed_dataset.sizes.shape[0]
        if dataset_impl == "gpt2"
        else pos_indexed_dataset.sizes.shape[0]
    )
    print_rank_0("    {}:".format(name))
    print_rank_0("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    if dataset_impl == "gpt2":
        dataset = GPT2Dataset(
            name,
            data_prefix,
            documents,
            indexed_dataset,
            num_samples,
            num_epochs,
            seq_length,
            seed,
            pack_impl=pack_impl,
            allow_chopped=allow_chopped,
            build_index_mappings=build_index_mappings,
            label_dataset=label_dataset,
            reward_dataset=reward_dataset,
            ref_dataset=precompute_indexed_dataset,
        )
    elif dataset_impl == "pairwise":
        dataset = PairwiseDataset(
            name,
            pos_data_prefix,
            documents,
            pos_indexed_dataset,
            neg_indexed_dataset,
            num_samples,
            seq_length,
            seed,
            pack_impl=pack_impl,
            allow_chopped=allow_chopped,
            build_index_mappings=build_index_mappings,
            pos_label_dataset=pos_label_dataset,
            neg_label_dataset=neg_label_dataset,
            pos_ref_dataset=pos_ref_dataset,
            neg_ref_dataset=neg_ref_dataset,
        )
    return dataset


def build_train_valid_test_datasets(
    data_prefix,
    use_shared_fs,
    data_impl,
    pack_impl,
    allow_chopped,
    splits_string,
    train_valid_test_num_samples,
    train_valid_test_epochs,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
            )
            dataset = GPT2Dataset(
                name,
                data_prefix,
                documents,
                indexed_dataset,
                train_valid_test_num_samples[index],
                train_valid_test_epochs[index],
                seq_length,
                seed,
                pack_impl=pack_impl,
                allow_chopped=allow_chopped,
                use_shared_fs=use_shared_fs,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return train_dataset, valid_dataset, test_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    if num_samples is not None:
        # Add 0.5% (the 1.005 factor) so in case the blending dataset does
        # not uniformly distribute the number of samples, we still have
        # samples left to feed to the network.
        weighted_num_samples = []
        for weight in weights:
            weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    else:
        weighted_num_samples = [None for _ in weights]
    return weights, weighted_num_samples


def build_weighted_datasets(
    neox_args,
    train_num_samples,
    valid_num_samples,
    test_num_samples,
    train_epochs,
    valid_epochs,
    test_epochs,
    build_index_mappings=True,
):
    # build individual datasets
    train_datasets, valid_datasets, test_datasets = [], [], []
    for i, (
        train_path,
        train_label_path,
        train_reward_path,
        valid_path,
        valid_label_path,
        valid_reward_path,
        test_path,
        test_label_path,
        test_reward_path,
        pos_train_path,
        neg_train_path,
        pos_train_label_path,
        neg_train_label_path,
        pos_valid_path,
        neg_valid_path,
        pos_valid_label_path,
        neg_valid_label_path,
        pos_test_path,
        neg_test_path,
        pos_test_label_path,
        neg_test_label_path,
    ) in enumerate(
        zip_longest(
            neox_args.train_data_paths if neox_args.train_data_paths else [],
            neox_args.train_label_data_paths
            if neox_args.train_label_data_paths
            else [],
            neox_args.train_reward_data_paths
            if neox_args.train_reward_data_paths
            else [],
            neox_args.valid_data_paths if neox_args.valid_data_paths else [],
            neox_args.valid_label_data_paths
            if neox_args.valid_label_data_paths
            else [],
            neox_args.valid_reward_data_paths
            if neox_args.valid_reward_data_paths
            else [],
            neox_args.test_data_paths if neox_args.test_data_paths else [],
            neox_args.test_label_data_paths if neox_args.test_label_data_paths else [],
            neox_args.test_reward_data_paths
            if neox_args.test_reward_data_paths
            else [],
            neox_args.pos_train_data_paths if neox_args.pos_train_data_paths else [],
            neox_args.neg_train_data_paths if neox_args.neg_train_data_paths else [],
            neox_args.pos_train_label_data_paths
            if neox_args.pos_train_label_data_paths
            else [],
            neox_args.neg_train_label_data_paths
            if neox_args.neg_train_label_data_paths
            else [],
            neox_args.pos_valid_data_paths if neox_args.pos_valid_data_paths else [],
            neox_args.neg_valid_data_paths if neox_args.neg_valid_data_paths else [],
            neox_args.pos_valid_label_data_paths
            if neox_args.pos_valid_label_data_paths
            else [],
            neox_args.neg_valid_label_data_paths
            if neox_args.neg_valid_label_data_paths
            else [],
            neox_args.pos_test_data_paths if neox_args.pos_test_data_paths else [],
            neox_args.neg_test_data_paths if neox_args.neg_test_data_paths else [],
            neox_args.pos_test_label_data_paths
            if neox_args.pos_test_label_data_paths
            else [],
            neox_args.neg_test_label_data_paths
            if neox_args.neg_test_label_data_paths
            else [],
        )
    ):
        if train_path or pos_train_path:
            train_datasets.append(
                build_the_dataset(
                    data_prefix=train_path,
                    name=f"train_{i}",
                    data_impl=neox_args.data_impl,
                    pack_impl=neox_args.pack_impl,
                    allow_chopped=neox_args.allow_chopped,
                    num_samples=train_num_samples[i],
                    num_epochs=train_epochs,
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    label_prefix=train_label_path,
                    dataset_impl=neox_args.dataset_impl,
                    pos_data_prefix=pos_train_path,
                    neg_data_prefix=neg_train_path,
                    pos_label_prefix=pos_train_label_path,
                    neg_label_prefix=neg_train_label_path,
                    precompute_model_name=neox_args.precompute_model_name,
                    reward_prefix=train_reward_path,
                )
            )

        if valid_path or pos_valid_path:
            valid_datasets.append(
                build_the_dataset(
                    data_prefix=valid_path,
                    name=f"valid_{i}",
                    data_impl=neox_args.data_impl,
                    pack_impl=neox_args.pack_impl,
                    allow_chopped=neox_args.allow_chopped,
                    num_samples=valid_num_samples[i],
                    num_epochs=valid_epochs,
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    label_prefix=valid_label_path,
                    dataset_impl=neox_args.dataset_impl,
                    pos_data_prefix=pos_valid_path,
                    neg_data_prefix=neg_valid_path,
                    pos_label_prefix=pos_valid_label_path,
                    neg_label_prefix=neg_valid_label_path,
                    precompute_model_name=neox_args.precompute_model_name,
                    reward_prefix=valid_reward_path,
                )
            )

        if test_path or pos_test_path:
            test_datasets.append(
                build_the_dataset(
                    data_prefix=test_path,
                    name=f"test_{i}",
                    data_impl=neox_args.data_impl,
                    pack_impl=neox_args.pack_impl,
                    allow_chopped=neox_args.allow_chopped,
                    num_samples=test_num_samples[i],
                    num_epochs=test_epochs,
                    seq_length=neox_args.seq_length,
                    seed=neox_args.seed,
                    skip_warmup=(not neox_args.mmap_warmup),
                    build_index_mappings=build_index_mappings,
                    label_prefix=test_label_path,
                    dataset_impl=neox_args.dataset_impl,
                    pos_data_prefix=pos_test_path,
                    neg_data_prefix=neg_test_path,
                    pos_label_prefix=pos_test_label_path,
                    neg_label_prefix=neg_test_label_path,
                    precompute_model_name=neox_args.precompute_model_name,
                    reward_prefix=test_reward_path,
                )
            )
    return train_datasets, valid_datasets, test_datasets


def weights_by_num_docs(l: list, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """
    if len(l) == 1:
        return [1.0]

    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i**alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights


def validate_train_epochs(neox_args):
    """Check for unsupported neox_args when using train_epochs instead of train_iters"""
    if neox_args.train_epochs is None:
        return

    if neox_args.train_epochs and neox_args.train_iters:
        raise ValueError(
            "Cannot specify both train epochs and train iters simultaneously"
        )

    if neox_args.pack_impl != "packed":
        raise ValueError(
            "Packing implementations other than 'packed' are currently unsupported with train_epochs"
        )

    if neox_args.weight_by_num_documents:
        raise ValueError(
            "Weighting by number of documents is currently unsupported with train_epochs"
        )

    if neox_args.train_data_weights and (
        not all(weight == 1.0 for weight in neox_args.train_data_weights)
    ):
        raise ValueError(
            "train_data_weights != None is currently unsupported with train_epochs"
        )

    if neox_args.dataset_impl != "gpt2":
        raise ValueError(
            "non gpt2 datasets are not currently unsupported with train_epochs"
        )


def build_train_valid_test_data_loaders(neox_args):
    """XXX"""

    validate_train_epochs(neox_args)

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = (
            mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        )
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model and context parallel group.
    if (
        pipe_load
        and (neox_args.dataset_impl == "online")
        and (mpu.get_model_parallel_rank() == 0)
        and (mpu.get_context_parallel_rank() == 0)
    ):
        # Can skip most of the work...
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        # Build datasets...
        print(
            f"train_iters: {train_iters}, eval_iters: {eval_iters}, test_iters: {test_iters}"
        )
        train_datasets = OnlineDataset(
            leave_one_out=neox_args.reinforce_leave_one_out,
            data_split="train",
            num_samples=train_iters * neox_args.train_batch_size,
            seq_length=neox_args.seq_length,
            dataserver_ips=neox_args.online_dataserver_ips,
            dataserver_ports=neox_args.online_dataserver_ports,
        )
        valid_datasets = OnlineDataset(
            leave_one_out=neox_args.reinforce_leave_one_out,
            data_split="valid",
            num_samples=eval_iters * neox_args.train_batch_size,
            seq_length=neox_args.seq_length,
            dataserver_ips=neox_args.online_dataserver_ips,
            dataserver_ports=neox_args.online_dataserver_ports,
        )
        test_datasets = OnlineDataset(
            leave_one_out=neox_args.reinforce_leave_one_out,
            data_split="test",
            num_samples=test_iters * neox_args.train_batch_size,
            seq_length=neox_args.seq_length,
            dataserver_ips=neox_args.online_dataserver_ips,
            dataserver_ports=neox_args.online_dataserver_ports,
        )
        # print length of datasets
        # Build dataloders.
        train_dataloader = make_data_loader(train_datasets, neox_args=neox_args)
        valid_dataloader = make_data_loader(valid_datasets, neox_args=neox_args)
        test_dataloader = make_data_loader(test_datasets, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    elif mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        if neox_args.train_iters is not None:
            train_iters = neox_args.train_iters
            eval_iters = (
                train_iters // neox_args.eval_interval + 1
            ) * neox_args.eval_iters
            test_iters = neox_args.eval_iters
            train_val_test_num_samples = [
                train_iters * neox_args.train_batch_size,
                eval_iters * neox_args.train_batch_size,
                test_iters * neox_args.train_batch_size,
            ]
            train_val_test_epochs = [None, None, None]
        elif neox_args.train_epochs is not None:
            train_val_test_num_samples = [None, None, None]
            train_val_test_epochs = [1, 1, 1]

        if (neox_args.train_data_paths) or (neox_args.pos_train_data_paths):
            # when individual train / valid / test data paths are provided
            # normalize weight values and get num samples for each dataset
            train_weights, train_num_samples = get_normalized_weights_and_num_samples(
                neox_args.train_data_weights, train_val_test_num_samples[0]
            )
            valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
                neox_args.valid_data_weights, train_val_test_num_samples[1]
            )
            test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                neox_args.test_data_weights, train_val_test_num_samples[2]
            )

            # build individual datasets
            train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                neox_args,
                train_num_samples,
                valid_num_samples,
                test_num_samples,
                train_val_test_epochs[0],
                train_val_test_epochs[1],
                train_val_test_epochs[2],
                build_index_mappings=not neox_args.weight_by_num_documents,
            )

            if neox_args.weight_by_num_documents:
                # gets the number of documents in each datapath
                get_num_docs_list = lambda datasets: [
                    dataset.indexed_dataset.sizes.shape[0] for dataset in datasets
                ]
                train_num_docs, valid_num_docs, test_num_docs = (
                    get_num_docs_list(train_datasets),
                    get_num_docs_list(valid_datasets),
                    get_num_docs_list(test_datasets),
                )

                # builds weights according to alpha + the number of docs
                fn = partial(
                    weights_by_num_docs, alpha=neox_args.weighted_sampler_alpha
                )
                train_weights, valid_weights, test_weights = (
                    fn(train_num_docs),
                    fn(valid_num_docs),
                    fn(test_num_docs),
                )
                (
                    train_weights,
                    train_num_samples,
                ) = get_normalized_weights_and_num_samples(
                    train_weights, train_val_test_num_samples[0]
                )
                (
                    valid_weights,
                    valid_num_samples,
                ) = get_normalized_weights_and_num_samples(
                    valid_weights, train_val_test_num_samples[1]
                )
                test_weights, test_num_samples = get_normalized_weights_and_num_samples(
                    test_weights, train_val_test_num_samples[2]
                )

                # rebuild datasets weighted according to new weights
                train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
                    neox_args,
                    train_num_samples,
                    valid_num_samples,
                    test_num_samples,
                    train_val_test_epochs[0],
                    train_val_test_epochs[1],
                    train_val_test_epochs[2],
                )

            if train_datasets:
                train_ds = BlendableDataset(train_datasets, train_weights)
            if valid_datasets:
                valid_ds = BlendableDataset(valid_datasets, valid_weights)
            if test_datasets:
                test_ds = BlendableDataset(test_datasets, test_weights)
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix=neox_args.data_path,
                use_shared_fs=neox_args.use_shared_fs,
                data_impl=neox_args.data_impl,
                splits_string=neox_args.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                train_valid_test_epochs=train_val_test_epochs,
                seq_length=neox_args.seq_length,
                seed=neox_args.seed,
                skip_warmup=(not neox_args.mmap_warmup),
                pack_impl=neox_args.pack_impl,
                allow_chopped=neox_args.allow_chopped,
            )

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds, neox_args=neox_args)
        valid_dataloader = make_data_loader(valid_ds, neox_args=neox_args)
        test_dataloader = make_data_loader(test_ds, neox_args=neox_args)

        # Flags to know if we need to do training/validation/testing.
        if neox_args.train_epochs:
            do_train = train_dataloader is not None
            do_valid = valid_dataloader is not None
            do_test = test_dataloader is not None
        else:
            do_train = train_dataloader is not None and neox_args.train_iters > 0
            do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
            do_test = test_dataloader is not None and neox_args.eval_iters > 0

        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        # The same data should be used for the model parallel and context parallel groups
        torch.distributed.broadcast(
            flags,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group(),
        )
        torch.distributed.broadcast(
            flags,
            mpu.get_context_parallel_src_rank(),
            group=mpu.get_context_parallel_group(),
        )
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()
    data_loaders = {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
    }
    return data_loaders


def shift_and_wrap_data_loaders(neox_args, data_loaders, loop=True):
    """Shift start iteration and wrap data_loaders in iterators"""
    train_dataloader = data_loaders["train"]
    valid_dataloader = data_loaders["valid"]
    test_dataloader = data_loaders["test"]

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (
            neox_args.iteration * neox_args.gradient_accumulation_steps
        ) % len(train_dataloader)
        print_rank_0(
            "setting training data start iteration to {}".format(
                train_dataloader.batch_sampler.start_iter
            )
        )
    if valid_dataloader is not None:
        start_iter_val = (
            (neox_args.iteration * neox_args.gradient_accumulation_steps)
            // neox_args.eval_interval
        ) * neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % len(
            valid_dataloader
        )
        print_rank_0(
            "setting validation data start iteration to {}".format(
                valid_dataloader.batch_sampler.start_iter
            )
        )

    def loop_iterator(data_loader):
        while True:
            for x in data_loader:
                yield x
            data_loader.start_iter = 0

    # Build iterators.
    if train_dataloader is not None:
        if loop:
            train_data_iterator = cycle(train_dataloader)
        else:
            train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        if loop:
            valid_data_iterator = cycle(valid_dataloader)
        else:
            valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        if loop:
            test_data_iterator = cycle(test_dataloader)
        else:
            test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)
