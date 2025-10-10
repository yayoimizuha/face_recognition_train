import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from typing import cast
from collections.abc import Sized
from torch.utils.data import DistributedSampler as _DistributedSampler


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        if cuda_deterministic:  # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def sync_random_seed(seed=None, device="auto"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on. One of 'cuda'|'xpu'|'cpu'|'auto'.
            Default to 'auto' which selects 'cuda' if available, else 'xpu' if available, else 'cpu'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if device == "auto":
        if torch.cuda.is_available():
            use_device = "cuda"
        elif torch.xpu.is_available():
            use_device = "xpu"
        else:
            use_device = "cpu"
    else:
        # validate requested device
        if device == "cuda" and not torch.cuda.is_available():
            use_device = "cpu"
        elif device == "xpu" and not torch.xpu.is_available():
            use_device = "cpu"
        else:
            use_device = device
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=use_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=use_device)

    dist.broadcast(random_num, src=0)

    return random_num.item()


class DistributedSampler(_DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,  # world_size
        rank=None,  # local_rank
        shuffle=True,
        seed=0,
    ):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(int(self.epoch + self.seed))
            ds = cast(Sized, self.dataset)
            indices = torch.randperm(len(ds), generator=g).tolist()
        else:
            ds = cast(Sized, self.dataset)
            indices = torch.arange(len(ds)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
