import numbers
import os
import queue as Queue
import threading
from typing import Iterable, Union, List

# import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from glob import glob
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir: str,
    local_rank: int,
    batch_size: int,
    dali: bool = False,
    dali_aug: bool = False,
    seed: int = 2048,
    num_workers: int = 2,
    webdataset: bool = False,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    # elif os.path.exists(rec) and os.path.exists(idx):
    #     train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # WebDataset shards
    elif webdataset:
        train_set = _build_webdataset(root_dir)

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank, dali_aug=dali_aug)

    rank, world_size = get_dist_info()
    # IterableDataset (WebDataset) cannot use a Sampler; use internal sharding and shuffling
    if webdataset:
        if seed is None:
            init_fn = None
        else:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
        )
    else:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

        if seed is None:
            init_fn = None
        else:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
        )

    return train_loader


def _build_webdataset(src: str):
    """Build a WebDataset pipeline from a directory, pattern, list file, or URL.

    Expected sample keys: image ext in {jpg,jpeg,png} and integer label in 'cls'.
    Performs simple augmentation: random horizontal flip and normalization to [-1, 1].
    The dataset is sharded across ranks and workers automatically.
    """
    try:
        import webdataset as wds
    except Exception as e:
        raise RuntimeError(
            "webdataset package is required for WebDataset training. Please install it."
        ) from e

    # Resolve shards input: directory -> *.tar, file list -> read lines, else use as-is
    shards: Union[str, List[str]]
    if os.path.isdir(src):
        shards = sorted(glob(os.path.join(src, "*.tar")))
        if len(shards) == 0:
            raise FileNotFoundError(f"No .tar shards found in directory: {src}")
    elif os.path.isfile(src) and src.lower().endswith(".txt"):
        with open(src, "r", encoding="utf-8") as f:
            shards = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if len(shards) == 0:
            raise FileNotFoundError(f"No shard entries found in list file: {src}")
    else:
        shards = src

    # Transforms matching ImageFolder path
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def decode_label(lbl):
        # Robustly convert various label representations to integer tensor
        import numpy as _np
        import json as _json
        # If label is a JSON dict, extract 'cls'
        if isinstance(lbl, dict) and "cls" in lbl:
            return torch.tensor(int(lbl["cls"]), dtype=torch.long)
        if isinstance(lbl, (bytes, bytearray)):
            try:
                lbl = int(lbl.decode("utf-8").strip())
            except Exception:
                # Try JSON then numpy buffer
                try:
                    obj = _json.loads(lbl.decode("utf-8"))
                    if isinstance(obj, dict) and "cls" in obj:
                        lbl = int(obj["cls"])
                    else:
                        lbl = int(obj)
                except Exception:
                    try:
                        lbl = int(_np.frombuffer(lbl, dtype=_np.int64).flatten()[0])
                    except Exception:
                        raise ValueError("Unsupported label byte format in WebDataset sample")
        elif isinstance(lbl, (int,)):
            pass
        elif hasattr(lbl, "item"):
            # torch/np scalar
            try:
                lbl = int(lbl.item())
            except Exception:
                lbl = int(lbl)
        elif isinstance(lbl, (list, tuple)) and len(lbl) > 0:
            lbl = int(lbl[0])
        else:
            # last resort
            lbl = int(lbl)
        return torch.tensor(lbl, dtype=torch.long)

    def preprocess(sample):
        img, lbl = sample
        # img is PIL.Image from decode('pil')
        img = transform(img)
        lbl = decode_label(lbl)
        return img, lbl

    dataset = (
        wds.WebDataset(
            shards,
            shardshuffle=1000,  # positive int to avoid compat warning
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            handler=wds.warn_and_continue,
            empty_check=False,  # allow some workers to have zero shards without raising
        )
        .shuffle(10000)
    .decode("pil")
        # Accept common image extensions; for label, try various common keys
        .to_tuple("jpg;jpeg;png;webp", "cls;cls.txt;label;label.txt;json")
        .map(preprocess)
    )

    return dataset

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


# class MXFaceDataset(Dataset):
#     def __init__(self, root_dir, local_rank):
#         super(MXFaceDataset, self).__init__()
#         self.transform = transforms.Compose(
#             [transforms.ToPILImage(),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#              ])
#         self.root_dir = root_dir
#         self.local_rank = local_rank
#         path_imgrec = os.path.join(root_dir, 'train.rec')
#         path_imgidx = os.path.join(root_dir, 'train.idx')
#         self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#         s = self.imgrec.read_idx(0)
#         header, _ = mx.recordio.unpack(s)
#         if header.flag > 0:
#             self.header0 = (int(header.label[0]), int(header.label[1]))
#             self.imgidx = np.array(range(1, int(header.label[0])))
#         else:
#             self.imgidx = np.array(list(self.imgrec.keys))
#
#     def __getitem__(self, index):
#         idx = self.imgidx[index]
#         s = self.imgrec.read_idx(idx)
#         header, img = mx.recordio.unpack(s)
#         label = header.label
#         if not isinstance(label, numbers.Number):
#             label = label[0]
#         label = torch.tensor(label, dtype=torch.long)
#         sample = mx.image.imdecode(img).asnumpy()
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, label
#
#     def __len__(self):
#         return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)
        img = fn.resize(img, size=(image_size, image_size))
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


class DALIWarper(object):
    @torch.no_grad()
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
