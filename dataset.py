import numbers
import os
import queue as Queue
import threading
from typing import Iterable, Union, List, Any, cast

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
from nvidia.dali import types  # added


class SafeImageFolder(ImageFolder):
    """ImageFolder that ignores hidden directories (e.g., .cache) as classes.

    Torchvision's default ImageFolder treats every subdirectory under the root
    as a class. If a hidden folder like ".cache" exists and contains no valid
    images, it raises FileNotFoundError. This subclass filters out directories
    that start with a dot.
    """

    @classmethod
    def find_classes(cls, directory: str):  # type: ignore[override]
        classes = [
            d.name
            for d in os.scandir(directory)
            if d.is_dir() and not d.name.startswith(".")
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_dataloader(
    root_dir: str,
    local_rank: int,
    batch_size: int,
    dali: bool = False,
    dali_aug: bool = False,
    seed: int = 2048,
    num_workers: int = 2,
    dataset_type: str = "imagefolder",
    device_type: str | None = None,
    ) -> Iterable:

    train_set = None
    ds_type = (dataset_type or "imagefolder").lower()

    transform = transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         ])

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # If DALI is enabled, return DALI iterator immediately to avoid constructing
    # any CPU-side dataset like ImageFolder that may scan hidden directories.
    if dali:
        return cast(Iterable, dali_data_iter(
            batch_size=batch_size,
            src=root_dir,
            dataset_type=ds_type,
            num_threads=2,
            local_rank=local_rank,
            dali_aug=dali_aug,
            device_type=device_type))

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # MXNet RecordIO support has been removed

    # WebDataset shards
    elif ds_type == "webdataset" and not dali:
        # Python WebDataset pipeline (non-DALI)
        train_set = _build_webdataset(root_dir)

    # Image Folder
    elif ds_type == "imagefolder" and not dali:
        train_set = SafeImageFolder(root_dir, transform)
    elif not dali and ds_type == "tfrecord":
        raise ValueError("dataset_type='tfrecord' is no longer supported. Use 'imagefolder' or 'webdataset'.")
    else:
        # Fallback for unknown types when not using DALI
        train_set = SafeImageFolder(root_dir, transform)

    # DALI already handled above

    rank, world_size = get_dist_info()
    pin_memory_enabled = torch.cuda.is_available() and (device_type in (None, "cuda")) and ds_type != "webdataset"

    # IterableDataset (WebDataset) cannot use a Sampler; use internal sharding and shuffling
    if ds_type == "webdataset" and not dali:
        if seed is None:
            init_fn = None
        else:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        train_loader = DataLoaderX(
            local_rank=local_rank,
            device_type=device_type,
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory_enabled,
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
            device_type=device_type,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory_enabled,
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
        import webdataset as wds  # type: ignore
        from typing import Any, cast as _cast
        wds = _cast(Any, wds)
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

    def decode_label(lbl: Any) -> torch.Tensor:
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
                        lbl = int(obj["cls"])  # type: ignore[arg-type]
                    else:
                        lbl = int(obj)  # type: ignore[arg-type]
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
                lbl = int(lbl.item())  # type: ignore[arg-type]
            except Exception:
                lbl = int(lbl)  # type: ignore[arg-type]
        elif isinstance(lbl, (list, tuple)) and len(lbl) > 0:
            lbl = int(lbl[0])  # type: ignore[arg-type]
        else:
            # last resort
            lbl = int(lbl)  # type: ignore[arg-type]
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
            nodesplitter=getattr(wds, 'split_by_node', None),
            workersplitter=getattr(wds, 'split_by_worker', None),
            handler=getattr(wds, 'warn_and_continue', None),
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
    def __init__(self, generator, local_rank, device_type: str | None, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.device_type = (device_type or ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")))
        self.daemon = True
        self.start()

    def run(self):
        # Set device context only for accelerator backends
        if self.device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        elif self.device_type == "xpu" and torch.xpu.is_available():
            torch.xpu.set_device(self.local_rank)
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

    def __init__(self, local_rank, device_type: str | None = None, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.local_rank = local_rank
        self.device_type = (device_type or ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")))
        # Use CUDA stream only on CUDA; for other backends, disable custom stream prefetch to avoid backend-specific APIs
        self.stream = torch.cuda.Stream(local_rank) if (self.device_type == "cuda" and torch.cuda.is_available()) else None

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank, self.device_type)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        if self.stream is not None and self.device_type == "cuda":
            with torch.cuda.stream(self.stream):
                # Move batch tensors to the target CUDA device asynchronously
                device = torch.device("cuda", self.local_rank)
                for k in range(len(self.batch)):
                    self.batch[k] = self.batch[k].to(device=device, non_blocking=True)
        else:
            # For non-CUDA backends, keep data on CPU; the main training loop will move tensors appropriately.
            pass

    def __next__(self):
        if self.stream is not None and self.device_type == "cuda":
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


# WebDataset の 'cls' ラベルを int64 スカラー(形状[1])に正規化（ASCII固定解釈）
def _wds_label_to_int64_np(x):
    # x: np.ndarray(dtype=uint8, shape=(N,)) を想定
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.dtype != np.uint8:
        x = x.view(np.uint8)
    bs = x.ravel().tobytes()

    # ASCII のみ許可して数値化
    allowed = b"0123456789+- \t\r\n"
    if not bs or not set(bs) <= set(allowed):
        raise RuntimeError("Label is not valid ASCII digits.")
    s = bs.decode("ascii", errors="ignore").strip()
    if not s or not s.lstrip("+-").isdigit():
        raise RuntimeError("Label ASCII is not an integer.")
    val = int(s)
    return np.array([np.int64(val)], dtype=np.int64)

def dali_data_iter(
    batch_size: int,
    src: str,
    dataset_type: str,
    num_threads: int,
    initial_fill=32768,
    random_shuffle=True,
    prefetch_queue_depth=1,
    local_rank=0,
    name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5),
    dali_aug=False,
    device_type: str | None = None
    ):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    # DALI is supported only with CUDA devices. Guard early.
    if not torch.cuda.is_available():
        raise RuntimeError("DALI pipeline requires CUDA-enabled PyTorch. Set config.dali=False for CPU/XPU training.")

    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    def dali_random_resize(img, resize_size, image_size=112):
        img = fn.resize(img, resize_x=resize_size, resize_y=resize_size)  # type: ignore[arg-type]
        img = fn.resize(img, resize_x=image_size, resize_y=image_size)  # type: ignore[arg-type]
        return img
    def dali_random_gaussian_blur(img, window_size):
        img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
        return img
    def dali_random_gray(img, prob_gray):
        saturate = fn.random.coin_flip(probability=1 - prob_gray)
        saturate = fn.cast(saturate, dtype=types.FLOAT)  # type: ignore[attr-defined]
        img = fn.hsv(img, saturation=saturate)
        return img
    def dali_random_hsv(img, hue, saturation):
        img = fn.hsv(img, hue=hue, saturation=saturation)
        return img
    def multiplexing(condition, true_case, false_case):
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case

    condition_resize = fn.random.coin_flip(probability=0.1)
    size_resize = fn.random.uniform(range=(int(112 * 0.5), int(112 * 0.8)), dtype=types.FLOAT)  # type: ignore[attr-defined]
    condition_blur = fn.random.coin_flip(probability=0.2)
    window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)  # type: ignore[attr-defined]
    condition_flip = fn.random.coin_flip(probability=0.5)
    condition_hsv = fn.random.coin_flip(probability=0.2)
    hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)  # type: ignore[attr-defined]
    hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)  # type: ignore[attr-defined]

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        # Select reader based on dataset_type
        reader_type = (dataset_type or "imagefolder").lower()
        if reader_type == "imagefolder":
            jpegs, labels = fn.readers.file(
                file_root=src,
                random_shuffle=random_shuffle,
                # pad_last_batch を True にし、全 rank で常に同一 batch_size を維持し、
                # PartialFC の all_gather でサイズ不一致による NCCL ハングを防ぐ。
                pad_last_batch=True,
                name=name,
                shard_id=rank,
                num_shards=world_size,
            )
            images = fn.decoders.image(jpegs, device="mixed")
            labels = fn.cast(labels, dtype=types.INT64)
            labels = fn.reshape(labels, shape=[1])
        elif reader_type == "webdataset":
            # Resolve shards list
            from glob import glob as _glob
            if os.path.isdir(src):
                shards = sorted(_glob(os.path.join(src, "*.tar")))
            elif os.path.isfile(src) and src.lower().endswith(".txt"):
                with open(src, "r", encoding="utf-8") as _f:
                    shards = [ln.strip() for ln in _f if ln.strip() and not ln.strip().startswith('#')]
            else:
                shards = [src]
            if len(shards) == 0:
                raise FileNotFoundError(f"No WebDataset shards found for: {src}")
            wds_out = fn.readers.webdataset(
                paths=shards,
                ext=["jpg", "cls"],
                random_shuffle=random_shuffle,
                # WebDataset でも最終バッチが揃わないケースがあるため True に設定。
                pad_last_batch=True,
                name=name,
                shard_id=rank,
                num_shards=world_size,
            )
            # readers.webdataset returns a list of outputs; cast to appease type checker
            jpegs = wds_out[0]  # type: ignore[index]
            labels = wds_out[1]  # type: ignore[index]
            images = fn.decoders.image(jpegs, device="mixed")
            # 可変長ASCII/バイナリを CPU 上で int64[1] に正規化（バッチ内で形状を揃える）
            labels = fn.python_function(labels, function=_wds_label_to_int64_np, batch_processing=False)
            # すでに (1,) を返すので追加の reshape は不要
        else:
            raise ValueError(
                f"Unsupported dataset_type='{reader_type}' for DALI. Use 'imagefolder' or 'webdataset'."
            )
        if dali_aug:
            images = fn.cast(images, dtype=types.UINT8)  # type: ignore[attr-defined]
            images = multiplexing(condition_resize, dali_random_resize(images, size_resize, image_size=112), images)
            images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
            images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
            images = dali_random_gray(images, 0.1)

        # Ensure uniform image size across the batch for tensorization
        images = fn.resize(images, resize_x=112, resize_y=112)

        # Help static type checker: DALI accepts DataNode here
        images = cast(Any, images)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            mean=mean,
            std=std,
            mirror=condition_flip,
            output_layout="CHW",
        )  # type: ignore[attr-defined, arg-type]
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(
        DALIClassificationIterator(
            pipelines=[pipe],
            reader_name=name,
            ),
        device_type=device_type
    )


class DALIWarper(object):
    @torch.no_grad()
    def __init__(self, dali_iter, device_type: str | None = None):
        self.iter = dali_iter
        self.device_type = (device_type or ("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")))

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        if self.device_type == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self.device_type == "xpu" and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
        tensor_data = data_dict['data'].to(device)
        tensor_label: torch.Tensor = data_dict['label'].to(device).long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
