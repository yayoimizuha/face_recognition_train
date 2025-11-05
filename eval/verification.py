"""Helper for evaluation on the Labeled Faces in the Wild dataset"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle
from io import BytesIO

# import mxnet as mx
import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from PIL import Image

# from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            sorted_indices = np.argsort(far_train)
            x_sorted = far_train[sorted_indices]
            y_sorted = thresholds[sorted_indices]
            for i in range(1, len(x_sorted)):
                if x_sorted[i] <= x_sorted[i - 1]:
                    x_sorted[i] = x_sorted[i - 1] + 1e-9
            f = interpolate.interp1d(x_sorted, y_sorted, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size):
    with open(path, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    data_list = []
    for _ in range(2):
        # CPU側ではuint8で保持し、転送量とメモリ消費を削減
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]), dtype=torch.uint8)
        data_list.append(data)

    for i in range(len(issame_list) * 2):
        _bin = bins[i]

        img = Image.open(BytesIO(_bin)).convert("RGB")
        w, h = img.size
        short_side = min(w, h)
        if short_side != image_size[0]:
            scale = image_size[0] / short_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        for flip_idx, do_flip in enumerate([False, True]):
            img_to_process = img
            if do_flip:
                img_to_process = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            img_np = np.array(img_to_process, dtype=np.uint8)
            img_np = img_np.transpose((2, 0, 1))

            data_list[flip_idx][i][:] = torch.from_numpy(img_np.copy())

        if (i + 1) % 1000 == 0:
            print(f"loading bin... {i + 1}")

    print(f"Final tensor shape: {data_list[0].shape}")
    return data_list, issame_list


# @torch.no_grad()
# def load_bin(path, image_size):
#     try:
#         with open(path, 'rb') as f:
#             bins, issame_list = pickle.load(f)  # py2
#     except UnicodeDecodeError as e:
#         with open(path, 'rb') as f:
#             bins, issame_list = pickle.load(f, encoding='bytes')  # py3
#     data_list = []
#     for flip in [0, 1]:
#         data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
#         data_list.append(data)
#     for idx in range(len(issame_list) * 2):
#         _bin = bins[idx]
#         img = mx.image.imdecode(_bin)
#         if img.shape[1] != image_size[0]:
#             img = mx.image.resize_short(img, image_size[0])
#         img = nd.transpose(img, axes=(2, 0, 1))
#         for flip in [0, 1]:
#             if flip == 1:
#                 img = mx.ndarray.flip(data=img, axis=2)
#             data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
#         if idx % 1000 == 0:
#             print('loading bin', idx)
#     print(data_list[0].shape)
#     return data_list, issame_list


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    """シンプルで安全な高速化: 重複推論の排除とTorchベース前処理。

    - 末尾の不完全バッチを含む可変バッチで逐次処理（重複窓スライスを廃止）
    - 正規化はTorchで実施し、各バッチでGPUへ送って推論
    - 埋め込みはCPU側に一括確保して書き込み、最後にNumPyへ変換
    """
    print("testing verification..")

    backbone.eval()
    # 固定解像度での推論を想定しcuDNNのベンチマークを有効化
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    # 学習可能パラメータが存在する前提でデバイスを取得
    device = next(backbone.parameters()).device

    data_list = data_set[0]
    issame_list = data_set[1]

    embeddings_list_gpu = []
    time_consumed = 0.0

    for i in range(len(data_list)):
        data_cpu: torch.Tensor = data_list[i]  # uint8, [N, 3, H, W] on CPU
        total = int(data_cpu.shape[0])

        embeddings_t = None  # GPU tensor [N, D]
        data_gpu = data_cpu.to(device)
        # flipごとに一度だけ正規化（[-1, 1]）を実施
        data_gpu = data_gpu.float()
        data_gpu.mul_(1.0 / 127.5).add_(-1.0)

        start_idx = 0
        t_flip0 = datetime.datetime.now()
        while start_idx < total:
            end_idx = min(start_idx + batch_size * 4, total)
            # 既にGPU上にあり正規化済みのテンソルをスライス
            imgs = data_gpu[start_idx:end_idx]

            net_out: torch.Tensor = backbone(imgs)  # [B, D]

            if embeddings_t is None:
                feat_dim = int(net_out.shape[1])
                embeddings_t = torch.empty((total, feat_dim), dtype=torch.float32, device=device)

            embeddings_t[start_idx:end_idx] = net_out.detach().to(torch.float32)
            start_idx = end_idx

        t_flip1 = datetime.datetime.now()
        time_consumed += (t_flip1 - t_flip0).total_seconds()

        # このflipの埋め込みをGPUのまま保持
        embeddings_list_gpu.append(embeddings_t)

    # 埋め込みノルムの簡易統計（GPU上で計算）
    norms = torch.linalg.norm(embeddings_list_gpu[0], dim=1)
    _xnorm = float(norms.mean().item()) if norms.numel() > 0 else 0.0

    # flipなしの値は互換性維持のためダミー
    acc1 = 0.0
    std1 = 0.0

    # flip有りで集計（和を取ってからL2正規化）
    embeddings = embeddings_list_gpu[0] + embeddings_list_gpu[1]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.detach().cpu().numpy()
    print(embeddings_np.shape)
    print("infer time", time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings_np, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    # 返却互換性のため、リストはNumPyへ変換して返す
    embeddings_list = [t.detach().cpu().numpy() for t in embeddings_list_gpu]
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


# def dumpR(data_set,
#           backbone,
#           batch_size,
#           name='',
#           data_extra=None,
#           label_shape=None):
#     print('dump verification embedding..')
#     data_list = data_set[0]
#     issame_list = data_set[1]
#     embeddings_list = []
#     time_consumed = 0.0
#     for i in range(len(data_list)):
#         data = data_list[i]
#         embeddings = None
#         ba = 0
#         while ba < data.shape[0]:
#             bb = min(ba + batch_size, data.shape[0])
#             count = bb - ba
#
#             _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
#             time0 = datetime.datetime.now()
#             if data_extra is None:
#                 db = mx.io.DataBatch(data=(_data,), label=(_label,))
#             else:
#                 db = mx.io.DataBatch(data=(_data, _data_extra),
#                                      label=(_label,))
#             model.forward(db, is_train=False)
#             net_out = model.get_outputs()
#             _embeddings = net_out[0].asnumpy()
#             time_now = datetime.datetime.now()
#             diff = time_now - time0
#             time_consumed += diff.total_seconds()
#             if embeddings is None:
#                 embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
#             embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
#             ba = bb
#         embeddings_list.append(embeddings)
#     embeddings = embeddings_list[0] + embeddings_list[1]
#     embeddings = sklearn.preprocessing.normalize(embeddings)
#     actual_issame = np.asarray(issame_list)
#     outname = os.path.join('temp.bin')
#     with open(outname, 'wb') as f:
#         pickle.dump((embeddings, issame_list),
#                     f,
#                     protocol=pickle.HIGHEST_PROTOCOL)


# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='do verification')
#     # general
#     parser.add_argument('--data-dir', default='', help='')
#     parser.add_argument('--model',
#                         default='../model/softmax,50',
#                         help='path to load model.')
#     parser.add_argument('--target',
#                         default='lfw,cfp_ff,cfp_fp,agedb_30',
#                         help='test targets.')
#     parser.add_argument('--gpu', default=0, type=int, help='gpu id')
#     parser.add_argument('--batch-size', default=32, type=int, help='')
#     parser.add_argument('--max', default='', type=str, help='')
#     parser.add_argument('--mode', default=0, type=int, help='')
#     parser.add_argument('--nfolds', default=10, type=int, help='')
#     args = parser.parse_args()
#     image_size = [112, 112]
#     print('image_size', image_size)
#     ctx = mx.gpu(args.gpu)
#     nets = []
#     vec = args.model.split(',')
#     prefix = args.model.split(',')[0]
#     epochs = []
#     if len(vec) == 1:
#         pdir = os.path.dirname(prefix)
#         for fname in os.listdir(pdir):
#             if not fname.endswith('.params'):
#                 continue
#             _file = os.path.join(pdir, fname)
#             if _file.startswith(prefix):
#                 epoch = int(fname.split('.')[0].split('-')[1])
#                 epochs.append(epoch)
#         epochs = sorted(epochs, reverse=True)
#         if len(args.max) > 0:
#             _max = [int(x) for x in args.max.split(',')]
#             assert len(_max) == 2
#             if len(epochs) > _max[1]:
#                 epochs = epochs[_max[0]:_max[1]]
#
#     else:
#         epochs = [int(x) for x in vec[1].split('|')]
#     print('model number', len(epochs))
#     time0 = datetime.datetime.now()
#     for epoch in epochs:
#         print('loading', prefix, epoch)
#         sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
#         # arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
#         all_layers = sym.get_internals()
#         sym = all_layers['fc1_output']
#         model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#         # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
#         model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0],
#                                           image_size[1]))])
#         model.set_params(arg_params, aux_params)
#         nets.append(model)
#     time_now = datetime.datetime.now()
#     diff = time_now - time0
#     print('model loading time', diff.total_seconds())
#
#     ver_list = []
#     ver_name_list = []
#     for name in args.target.split(','):
#         path = os.path.join(args.data_dir, name + ".bin")
#         if os.path.exists(path):
#             print('loading.. ', name)
#             data_set = load_bin(path, image_size)
#             ver_list.append(data_set)
#             ver_name_list.append(name)
#
#     if args.mode == 0:
#         for i in range(len(ver_list)):
#             results = []
#             for model in nets:
#                 acc1, std1, acc2, std2, xnorm, embeddings_list = test(
#                     ver_list[i], model, args.batch_size, args.nfolds)
#                 print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
#                 print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
#                 print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], acc2, std2))
#                 results.append(acc2)
#             print('Max of [%s] is %1.5f' % (ver_name_list[i], np.max(results)))
#     elif args.mode == 1:
#         raise ValueError
#     else:
#         model = nets[0]
#         dumpR(ver_list[0], model, args.batch_size, args.target)
