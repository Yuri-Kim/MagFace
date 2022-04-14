#!/usr/bin/env python
import sys
sys.path.append("..")

from utils import cv2_trans as transforms
from termcolor import cprint
import cv2
import torchvision
import torch.utils.data as data
import torch
import random
import numpy as np
import os
import warnings

import mxnet as mx
import numbers


class MagTrainDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        self.weight = {}
        self.im_names = []
        self.targets = []
        self.pre_types = []
        with open(self.ann_file) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[2]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]
        img = cv2.imread(im_name)

        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.im_names)


def train_loader(args):
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = MagTrainDataset(
        args.train_list,
        transform=train_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader


class WebFaceDataset(data.Dataset):
    def __init__(self, root_dir, local_rank):
        super(WebFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        sample_cv =  cv2.imread(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample_cv, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(data.Dataset):
    def __init__(self, local_rank):
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


def train_loader_webface(args):
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = WebFaceDataset(
        args.train_list,
        transform=train_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader