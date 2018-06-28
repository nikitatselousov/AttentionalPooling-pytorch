#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(yas@meitu.com)


import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import random
from torch.utils import data

import cv2

def default_loader(path):
    return Image.open(path)

class CSDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_transform = img_transform
        self.label_transform = label_transform


        data_dir = root
        # for split in ["train", "trainval", "val"]:
        img_file = osp.join(data_dir, "img_" + split + ".npy")
        lab_file = osp.join(data_dir, "lab_" + split + ".npy")
        ptn_file = osp.join(data_dir, "ptn_" + split + ".npy")
        self.imgs = np.load(img_file)
        self.labels = np.load(lab_file)
        self.ptns = np.load(ptn_file)
        print("Loaded imgs: ", len(self.imgs))
        #self.files[split] = {"imgs": img_file, "labels": lab_file}

    def __len__(self):
        #print(len(self.imgs), self.split)
        return len(self.imgs)

    def __getitem__(self, index):

        
        img = Image.fromarray(self.imgs[index], 'L')
     

        label = int(self.labels[index])
        ptn = int(self.ptns[index])
        if self.img_transform is not None:
            img_o = self.img_transform(img)
            imgs = img_o
        else:
            imgs = img
        if self.label_transform is not None:
            label_o = self.label_transform(label)
            labels = label_o
        else:
            labels = label
        # print np.array(labels)
        return imgs, label, ptn


class CSTestSet(data.Dataset):
    def __init__(self, root, img_transform=None, label_transform=None):
        self.root = root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = collections.defaultdict(list)

        self.data_dir = osp.join(root, "cityspace")
        self.img_names = os.listdir(osp.join(self.data_dir, "leftImg8bit/all_val"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        label_name = "_".join(name.split('.')[0].split('_')[:-1]) + "_gtFine_labelTrainIds.png"
        img = Image.open(osp.join(self.data_dir, "leftImg8bit/all_val", name)).convert('RGB')
        label = Image.open(osp.join(self.data_dir, "gtFine/all_val", label_name)).convert('P')
        size = img.size
        # name = name.split(".")[0]

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label, name, size


if __name__ == '__main__':
    dst = CSDataSet("/root/group_incubation_bj")
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = (imgs).numpy()[0] # torchvision.utils.make_grid(imgs).numpy()
            # img = torchvision.utils.make_grid(imgs).numpy()
            print (img.shape)
            # cv2.imshow("main", img)
            # cv2.waitKey()
            # img = np.transpose(img, (1, 2, 0))
            # img = img[:, :, ::-1]
            print (img.shape)
            plt.imshow(img.squeeze())
            plt.show()
