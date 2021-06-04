import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import random
import cv2
import os, sys
from color_adjust import Color
from cv_proc import show
from face_distort import random_transform_reso

colorADJ = Color()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PairData(object):
    def __init__(self, baseDir, batch_size):
        self.baseDir = baseDir
        self.test_num = 2000
        self.test_paths, self.train_paths = self.get_train_and_test_files()
        self.batch_size = batch_size
        self.tr_cur = 0
        self.tr_len = len(self.train_paths)
        self.te_cur = 0
        self.te_len = len(self.test_paths)
    
    def get_train_and_test_files(self):
        fList = [self.baseDir+'/'+p for p in os.listdir(self.baseDir)]
        random.shuffle(fList)
        te_list = fList[:self.test_num]
        tr_list = fList[self.test_num:]
        return te_list, tr_list
    
    def to_tensor(self, mat):
        mat = torch.from_numpy(mat).float()
        mat = mat.permute(0, 3, 1, 2)
        return mat

    def next_test(self):
        if self.te_cur+self.batch_size>self.te_len:
            self.te_cur = 0
            random.shuffle(self.test_paths)
        paths = self.test_paths[self.te_cur:self.te_cur+self.batch_size]
        inputs, labels = self.get_batch(paths, False)
        self.te_cur += self.batch_size
        return inputs, labels

    def next_train(self):
        if self.tr_cur+self.batch_size>self.tr_len:
            self.tr_cur = 0
            random.shuffle(self.train_paths)
        paths = self.train_paths[self.tr_cur:self.tr_cur+self.batch_size]
        inputs, labels = self.get_batch(paths)
        self.tr_cur += self.batch_size
        return inputs, labels

    def get_batch(self, paths, is_train=True):
        img0 = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        img0 = cv2.resize(img0, (512, 256))
        imgs = [img0]
        for p in paths[1:]:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (512, 256))
            if is_train:
                img = colorADJ.process(img0/255.0, img/255.0)*255
                img = img.astype(np.uint8)
            imgs.append(img)
        inputs = []
        labels = []
        for img in imgs:
            _, w, _ = img.shape
            inp = img[:, :int(w/2), :]
            label = img[:, int(w/2):, :]
            if is_train:
                inp, label = random_transform_reso(inp, label)
            inputs.append(inp)
            labels.append(label)
        labels = self.to_tensor(np.array(labels, dtype=np.float32)/255)
        inputs = self.to_tensor(np.array(inputs, dtype=np.float32)/255)
        return inputs, labels

if __name__=='__main__':
    from reso_aug_model import RESOAUGNET
    net = RESOAUGNET()
    pd = PairData('pairs', 5)
    while True:
        x, y = pd.next_train()
        blocks = net(x)
        losses = net.loss(blocks, y)
        costs = [l.item() for l in losses]
        print(costs)
        # print(pd.tr_cur, x.shape, y.shape)
        # imgs = x.permute(0, 2, 3, 1).detach().numpy()*255
        # img = imgs.reshape([-1, 512, 3]).astype(np.uint8)
        # h, w, _ = img.shape
        # img = cv2.resize(img, (int(w/3), int(h/3)))
        # show(img, wait=200)
