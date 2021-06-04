from fan_util import load_fan, get_68_points, getMainFace, drawLandmarks, getWarpPoints, alignFace
import cv2
import numpy as np
from SAE_clip_resnext_dln import AutoEncoderUnet
import torch
import os, sys
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AEModel = AutoEncoderUnet()
AEModel.load_state_dict(torch.load('faceDLN_model.pth', map_location=device))
AEModel.eval()
FanModel = load_fan()

def getRandomFaces(baseDir='E:/workspace/videoFaceExtract/faces'):
    dirs = [baseDir+'/'+d for d in os.listdir(baseDir)]
    paths = []
    for d in dirs:
        pics = [d+'/'+p for p in os.listdir(d)]
        random.shuffle(pics)
        paths += pics[:5]
    random.shuffle(paths)
    return paths[:100]

def sharp(image):
    kernel = np.array(
        [[0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]],
        dtype=np.float32
    )
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def get_pics(path='asian'):
    pics = [path+'/'+p for p in os.listdir(path)]
    random.shuffle(pics)
    return pics

def multiply_points(pts, mat):
        pts = pts.astype(np.float32).transpose()
        ones = np.ones([1, pts.shape[1]], dtype=np.float32)
        pts = np.concatenate([pts, ones], axis=0)
        pts = mat @ pts
        return pts.astype(np.int32).transpose()

if __name__=='__main__':
    paths = get_pics()
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        points = get_68_points(img, FanModel)
        if points is None:
            continue
        dst = getWarpPoints(points)
        img, M = alignFace(dst, img)
        points = multiply_points(points, M)
        zmap = drawLandmarks(img, points)
        x = np.array([zmap], dtype=np.float32)/255
        x = torch.from_numpy(x).float().to(device)
        x = x.permute(0, 3, 1, 2)
        out = AEModel(x)
        out = out.permute(0, 2, 3, 1)*255
        out = out.detach().cpu().numpy().astype(np.uint8)[0]
        out = sharp(out)
        testImg = np.concatenate([img, zmap, out], axis=1)
        h = testImg.shape[0]
        w = testImg.shape[1]
        testImg = cv2.resize(testImg, (int(w/2), int(h/2)))
        cv2.imshow('', testImg)
        cv2.waitKey(1)
        cv2.imwrite('faces/'+str(i)+'.jpg', testImg)

