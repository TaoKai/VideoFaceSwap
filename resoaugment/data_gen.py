import cv2
import os, sys
from cv_proc import blurFilter, show
sys.path.append('./face-alignment')
from MyFanPred import get_68_points
import numpy as np
from skimage import transform as trans
import random
from color_adjust import Color

color = Color()
landmarks_2D = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.726104,     0.780233  ], #54
], dtype=np.float32)

def getWarpFanPoints(points, left):
    points = list(points)
    warpPoints = points[17:49]+points[54:55]
    warpPoints = np.array(warpPoints, dtype=np.float32)+np.array(left, dtype=np.float32)
    return warpPoints

def alignFace(dst, frame):
    shp = np.array([256, 256], dtype=np.float32)
    src = landmarks_2D*shp+np.array([48, 32])
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:3,:]
    if M is None:
        return None, None
    else:
        warped = cv2.warpAffine(frame, M[:2], (256+96, 256+64), borderValue = 0.0)
        return warped, M

def getWarpFace(img):
    points = get_68_points(img)
    if points is None:
        return None
    points = getWarpFanPoints(points, (0,0))
    warp, _ = alignFace(points, img)
    return warp

def getPicPaths(basePath):
    dirs = [basePath+'/'+d for d in os.listdir(basePath)]
    paths = []
    for d in dirs:
        pics = [d+'/'+p for p in os.listdir(d)]
        random.shuffle(pics)
        pics = pics[:800]
        paths += pics
    random.shuffle(paths)
    return paths

def generate_data(start_id):
    # paths = getPicPaths('/home/inveno/taokai/workspace/TSAE_resnext/faces')
    paths = getPicPaths('E:/workspace/videoFaceExtract/faces')
    save_dir = 'pairs/'
    for i, p in enumerate(paths):
        if i<start_id:
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = getWarpFace(img)
        if img is None:
            continue
        orig = img.copy()
        ho, wo, _ = img.shape
        blur = blurFilter(img)
        h, w, _ = blur.shape
        h = h//4
        w = w//4
        blur = cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
        blur = cv2.resize(blur, (wo, ho), interpolation=cv2.INTER_LINEAR)
        blur = cv2.resize(blur, (256,256), interpolation=cv2.INTER_LINEAR)
        orig = cv2.resize(orig, (256,256), interpolation=cv2.INTER_LINEAR)
        img = np.concatenate([blur, orig], axis=1)
        cv2.imwrite(save_dir+str(100000000+i)+'.jpg', img)
        print(i, p)


if __name__=='__main__':
    generate_data(0)