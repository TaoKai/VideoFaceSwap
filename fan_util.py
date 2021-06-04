import cv2
import os, sys
import numpy as np
sys.path.append('./face-alignment')
from MyFanPred import get_68_points, draw, load_fan
from scipy.spatial import Delaunay
from skimage import transform as trans
from codecs import open
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    is_flip = False
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # if np.random.random() < random_flip:
    #     result = result[:, ::-1]
    return result, mat

def getWarpPoints(points, left=(0, 0)):
    points = list(points)
    warpPoints = points[17:49]+points[54:55]
    warpPoints = np.array(warpPoints, dtype=np.float32)+np.array(left, dtype=np.float32)
    return warpPoints

def alignFace(dst, img):
    shp = np.array([204, 204], dtype=np.float32)
    src = landmarks_2D*shp+np.array([26, 26])
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:3,:]
    if M is None:
        return None, None, None
    else:
        img = cv2.warpAffine(img, M[:2], (256, 256), borderValue = 0.0)
        return img, M[:2]

def drawLandmarks(img, points):
    shp = img.shape
    dln = Delaunay(points)
    triangles = dln.simplices
    zmap = np.zeros(shp, dtype=np.uint8)
    # face = points[:17]
    # lbrow = points[17:22]
    # rbrow = points[22:27]
    # nose = points[27:36]
    # leye = points[36:42]
    # reye = points[42:48]
    # omouth = points[48:60]
    # imouth = points[60:68]
    # pts = [face, lbrow, rbrow, nose, leye, reye, omouth, imouth]
    # for pt in pts:
    #     pt = pt.astype(np.int32)
    #     cv2.polylines(zmap, [pt], True, (255, 255, 255), 2)
    for t in triangles:
        tp = points[t, :]
        tp = tp.astype(np.int32)
        cv2.fillPoly(zmap, [tp], (125, 125, 125))
        cv2.polylines(zmap, [tp], True, (255,255,255), 2)
    return zmap

def getMainFace(path, model):
    p = path
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    points = get_68_points(img, model)
    dst = getWarpPoints(points)
    zmap = drawLandmarks(img, points)
    mainImg, _, _ = alignFace(dst, img, zmap)
    return mainImg

def getPicList(path):
    pics = [path+'/'+p for p in os.listdir(path)]
    return pics

def processSingleMain(path):
    model = load_fan()
    mp = path+'/main.jpg'
    img = getMainFace(mp, model)
    cv2.imwrite(mp, img)
    print('write', mp)

def processMainFaces(path):
    dirs = [path+'/'+d for d in os.listdir(path)]
    for d in dirs:
        processSingleMain(d)

def buildSingleData(baseDir):
    model = load_fan()
    train_str = ''
    paths = [baseDir+'/'+p for p in os.listdir(baseDir)]
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        points = get_68_points(img, model)
        if points is None:
                continue
        px = points[:,0]
        py = points[:,1]
        x_str = ''
        y_str = ''
        for i in range(points.shape[0]):
            x_str += str(int(px[i]))+' '
            y_str += str(int(py[i]))+' '
        x_str = x_str.strip()
        y_str = y_str.strip()
        p_str = p+' '+x_str+' '+y_str
        train_str += p_str+'\n'
        print(p)
    open('train_data.txt', 'w', 'utf-8').write(train_str.strip())

def buildTrainData(baseDir):
    model = load_fan()
    dirs = [baseDir+'/'+d for d in os.listdir(baseDir)]
    train_str = ''
    for d in dirs:
        mainPic = d+'/main.jpg'
        pics = getPicList(d)
        for p in pics:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            points = get_68_points(img, model)
            if points is None:
                continue
            px = points[:,0]
            py = points[:,1]
            x_str = ''
            y_str = ''
            for i in range(points.shape[0]):
                x_str += str(int(px[i]))+' '
                y_str += str(int(py[i]))+' '
            x_str = x_str.strip()
            y_str = y_str.strip()
            p_str = p+' '+mainPic+' '+x_str+' '+y_str
            train_str += p_str+'\n'
            print(p)
    open('train_data.txt', 'w', 'utf-8').write(train_str.strip())

class FaceData():
    def __init__(self, train_path, batch_size=16):
        self.train_path = train_path
        self.batch_size = batch_size
        self.line_records = self.get_path_list()
        self.tr_len = len(self.line_records)
        self.tr_cur = 0
        self.train_indices = [i for i in range(self.tr_len)]
        random.shuffle(self.train_indices)

    def get_path_list(self):
        lines = open(self.train_path, 'r', 'utf-8').read().strip().split('\n')
        line_records = []
        for i, l in enumerate(lines):
            lsp = l.split(' ')
            p = lsp[0]
            points = lsp[1:]
            xp = points[:68]
            yp = points[68:]
            xp = np.array(xp, np.int32)
            yp = np.array(yp, np.int32)
            points = np.array([xp, yp], dtype=np.int32).T
            line_records.append((p, points))
            print('add record', i, end='\r')
        print('\n')
        return line_records

    def multiply_points(self, pts, mat):
        pts = pts.astype(np.float32).transpose()
        ones = np.ones([1, pts.shape[1]], dtype=np.float32)
        pts = np.concatenate([pts, ones], axis=0)
        pts = mat @ pts
        return pts.astype(np.int32).transpose()

    def get_train_batch(self, batch):
        zmaps = []
        targetImgs = []
        for i in batch:
            p, pts = self.line_records[i]
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img, r_mat = random_transform(img, **random_transform_args)
            dst = getWarpPoints(pts)
            img, M = alignFace(dst, img)
            pts = self.multiply_points(pts, r_mat)
            pts = self.multiply_points(pts, M)
            zmap = drawLandmarks(img, pts)
            if np.random.random()<0.5:
                zmap = zmap[:, ::-1]
                img = img[:, ::-1]
            zmaps.append(zmap)
            targetImgs.append(img)
        zmaps = np.array(zmaps, dtype=np.float32)/255
        targetImgs = np.array(targetImgs, dtype=np.float32)/255
        return zmaps, targetImgs
    
    def next(self):
        if self.tr_cur+self.batch_size<=self.tr_len:
            batch = self.train_indices[self.tr_cur:self.tr_cur+self.batch_size]
            self.tr_cur += self.batch_size
        else:
            self.tr_cur = 0
            random.shuffle(self.train_indices)
            batch = self.train_indices[self.tr_cur:self.tr_cur+self.batch_size]
            self.tr_cur += self.batch_size
        mainImgs, targetImgs = self.get_train_batch(batch)
        mainImgs = self.to_tensor(mainImgs)
        targetImgs = self.to_tensor(targetImgs)
        return mainImgs, targetImgs
    
    def to_tensor(self, mat):
        mat = torch.from_numpy(mat).float().to(device)
        mat = mat.permute(0, 3, 1, 2)
        return mat

    def get_test_batch(self):
        indices = self.train_indices.copy()
        random.shuffle(indices)
        indices = indices[:self.batch_size]
        zmaps = []
        imgs = []
        for i in indices:
            p, pts = self.line_records[i]
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            dst = getWarpPoints(pts)
            img, M = alignFace(dst, img)
            pts = self.multiply_points(pts, M)
            zmap = drawLandmarks(img, pts)
            zmaps.append(zmap)
            imgs.append(img)
        return zmaps, imgs
        

if __name__=='__main__':
    print('fan test start.')
    buildSingleData('faces/jiangshuyingnew')
    fd = FaceData('train_data.txt', batch_size=4)
    while True:
        zmaps, imgs = fd.next()
        zmaps = zmaps.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)
        imgs = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)
        img = np.concatenate([zmaps, imgs], axis=1)
        cv2.imshow('', img)
        cv2.waitKey(200)