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
from color_adjust import Color

colorADJ = Color()
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

def show(img, wait=0, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(wait)

def random_warp(image):
    shp = image.shape
    image = cv2.resize(image, (800, 800))
    num = 10
    rangeX = np.linspace(25, 800-25, num)
    mapx = np.broadcast_to(rangeX, (num, num))
    rangeY = np.linspace(25, 800-25, num)
    mapy = np.broadcast_to(rangeY, (num, num)).T
    mapx = mapx + np.random.normal(size=(num, num), scale=7)
    mapy = mapy + np.random.normal(size=(num, num), scale=7)
    interp_mapx = cv2.resize(mapx, (750, 750)).astype('float32')
    interp_mapy = cv2.resize(mapy, (750, 750)).astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    warped = image.copy()
    warped[25:800-25, 25:800-25, :] = warped_image
    warped = cv2.resize(warped, (shp[1], shp[0]))
    image = cv2.resize(image, (shp[1], shp[0]))
    return warped, image

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

def random_facepair_68(img, M):
    img = random_transform(img, **random_transform_args)
    warp, img = random_warp(img)
    warp = cv2.warpAffine(warp, M, (256,256), borderValue=0.0)
    img = cv2.warpAffine(img, M, (256,256), borderValue=0.0)
    return warp, img

def getFileList(path):
    files = os.listdir(path)
    files = [path+'/'+fp for fp in files]
    return files

def innerGen68(path, model):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    warpPoints = get_68_points(img, model)
    if warpPoints is None:
        return None
    warpPoints = list(warpPoints)
    points = warpPoints[17:49]+warpPoints[54:55]
    points = np.array(points, dtype=np.float32)
    src = landmarks_2D*np.array([204, 204])+np.array([26, 26])
    tform = trans.SimilarityTransform()
    tform.estimate(points, src)
    M = tform.params[0:3,:]
    if M is None:
        return None
    else:
        ml = list(M[:2].reshape(-1))
        str_ml = ''
        for m in ml:
            str_ml += str(m)+' '
        return str_ml.strip()

def generateAB68(pathA, pathB):
    Afiles = getFileList(pathA)
    Bfiles = getFileList(pathB)
    A_str = ''
    B_str = ''
    model = load_fan()
    for a in Afiles:
        m_str = innerGen68(a, model)
        if m_str is not None:
            A_str += a+' '+m_str+'\n'
            print(a, m_str)
    for b in Bfiles:
        m_str = innerGen68(b, model)
        if m_str is not None:
            B_str += b+' '+m_str+'\n'
            print(b, m_str)
    open('A_68.txt', 'w', 'utf-8').write(A_str.strip())
    open('B_68.txt', 'w', 'utf-8').write(B_str.strip())

class FaceData(object):
    def __init__(self, pathA, pathB, batch_size):
        self.batch_size = batch_size
        self.Alist = self.getPaths(pathA)
        self.Blist = self.getPaths(pathB)
        self.cur_A = 0
        self.cur_B = 0
        self.len_A = len(self.Alist)
        self.len_B = len(self.Blist)
        random.shuffle(self.Alist)
        random.shuffle(self.Blist)
    
    def getAffineMat(self, strs):
        mat = []
        strs = strs[1:]
        for s in strs:
            mat.append(float(s))
        mat = np.array(mat, dtype=np.float32).reshape(2, 3)
        return mat

    def getTestBatch(self):
        lst = self.Blist.copy()
        random.shuffle(lst)
        lst = lst[:64]
        imgs = []
        for l in lst:
            ls = l.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.warpAffine(img, M, (256,256), borderValue=0.0)
                imgs.append(img)
        imgs = np.array(imgs, dtype=np.uint8)
        return imgs

    def get_mean_color(self, pics):
        cnt = 0
        mean_color = np.zeros(3, dtype=np.float32)
        for pp in pics:
            ls = pp.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.warpAffine(img, M, (256,256), borderValue=0.0)
            img = img/255.0
            mean_color += img.mean(axis=(0, 1))
            cnt += 1
        return mean_color/cnt

    def getDoubleBatch(self, batchA, batchB):
        imgsA = []
        imgsB = []
        warpsA = []
        warpsB = []
        for fpA, fpB in zip(batchA, batchB):
            lsA = fpA.split(' ')
            pA = lsA[0]
            MA = self.getAffineMat(lsA)
            imgA = cv2.imread(pA, cv2.IMREAD_COLOR)
            lsB = fpB.split(' ')
            pB = lsB[0]
            MB = self.getAffineMat(lsB)
            imgB = cv2.imread(pB, cv2.IMREAD_COLOR)
            if imgA is None or imgB is None:
                continue
            warpA, imgA = random_facepair_68(imgA, MA)
            warpB, imgB = random_facepair_68(imgB, MB)
            raw_mask = np.ones(imgA.shape, dtype=np.float32)
            imgA = colorADJ.process(imgB/255.0, imgA/255.0, raw_mask)*255
            warpA = colorADJ.process(warpB/255.0, warpA/255.0, raw_mask)*255
            imgsA.append(imgA)
            imgsB.append(imgB)
            warpsA.append(warpA)
            warpsB.append(warpB)
        imgsA = np.array(imgsA, dtype=np.float32)/255
        warpsA = np.array(warpsA, dtype=np.float32)/255
        imgsB = np.array(imgsB, dtype=np.float32)/255
        warpsB = np.array(warpsB, dtype=np.float32)/255
        return warpsA, imgsA, warpsB, imgsB

    def getTrainBatch(self, batch):
        imgs = []
        warps = []
        for fp in batch:
            ls = fp.split(' ')
            p = ls[0]
            M = self.getAffineMat(ls)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            warp, img = random_facepair_68(img, M)
            imgs.append(img)
            warps.append(warp)
        imgs = np.array(imgs, dtype=np.float32)/255
        warps = np.array(warps, dtype=np.float32)/255
        return warps, imgs

    def to_tensor(self, mat):
        mat = torch.from_numpy(mat).float()
        mat = mat.permute(0, 3, 1, 2)
        return mat

    def next(self):
        if self.cur_A+self.batch_size<=self.len_A:
            batchA = self.Alist[self.cur_A:self.cur_A+self.batch_size]
            self.cur_A += self.batch_size
        else:
            self.cur_A = 0
            random.shuffle(self.Alist)
            batchA = self.Alist[self.cur_A:self.cur_A+self.batch_size]
            self.cur_A += self.batch_size
        if self.cur_B+self.batch_size<=self.len_B:
            batchB = self.Blist[self.cur_B:self.cur_B+self.batch_size]
            self.cur_B += self.batch_size
        else:
            self.cur_B = 0
            random.shuffle(self.Blist)
            batchB = self.Blist[self.cur_B:self.cur_B+self.batch_size]
            self.cur_B += self.batch_size
        warpsA, imgsA, warpsB, imgsB = self.getDoubleBatch(batchA, batchB)
        warpsA = warpsA.clip(0, 1)
        imgsA = imgsA.clip(0, 1)
        warpsB = warpsB.clip(0, 1)
        imgsB = imgsB.clip(0, 1)
        warpsA = self.to_tensor(warpsA)
        imgsA = self.to_tensor(imgsA)
        warpsB = self.to_tensor(warpsB)
        imgsB = self.to_tensor(imgsB)
        return warpsA, imgsA, warpsB, imgsB
    
    def getPaths(self, path):
        lines = open(path, 'r', 'utf-8').read().strip().split('\n')
        return lines

if __name__ == "__main__":
    # pathA = 'faces/angelababy'
    # pathB = 'faces/rongzuer'
    # generateAB68(pathA, pathB)
    face = FaceData('A_68.txt', 'B_68.txt', 4)
    while True:
        warpsA, imgsA, warpsB, imgsB = face.next()
        print(face.cur_A, face.cur_B, warpsA.shape, imgsA.shape, warpsB.shape, imgsB.shape)
        wa = warpsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)*255
        wa = wa.astype(np.uint8)
        ia = imgsA.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)*255
        ia = ia.astype(np.uint8)
        ia = cv2.resize(ia, (wa.shape[1], wa.shape[0]))
        wb = warpsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)*255
        wb = wb.astype(np.uint8)
        ib = imgsB.detach().numpy().transpose(0, 2, 3, 1).reshape(-1, 256, 3)*255
        ib = ib.astype(np.uint8)
        ib = cv2.resize(ib, (wa.shape[1], wa.shape[0]))
        out = np.concatenate([wa, ia, wb, ib], axis=1)
        cv2.imshow('out', out)
        cv2.waitKey(200)