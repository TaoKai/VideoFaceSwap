import cv2
import numpy as np
from SAE_clip_resnext import AutoEncoder
from TSegNet import TSegNet
from videoProc import saveVideo
from mtcnn import DetectFace
import torch
import os, sys
from dlib_util import get_det_pred, test_68points, show_68points
from skimage import transform as trans
from Kalman import KalmanObject
from color_adjust import Color, MatchHist
sys.path.append('./CosFace_pytorch')
from cosface_pred import get_img_feature, get_distance
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder()
model.load_state_dict(torch.load('faceTSAE_model_huangbo.pth', map_location=device))
model.eval()
seg_model = TSegNet()
seg_model.load_state_dict(torch.load('TSEG_model.pth', map_location=device))
seg_model.eval()
detectFace = DetectFace()
det68, pred68 = get_det_pred()
color_adj = Color()

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

def simple_draw2(points, frame):
    for pt in points:
        cv2.circle(frame, (pt[0], pt[1]), 3, (0,0,255), 2)
    cv2.imshow('bg', frame)
    cv2.waitKey(1)

def simple_draw():
    bg = np.ones([256, 256, 3], dtype=np.uint8)*255
    shp = bg.shape[:2][::-1]
    for pt in landmarks_2D:
        pt *= shp
        cv2.circle(bg, (pt[0], pt[1]), 3, (0,0,255), 2)
    cv2.imshow('bg', bg)
    cv2.waitKey(0)

def get_left_top(data):
    points = data[1][0]
    M_inv = np.linalg.inv(data[2][0])[:2]
    ones = np.ones(points.shape[0], dtype=np.float32).reshape(-1, 1)
    points = np.concatenate([points, ones], axis=1).T
    points = M_inv @ points
    points = points.min(axis=1)-np.array([13, 14], dtype=np.float32)
    lt = points.astype(np.int32)
    return lt

def sharp(image):
    kernel = np.array(
        [[0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]],
        dtype=np.float32
    )
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def getWarpPoints(dic, left):
    points = []
    for _, v in dic.items():
        points += v
    warpPoints = points[17:49]+points[54:55]
    warpPoints = np.array(warpPoints, dtype=np.float32)+np.array(left, dtype=np.float32)
    return warpPoints

def resort_frames(pics, name):
    dic = {}
    for p in pics:
        pid = p.split(name)[-1].split('.')[0]
        dic[int(pid)] = p
    id_list = list(dic.keys())
    id_list.sort()
    frames = []
    for i in id_list:
        frames.append(dic[i])
    return frames

def get_mean_color(img, mask):
    colors = img[mask>0.1].astype(np.float32)
    mean_color = colors.mean(axis=0)
    return mean_color

def get_hsv_vec(img, mask):
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = mask.reshape(mask.shape+(1,))
    img = img*mask/np.array([180, 255, 255])
    hsv_vec = img.reshape(-1, 3)
    return hsv_vec

def convert2targetHSV(imgA, maskA, imgB, maskB):
    hsvA = get_hsv_vec(imgA, maskA)
    hsvB = get_hsv_vec(imgB, maskB)
    meanA = hsvA.mean(axis=0)
    meanB = hsvB.mean(axis=0)
    hsvA = hsvA-meanA+meanB
    hsvA = hsvA.clip(0, 1)*np.array([180, 255, 255])
    hsvA = hsvA.reshape(imgA.shape).astype(np.uint8)
    hsvA = cv2.cvtColor(hsvA, cv2.COLOR_HSV2BGR)
    return hsvA

def convert_simgle_frame_face(box, frame, crop_size):
    shp = frame.shape
    b = box[:4].astype(np.int32)
    center = b.reshape(2, 2).mean(axis=0)
    lt = center-crop_size
    rb = center+crop_size
    max_l = max(frame.shape[0], frame.shape[1])
    lt = lt.clip(0, max_l).astype(np.int32)
    rb = rb.clip(0, max_l).astype(np.int32)
    img = frame[lt[1]:rb[1], lt[0]:rb[0], :]
    dic = test_68points(img, det68, pred68)
    if dic is None:
        return frame, None
    warpPoints = getWarpPoints(dic, lt)
    warp, M = alignFace(warpPoints, frame)
    warp_orig = np.array([warp], dtype=np.float32)/255
    warp_orig = torch.from_numpy(warp_orig).float().permute(0, 3, 1, 2)
    cut = model(warp_orig, 'A')
    warp_mask = seg_model(warp_orig)
    cut_mask = seg_model(cut)
    cut = cut[0].permute(1, 2, 0).detach().cpu().numpy()*255
    warp_mask = warp_mask[0].detach().cpu().numpy()
    cut_mask = cut_mask[0].detach().cpu().numpy()
    merge = color_adj.process(warp/255.0, cut/255.0, warp_mask)
    cut = merge*255.0
    cut = sharp(cut)
    cut = cut.clip(0, 255)
    cut = resize_img(cut, 0.07)
    M_inv = np.linalg.inv(M)[:2]
    cut_inv = cv2.warpAffine(cut, M_inv, (shp[1], shp[0]), borderValue=0.0)
    cut_mask_inv = cv2.warpAffine(cut_mask, M_inv, (shp[1], shp[0]), borderValue=0.0).reshape(shp[0], shp[1])
    warp_mask_inv = cv2.warpAffine(warp_mask, M_inv, (shp[1], shp[0]), borderValue=0.0).reshape(shp[0], shp[1])
    rect = curveRect2(M_inv, frame)
    merge_mask = cut_mask_inv*warp_mask_inv*rect
    blur, erode = blurMask(merge_mask)
    # merge_mask = np.concatenate([merge_mask, blur, erode], axis=1)
    merge_mask = blur
    blur = blur.reshape(blur.shape+(1,))
    frame = cut_inv*blur+frame*(1-blur)
    frame = frame.astype(np.uint8)
    return frame, merge_mask

def get_img_face_distances(img, faceFeatures):
    feat = get_img_feature(img)
    feature_dists = []
    for features in faceFeatures:
        dists = []
        for ft in features:
            d = get_distance(feat, ft)
            dists.append(d)
        feature_dists.append(np.array(dists).max())
    return feature_dists

def is_target(dists, target_id):
    ratio = 0.25
    dists = np.array(dists)
    tid = np.argmax(dists)
    prob = np.max(dists)
    if tid==target_id and prob>=ratio:
        return True
    return False

def test_more_points():
    _FEAT = get_random_face_features
    faceFeatures = [
        _FEAT('faces/zhouxingchi_film'),
        _FEAT('faces/zhangmanyu_film')
    ]
    targetID = 0
    path = 'videos/xingchimanyu_src.mp4'
    cap = cv2.VideoCapture(path)
    start = 0
    stop = start+99999
    cnt = 0
    frames = []
    crop_size = 200
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cnt>=start and cnt<=stop:
                cuts, data = detectFace(frame)
                merge_mask = None
                cuts = list(cuts)
                if len(cuts)<=0:
                    frames.append(frame)
                    continue
                hasTarget = False
                for i, cut in enumerate(cuts):
                    dists = get_img_face_distances(cut, faceFeatures)
                    if is_target(dists, targetID):
                        hasTarget = True
                        frame, merge_mask = convert_simgle_frame_face(data[0][i], frame, crop_size)
                        if merge_mask is not None:
                            cv2.imshow('img', merge_mask)
                            cv2.imshow('frame', frame.astype(np.uint8))
                            key = cv2.waitKey(1)
                            frames.append(frame)
                        break
                if not hasTarget:
                    frames.append(frame)
            elif cnt>stop:
                break
            print(cnt)
            cnt += 1
        else:
            break
    cap.release()
    saveVideo('tmp.mp4', frames)

def resize_img(img, ratio):
    shp = img.shape
    h = shp[0]
    w = shp[1]
    new_h = int(h*(1-ratio))
    new_w = int(w*(1-ratio))
    start_x = int(w*ratio/2)
    start_y = int(h*ratio/2)
    new_img = np.zeros(shp, dtype=np.uint8)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_img[start_y:start_y+new_h, start_x:start_x+new_w, :] = img
    return new_img

def blurMask(mask):
    mask[mask<0.1] = 0
    k_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    erode = cv2.erode(mask, kernel, iterations=6)
    g_size = 53
    blur = cv2.GaussianBlur(erode, (g_size, g_size), 0)
    return blur, erode

def curveRect2(M_inv, frame):
    k_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    rect = np.ones((168,176), dtype=np.float32)
    rect = cv2.warpAffine(rect, M_inv, (frame.shape[1], frame.shape[0]), borderValue=0.0)
    rect = cv2.erode(rect, kernel, iterations=8)
    return rect

def alignFace(dst, frame):
    shp = np.array([128, 128], dtype=np.float32)
    src = landmarks_2D*shp+np.array([24, 16])
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:3,:]
    if M is None:
        return None, None
    else:
        warped = cv2.warpAffine(frame, M[:2], (176, 168), borderValue = 0.0)
        return warped, M

def get_random_face_features(path, num=5):
    files = [path+'/'+p for p in os.listdir(path)]
    feats = []
    for i,p in enumerate(files):
        if i%num==0:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            cuts, _ = detectFace(img)
            if cuts.shape[0]>0:
                img = cuts[0]
                feat = get_img_feature(img)
                feats.append(feat)
    return feats

if __name__ == "__main__":
    test_more_points()