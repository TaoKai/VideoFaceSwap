import cv2
import numpy as np
import torch
import os, sys
sys.path.append('./face-alignment')
from MyFanPred import get_68_points as get_68_fan_points
from MyFanPred import load_fan
from mtcnn import DetectFace
from skimage import transform as trans
from color_adjust import Color, MatchHist
from fan_util import drawLandmarks
from SAE_clip_resnext_dln import AutoEncoder
sys.path.append('./resoaugment')
from next_resoaug import SAE_RESNEXT_ENCODER as RESOAUGMODEL
sys.path.append('./CosFace_pytorch')
from cosface_pred import get_img_feature, get_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fan = load_fan()
color_adj = Color()
detectFace = DetectFace()
AEModel = AutoEncoder()
AEModel.load_state_dict(torch.load('models/angelababy_libingbing.pth', map_location=device))
AEModel.eval()
resoModel = RESOAUGMODEL()
resoModel.load_state_dict(torch.load('resoaugment/best_generator.pth', map_location=device))
resoModel.eval()

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

def show(img, wait=0, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(wait)

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
        return img, M

def multiply_points(pts, mat):
    pts = pts.astype(np.float32).transpose()
    ones = np.ones([1, pts.shape[1]], dtype=np.float32)
    pts = np.concatenate([pts, ones], axis=0)
    pts = mat @ pts
    return pts.astype(np.int32).transpose()

def getWarpFanPoints(points, left):
    points = list(points)
    warpPoints = points[17:49]+points[54:55]
    warpPoints = np.array(warpPoints, dtype=np.float32)+np.array(left, dtype=np.float32)
    points = np.array(points, dtype=np.float32)+np.array(left, dtype=np.float32)
    return warpPoints, points

def getMask(points):
    mask_pts = list(points.copy())
    cheek = mask_pts[:17]
    cheek.reverse()
    brow = mask_pts[17:20]+mask_pts[24:27]
    for b in brow:
        b[1] -= 27
    face = cheek+brow
    face = np.array(face, dtype=np.int32)
    shp = (256, 256, 3)
    mask = np.zeros(shp, dtype=np.uint8)
    cv2.fillPoly(mask, [face], (255, 255, 255))
    return mask/255.0

def resize_img(img, ratio):
    shp = img.shape
    h = shp[0]
    w = shp[1]
    new_h = int(h*(1-ratio))
    new_w = int(w*(1-ratio))
    start_x = int(w*ratio/2)
    start_y = int(h*ratio/2)
    new_img = np.zeros(shp, dtype=np.float32)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(shp)==2:
        new_img[start_y:start_y+new_h, start_x:start_x+new_w] = img
    else:
        new_img[start_y:start_y+new_h, start_x:start_x+new_w, :] = img
    return new_img

def blurMask(mask, iters):
    mask[mask<0.1] = 0
    k_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blur = cv2.erode(mask, kernel, iterations=iters)
    g_size = 83
    blur = cv2.GaussianBlur(blur, (g_size, g_size), 0)
    return blur

def getOval(shape):
    h, w, _ = shape
    oval = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
    oval = np.array([oval, oval, oval], dtype=np.float32).transpose([1, 2, 0])
    return oval

def get_blur_mask(warp_mask, iters):
    h, w, _ = warp_mask.shape
    rect_mask = np.ones((h, w, 3), dtype=np.float32)
    rect_mask = resize_img(rect_mask, 0.14)
    oval_mask = getOval(warp_mask.shape)
    oval_mask = resize_img(oval_mask, 0.05)
    mask = warp_mask*rect_mask*oval_mask
    big_img = np.zeros((h*3, w*3, 3), dtype=np.float32)
    big_img[h:2*h, w:2*w, :] = mask
    big_img = blurMask(big_img, iters)
    mask = big_img[h:2*h, w:2*w, :]
    mask = cv2.resize(mask, (w, h))
    return mask

def getWarpImage(img, M, shape):
    h, w, _ = shape
    frame = cv2.warpAffine(img, M, (w, h), borderValue=0.0)
    return frame

def get_random_face_features(path, num=3):
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
    ratio = 0.50
    dists = np.array(dists)
    tid = np.argmax(dists)
    prob = np.max(dists)
    if tid==target_id and prob>=ratio:
        return True
    return False

def convert_one_face(box, frame, crop_size):
    b = box[:4].astype(np.int32)
    center = b.reshape(2, 2).mean(axis=0)
    lt = center-crop_size
    rb = center+crop_size
    max_l = max(frame.shape[0], frame.shape[1])
    lt = lt.clip(0, max_l).astype(np.int32)
    rb = rb.clip(0, max_l).astype(np.int32)
    img = frame[lt[1]:rb[1], lt[0]:rb[0], :]
    dic = get_68_fan_points(img, fan)
    if dic is None:
        return frame
    warpPoints, points = getWarpFanPoints(dic, lt)
    warp, M = alignFace(warpPoints, frame)
    points = multiply_points(points, M[:2])
    raw_mask = getMask(points)
    warp_torch = np.array([warp], dtype=np.float32)/255
    warp_torch = torch.from_numpy(warp_torch).float().permute(0, 3, 1, 2)
    warp_torch = AEModel(warp_torch, 'A')
    warp_torch = warp_torch[0].permute(1, 2, 0).detach().cpu().numpy()*255
    warp_torch = cv2.resize(warp_torch, (256,256))
    warp_torch = np.array([warp_torch], dtype=np.float32)/255
    warp_torch = torch.from_numpy(warp_torch).float().permute(0, 3, 1, 2)
    warp_torch = resoModel(warp_torch)
    warp_torch = warp_torch[0].permute(1, 2, 0).detach().cpu().numpy()*255
    warp_torch = cv2.resize(warp_torch, (256,256))
    warp_torch = color_adj.process(warp/255, warp_torch/255, raw_mask)*255
    warp_torch = warp_torch.astype(np.uint8)
    warp_torch = resize_img(warp_torch, 0.03)
    blur_mask = get_blur_mask(raw_mask, 7)
    M_inv = np.linalg.inv(M)[:2]
    torch_inv = getWarpImage(warp_torch, M_inv, frame.shape)
    blur_inv = getWarpImage(blur_mask, M_inv, frame.shape)
    img = torch_inv*blur_inv+frame*(1-blur_inv)
    img = img.astype(np.uint8)
    return img

def saveVideo(path, frames, frate):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frate, (frames[0].shape[1],frames[0].shape[0]))
    for f in frames:
        videoWriter.write(f)
    videoWriter.release()

def concat_frame(f1, f2, axis=0):
    frame = np.concatenate([f1, f2], axis=axis)
    return frame

def test_pics(path):
    pics = [path+'/'+p for p in os.listdir(path)]
    for i, p in enumerate(pics):
        orig_img = cv2.imread(p, cv2.IMREAD_COLOR)
        cuts, data = detectFace(orig_img)
        face_img = orig_img.copy()
        if cuts.shape[0]>0:
            for j, b in enumerate(data[0]):
                face_img = convert_one_face(b, face_img, 150)
            img = np.concatenate([face_img, orig_img], axis=0)
            cv2.imwrite('test_faces/'+str(i)+'_'+str(j)+'.jpg', img)

def test_video(path):
    _FEAT = get_random_face_features
    targetID = 0
    faceFeatures = [
        _FEAT('faces/libingbing'),
    ]
    crop_size = 200
    cap = cv2.VideoCapture(path)
    frate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            origin = frame.copy()
            print(cnt, 'frame read.')
            cnt += 1
            cuts, data = detectFace(frame)
            if cuts.shape[0]<=0:
                frame = concat_frame(frame, origin, 0)
                h, w, _ = frame.shape
                frame = cv2.resize(frame, (int(w/2),int(h/2)))
                frames.append(frame)
                continue
            for i in range(cuts.shape[0]):
                dists = get_img_face_distances(cuts[i], faceFeatures)
                if is_target(dists, targetID):
                    box = data[0][i]
                    frame = convert_one_face(box, frame, crop_size)
            frame = concat_frame(frame, origin, 0)
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (int(w/2),int(h/2)))
            frames.append(frame)
            show(frame, 1)
        else:
            break
    cap.release()
    saveVideo('tmp.mp4', frames, frate)

test_video('test_faces/libingbing_cut.mp4')
# test_pics('qimao')
