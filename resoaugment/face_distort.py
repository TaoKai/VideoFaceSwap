import random
from skimage import transform as trans
import numpy as np
import cv2

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}
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

def random_warp(image):
    shp = image.shape
    if shp != (256,256,3):
        image = cv2.resize(image, (256, 256))
    num = 10
    rangeX = np.linspace(8, 256-8, num)
    mapx = np.broadcast_to(rangeX, (num, num))
    rangeY = np.linspace(8, 256-8, num)
    mapy = np.broadcast_to(rangeY, (num, num)).T
    mapx = mapx + np.random.normal(size=(num, num), scale=3)
    mapy = mapy + np.random.normal(size=(num, num), scale=3)
    interp_mapx = cv2.resize(mapx, (240, 240)).astype('float32')
    interp_mapy = cv2.resize(mapy, (240, 240)).astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    warped = image.copy()
    warped[8:256-8, 8:256-8, :] = warped_image
    warped = cv2.resize(warped, (shp[1], shp[0]))
    image = cv2.resize(image, (shp[1], shp[0]))
    return warped, image

def random_warp_legacy(image):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64, 64))
    return warped_image, target_image

def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

def faceAlign(img, points):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(points, dtype=np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2,:]

    shp = [112, 96]
    if M is None:
        return None
    else:
        warped = cv2.warpAffine(img, M, (shp[1],shp[0]), borderValue = 0.0)
        return warped

# random_transform_args = {
#     'rotation_range': 10,
#     'zoom_range': 0.05,
#     'shift_range': 0.05,
#     'random_flip': 0.5,
# }
def random_transform_reso(img0, img1):
    h, w = img0.shape[0:2]
    rotation = np.random.uniform(-10, 10)
    scale = np.random.uniform(1 - 0.05, 1 + 0.05)
    tx = np.random.uniform(-0.05, 0.05) * w
    ty = np.random.uniform(-0.05, 0.05) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result0 = cv2.warpAffine(img0, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    result1 = cv2.warpAffine(img1, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < 0.5:
        result0 = result0[:, ::-1]
        result1 = result1[:, ::-1]
    return result0, result1

def random_facepair_68(img, M):
    img = random_transform(img, **random_transform_args)
    warp, img = random_warp(img)
    warp = cv2.warpAffine(warp, M, (352,320), borderValue=0.0)
    img = cv2.warpAffine(img, M, (352,320), borderValue=0.0)
    # if random.random()>0.7:
    #     warp = cv2.flip(warp, 1)
    #     img = cv2.flip(img, 1)
    return warp, img

def random_facepair_crop(img, x, y):
    warp, img = random_warp(img)
    warp = warp[y:y+64, x:x+64, :]
    img = img[y:y+64, x:x+64, :]
    if random.random()>0.7:
        warp = cv2.flip(warp, 1)
        img = cv2.flip(img, 1)
    return warp, img

def random_facepair(img):
    img = random_transform(img, **random_transform_args)
    warp, img = random_warp(img)
    # warp = None
    # if random.random()>0.5:
    #     warp, img = random_warp(img)
    # else:
    #     warp = img.copy()
    # if random.random()>0.5:
    #     warp = cv2.flip(warp, 1)
    #     img = cv2.flip(img, 1)
    return warp, img
    

if __name__=='__main__':
    path = 'faces/andy/andy_192.jpg'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    while True:
        warp, img = random_warp(img)
        warp = random_transform(warp, **random_transform_args)
        cv2.imshow('out', warp)
        cv2.waitKey(500)
