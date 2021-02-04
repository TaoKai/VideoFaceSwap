import os, sys
from codecs import open
import requests
import cv2
import random
import numpy as np

def readURL(path):
    lines = open(path, 'r', 'utf-8').read().split('\r\n')
    return lines

def download_and_process(name):
    urls = readURL(name+'.txt')
    basePath = os.path.join('video', name)
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=8)
            if r.content is not None:
                f = open(os.path.join('video', 'tmp.mp4'), 'wb')
                f.write(r.content)
            print(name, i, 'writed.')
            path = os.path.join(basePath, str(i)+'.mp4')
            frames = readVideo('video/tmp.mp4')
            if frames is not None:
                saveVideo(path, frames)
        except:
            print(name, i, 'load failed.')

def selectEvenly(cap, num=32):
    fCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    interval = int(fCount/num-1)
    cnt = 0
    frames = []
    print('process', int(fCount), 'frames.')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if cnt == interval:
                frames.append(frame)
                cnt = 0
            cnt += 1
        else:
            break
    return frames[:num]

def cropFrames(frames, random_crop=True):
    l = 224
    h = frames[0].shape[0]
    w = frames[0].shape[1]
    rh = int((h-l)/2)
    rw = int((w-l)/2)
    if random_crop:
        rh = random.random()*(h-l)
        rw = random.random()*(w-l)
        rh = int(rh)
        rw = int(rw)
    cr_frames = []
    for f in frames:
        cf = f[rh:rh+l, rw:rw+l, :]
        cr_frames.append(cf)
    return cr_frames

def resizeFrames(frames, short=320):
    h = frames[0].shape[0]
    w = frames[0].shape[1]
    new_h = 0
    new_w = 0
    if h>w:
        new_w = short
        new_h = int(h*new_w/w)
    else:
        new_h = short
        new_w = int(w*new_h/h)
    re_frames = []
    for f in frames:
        rf = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        re_frames.append(rf)
    return re_frames

def saveVideo(path, frames):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (frames[0].shape[1],frames[0].shape[0]))
    for f in frames:
        videoWriter.write(f)
    videoWriter.release()

def readVideo(path):
    cap = cv2.VideoCapture(path)
    if cap is None:
        return None
    fCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fRate = cap.get(cv2.CAP_PROP_FPS)
    if fCount<=0:
        return None
    frames = selectEvenly(cap, num=64)
    frames = resizeFrames(frames, short=256)
    frames = cropFrames(frames, random_crop=False)
    cap.release()
    return frames

def selectEvenFrames(frames, num=64):
    fCount = len(frames)
    if fCount<=num:
        return frames
    interval = int(fCount/num)
    cnt = 0
    comp_frames = []
    print('process', int(fCount), 'frames.')
    while cnt<fCount:
        comp_frames.append(frames[cnt])
        cnt += interval
        if cnt==0:
            break
    return comp_frames[:num]

def compressFrames(frames):
    frames = selectEvenFrames(frames, num=64)
    frames = resizeFrames(frames, short=256)
    frames = cropFrames(frames, random_crop=False)
    return frames

def compressVideo(cap):
    fCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fRate = cap.get(cv2.CAP_PROP_FPS)
    if fCount<=0:
        return None
    frames = selectEvenly(cap, num=64)
    frames = resizeFrames(frames, short=256)
    frames = cropFrames(frames, random_crop=False)
    return frames

def mergeVideo():
    paths = ['videos/liming1.mp4', 'videos/liming2.mp4']
    frames = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
            print(len(frames))
        cap.release()
    saveVideo('videos/liming_src.mp4', frames)

def flipImages():
    path = 'faces/zhangmanyu_film'
    pics = [path+'/'+fp for fp in os.listdir(path)]
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1)
        sp = p.split('.')[0]+'_flip.jpg'
        cv2.imwrite(sp, img)
        print(sp)

def extractFrames():
    path = 'videos/xingchimanyu_src.mp4'
    cap = cv2.VideoCapture(path)
    start = 0
    total = 10000
    cnt = 0
    intv = 1
    name = path.split('/')[-1].split('.')[0]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cnt>=start+total:
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        if ret and cnt%intv==0 and cnt>start and cnt<start+total:
            cv2.imwrite('frames/'+name+str(cnt)+'.jpg', frame)
            print(name, cnt-start+1)
        cnt += 1
    cap.release()

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

def gen_by_frames():
    path = 'frames/'
    name = 'dilireba_src'
    pics = [path+p for p in os.listdir(path)]
    pics = resort_frames(pics, name)
    frames = []
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        frames.append(img)
        print(p)
    saveVideo(name+'_src.mp4', frames)

if __name__ == "__main__":
    extractFrames()