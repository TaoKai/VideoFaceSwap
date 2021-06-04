import os, sys
import cv2
import face_alignment
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_fan():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    return fa

def get_68_points(img, model):
    points = model.get_landmarks(img)
    if points is None:
        return None
    else:
        return points[0]

def draw(img, points):
    for p in points:
        x, y = p.astype(np.int32)
        cv2.circle(img, (x, y), 3, (0,0,255), 1)
    return img

def markFrames(path, num=500):
    cap = cv2.VideoCapture(path)
    frames = []
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            points = fa.get_landmarks(frame)
            if points is not None:
                points = points[0]
            else:
                continue
            frame = draw(frame, points)
            if cnt<num:
                frames.append(frame)
                cnt += 1
                cv2.imshow('', frame)
                cv2.waitKey(1)
                print('add frame', cnt)
            else:
                break
    cap.release()
    return frames

def saveVideo(path, frames):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (frames[0].shape[1],frames[0].shape[0]))
    for f in frames:
        videoWriter.write(f)
    videoWriter.release()

if __name__=="__main__":
    videoPath = 'niangao.mp4'
    frames = markFrames(videoPath)
    saveVideo('tmp.mp4', frames)