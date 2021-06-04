from net import sphere
import torch
from lfw_eval import extractDeepFeature
import numpy as np
from PIL import Image
import os, sys, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cosface_model():
    model_path = 'CosFace_pytorch/cosface_model.pth'
    model = sphere().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_cosface_model()

def get_img_feature(img):
    with torch.no_grad():
        img = Image.fromarray(img)
        feat = extractDeepFeature(img, model, False)
    return feat

def get_distance(f1, f2):
    distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    return distance.item()

if __name__ == "__main__":
    model = load_cosface_model()
    img = np.random.randint(0, 255, [112, 96, 3]).astype(np.uint8)
    img2 = np.random.randint(0, 255, [112, 96, 3]).astype(np.uint8)
    feat = get_img_feature(img)
    feat2 = get_img_feature(img2)
    dist = get_distance(feat, feat2)
    print(dist)