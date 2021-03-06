import torch
import torch.optim as optim
import torch.nn as nn
from SAE_clip_resnext import AutoEncoder
from pic_process import FaceData
from codecs import open
import numpy as np
import cv2
import os, sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def testGen(pics, model):
    orig = pics.copy()
    pics = torch.from_numpy(pics).float().to(device)
    pics = pics.permute(0, 3, 1, 2)/255.0
    outA = model(pics, 'A')
    outB = model(pics, 'B')
    outA = outA.permute(0, 2, 3, 1)*255
    outB = outB.permute(0, 2, 3, 1)*255
    outA = outA.detach().cpu().numpy().astype(np.uint8)
    outB = outB.detach().cpu().numpy().astype(np.uint8)
    img = np.concatenate([orig, outA, outB], axis=2)
    shp = img.shape
    img = img.reshape(shp[0]*shp[1], shp[2], shp[3])
    cv2.imwrite('test_out.jpg', img)
    print('save the test pics.')

def train(epoch, batch_size, mean_ratio):
    model_path = 'faceTSAE_model.pth'
    model = AutoEncoder()
    faceData = FaceData('A_68.txt', 'B_68.txt', batch_size)
    criterion = nn.L1Loss().to(device)
    model.to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderA.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderB.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    for i in range(epoch):
        model.train()
        mean_loss = 0
        jCnt = 0
        for j in range(faceData.len_A//batch_size+1):
            wa, ia, wb, ib = faceData.next()
            wa = wa.to(device)
            ia = ia.to(device)
            wb = wb.to(device)
            ib = ib.to(device)
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            outA = model(wa, 'A')
            outB = model(wb, 'B')
            loss1 = criterion(outA, ia)
            loss2 = criterion(outB, ib)
            loss1.backward()
            loss2.backward()
            optimizer_1.step()
            optimizer_2.step()
            print('epoch', i, 'step', j, 'loss A', loss1.item(), 'loss B', loss2.item())
            mean_loss += (loss1.item()+loss2.item())/2
            jCnt += 1
        mean_loss /= jCnt
        print('mean_loss', mean_loss)
        torch.save(model.state_dict(), model_path)
        model.eval()
        testPics = faceData.getTestBatch()
        testGen(testPics, model)
        if mean_loss <= mean_ratio:
            break
if __name__ == "__main__":
    train(600, 96, 0.019)
