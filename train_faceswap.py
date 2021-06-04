import torch
import torch.optim as optim
import torch.nn as nn
from SAE_clip_resnext_dln import AutoEncoder
from faceswap_data import FaceData
from codecs import open
import numpy as np
import cv2
import os, sys
from Loss import DSSIMLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reorder_test_pictures(img, h, col_num):
    old_h = img.shape[0]
    old_w = img.shape[1]
    block_num = (old_h//h)
    row_num = block_num//col_num+1
    new_img = np.ones([h*row_num, old_w*col_num, 3], dtype=np.uint8)*200
    for i in range(row_num):
        for j in range(col_num):
            img_block = img[(i*col_num+j)*h:(i*col_num+j+1)*h, :, :]
            if img_block.shape[0]<=0:
                break
            new_img[i*h:(i+1)*h, j*old_w:(j+1)*old_w, :] = img_block
    new_h, new_w, _ = new_img.shape
    new_img = cv2.resize(new_img, (int(new_w/2), int(new_h/2)), interpolation=cv2.INTER_AREA)
    return new_img

def testGen(pics, model, global_step, batch_size):
    def downscale_imgs(imgs, size=(256, 256)):
        new_imgs = []
        for img in imgs:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            new_imgs.append(img)
        return np.array(new_imgs, dtype=np.uint8)
    orig = pics.copy()
    orig = orig[:batch_size]
    pics = pics[:batch_size]
    pics = torch.from_numpy(pics).float().to(device)
    pics = pics.permute(0, 3, 1, 2)/255.0
    outA = model(pics, 'A')
    outB = model(pics, 'B')
    outA = outA.permute(0, 2, 3, 1)*255
    outB = outB.permute(0, 2, 3, 1)*255
    outA = outA.detach().cpu().numpy().astype(np.uint8)
    outB = outB.detach().cpu().numpy().astype(np.uint8)
    outA = downscale_imgs(outA)
    outB = downscale_imgs(outB)
    img = np.concatenate([orig, outA, outB], axis=2)
    shp = img.shape
    img = img.reshape(shp[0]*shp[1], shp[2], shp[3])
    img = reorder_test_pictures(img, 256, 4)
    cv2.imwrite('test/test_out'+str(10000000+global_step)+'.jpg', img)
    print('save the test pics.')

def train(epoch, batch_size, mean_ratio):
    model_path = 'faceTSAE_model.pth'
    model = AutoEncoder()
    faceData = FaceData('A_68.txt', 'B_68.txt', batch_size)
    criterion = DSSIMLoss().to(device)
    model.to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderA.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoderB.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    global_step = 0
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
            global_step += 1
            if global_step%500==0:
                torch.save(model.state_dict(), model_path)
                model.eval()
                testPics = faceData.getTestBatch()
                testGen(testPics, model, global_step, batch_size)
                model.train()
        mean_loss /= jCnt
        print('mean_loss', mean_loss)
        torch.save(model.state_dict(), model_path)
        model.eval()
        testPics = faceData.getTestBatch()
        testGen(testPics, model, global_step, batch_size)
        if mean_loss <= mean_ratio:
            break
if __name__ == "__main__":
    train(1500, 20, 0.019)

