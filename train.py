from SAE_clip_resnext_dln import AutoEncoderUnet
from fan_util import FaceData
from Loss import Loss
import torch
import torch.optim as optim
import os, sys
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_batch(zmaps, imgs, model, save_id):
    model.eval()
    x = zmaps
    x = np.array(x, dtype=np.float32)/255
    x = torch.from_numpy(x).float().to(device)
    x = x.permute(0, 3, 1, 2)
    out = model(x)
    out = out.permute(0, 2, 3, 1)*255
    out = out.detach().cpu().numpy().astype(np.uint8)
    imgs = np.array(imgs, np.uint8)
    img = np.concatenate([imgs, out], axis=2)
    shp = img.shape
    img = img.reshape(shp[0]*shp[1], shp[2], shp[3])
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img, (int(w/2), int(h/2)))
    cv2.imwrite('test/test_out'+str(save_id)+'.jpg', img)

def train(epoch, batch_size, stop_ratio, save_step):
    model_path = 'faceDLN_model.pth'
    model = AutoEncoderUnet()
    model.to(device)
    faceData = FaceData('train_data.txt', batch_size)
    criterion = Loss().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))
    save_id = 10000
    for i in range(epoch):
        zmaps, imgs = faceData.get_test_batch()
        test_batch(zmaps, imgs, model, save_id)
        save_id += 1
        model.train()
        mean_loss = 0
        jCnt = 0
        for j in range(faceData.tr_len//batch_size+1):
            x, y = faceData.next()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            print('epoch', i, 'step', j, 'loss', loss.item())
            mean_loss += loss.item()
            jCnt += 1
            if jCnt%save_step==0:
                torch.save(model.state_dict(), model_path)
                mainImgs, zmaps, imgs = faceData.get_test_batch()
                test_batch(mainImgs, zmaps, imgs, model, save_id)
                save_id += 1
                model.train()
        mean_loss /= jCnt
        print('mean_loss', mean_loss)
        torch.save(model.state_dict(), model_path)
        if mean_loss <= stop_ratio:
            break

if __name__ == '__main__':
    train(20, 16, 0.028, 500)