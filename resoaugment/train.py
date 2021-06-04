from pair_data import PairData
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import random
import cv2
from reso_aug_model import RESOAUGNET, MyDiscriminator
from next_resoaug import SAE_RESNEXT_ENCODER
from Losses import MyLogisticGAN
import os, sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta_1 = 0.5
beta_2 = 0.99
eps = 1e-8
lr = 5e-5

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def train(epoch, batch_size):
    data = PairData('pairs', batch_size)
    generator = SAE_RESNEXT_ENCODER()
    generator.to(device)
    generator_path = 'best_generator.pth'
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device))
    # discriminator = MyDiscriminator()
    # discriminator.to(device)
    # discriminator_path = 'best_discriminator.pth'
    # if os.path.exists(discriminator_path):
    #     discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    # criterion = MyLogisticGAN(discriminator)
    common_loss = generator.loss
    gen_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps)
    # dis_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps)
    global_cnt = 0
    for i in range(epoch):
        # discriminator.train()
        generator.train()
        for j in range(data.tr_len//batch_size+1):
            x, y = data.next_train()
            x = x.to(device)
            y = y.to(device)
            # with torch.no_grad():
            #     fakes = generator(x)
            # dis_loss = criterion.dis_loss(y, fakes)
            # dis_optim.zero_grad()
            # dis_loss.backward()
            # dis_optim.step()
            preds = generator(x)
            # gen_loss = criterion.gen_loss(y, preds)
            c_loss = common_loss(preds, y)
            gen_loss = c_loss
            gen_optim.zero_grad()
            gen_loss.backward()
            # nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.)
            gen_optim.step()
            print('epoch', i, 'step', j, 'gen_loss', gen_loss.item())
            global_cnt += 1
            if global_cnt%500 == 0:
                generator.eval()
                x, y = data.next_test()
                x = x.to(device)
                y = y.to(device)
                pred_y = generator(x)
                pred_y = pred_y.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
                x = x.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
                y = y.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
                img = np.concatenate([x, y, pred_y], axis=1).astype(np.uint8)
                cv2.imwrite('gan_test/'+str(10000000+global_cnt)+'.jpg', img)
                torch.save(generator.state_dict(), generator_path)
                # torch.save(discriminator.state_dict(), discriminator_path)
                generator.train()
        torch.save(generator.state_dict(), generator_path)
        # torch.save(discriminator.state_dict(), discriminator_path)
        generator.eval()
        x, y = data.next_test()
        x = x.to(device)
        y = y.to(device)
        pred_y = generator(x)
        pred_y = pred_y.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
        x = x.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
        y = y.permute(0,2,3,1).reshape([-1, 256, 3]).detach().cpu().numpy()*255
        img = np.concatenate([x, y, pred_y], axis=1).astype(np.uint8)
        cv2.imwrite('gan_test/'+str(10000000+global_cnt)+'.jpg', img)

train(100, 24)
