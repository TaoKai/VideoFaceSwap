import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Loss_legacy import DSSIMLoss
import numpy as np
from Blocks import DiscriminatorBlock
from CustomLayers import EqualizedLinear, EqualizedConv2d, View
from Losses import MyLogisticGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _RESOBLOCK(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(_RESOBLOCK, self).__init__()
        self.conv0 = nn.Conv2d(in_chan, out_chan, 1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3)
        self.conv2 = nn.Conv2d(in_chan, out_chan, 5)
        self.conv3 = nn.Conv2d(in_chan, out_chan, 7)
    
    def forward(self, x):
        shp = x.shape
        x0 = F.upsample(self.conv0(x), size=shp[2:], mode='bilinear')
        x1 = F.upsample(self.conv1(x), size=shp[2:], mode='bilinear')
        x2 = F.upsample(self.conv2(x), size=shp[2:], mode='bilinear')
        x3 = F.upsample(self.conv3(x), size=shp[2:], mode='bilinear')
        x_comb = (x0+x1+x2+x3)/1.0
        return x_comb

class _ResRESOBLOCK(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(_ResRESOBLOCK, self).__init__()
        self.resoblock0 = _RESOBLOCK(in_chan, out_chan)
        self.resoblock1 = _RESOBLOCK(out_chan, out_chan)
        self.leak0 = nn.LeakyReLU(0.1, inplace=False)
        self.leak2 = nn.LeakyReLU(0.1, inplace=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.conv = nn.Conv2d(in_chan, out_chan, 1)
    
    def forward(self, x):
        identity = self.conv(x)
        x = self.resoblock0(x)
        x = self.leak0(x)
        x = self.resoblock1(x)
        x = x+identity
        x = self.leak2(x)
        x = self.bn(x)
        return x

class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        act, gain = {'relu': (torch.relu, np.sqrt(2)), 'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}['lrelu']
        self.DisBlock0 = DiscriminatorBlock(3, 128, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.DisBlock1 = DiscriminatorBlock(128, 128, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.DisBlock2 = DiscriminatorBlock(128, 128, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.DisBlock3 = DiscriminatorBlock(128, 128, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.DisBlock4 = DiscriminatorBlock(128, 128, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.DisBlock5 = DiscriminatorBlock(128, 64, gain=gain, use_wscale=True, activation_layer=act, blur_kernel=[1, 2, 1])
        self.view = View(-1)
        self.linear0 = EqualizedLinear(1024, 1024, gain=gain, use_wscale=True)
        self.linear1 = EqualizedLinear(1024, 1024, gain=gain, use_wscale=True)
        self.act = act

    def forward(self, x):
        x = self.DisBlock0(x)
        x = self.DisBlock1(x)
        x = self.DisBlock2(x)
        x = self.DisBlock3(x)
        x = self.DisBlock4(x)
        x = self.DisBlock5(x)
        x = self.view(x)
        x = self.act(self.linear0(x))
        x = self.act(self.linear1(x))
        return x

class RESOAUGNET(nn.Module):
    def __init__(self):
        super(RESOAUGNET, self).__init__()
        self.resreso0 = _ResRESOBLOCK(3, 128)
        self.activ = nn.Sigmoid()
        self.dssim = DSSIMLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.conv = nn.Conv2d(128, 3, 1)
        self.noise = torch.randn([1, 3, 256, 256]).float().to(device)
        self.noise = self.noise*0.2
        self.noise.requires_grad = True
    
    def forward(self, x):
        x = x+self.noise
        x = self.resreso0(x)
        x = self.conv(x)
        x = self.activ(x)
        return x
    
    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.reshape([b, ch, w * h])
        # features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def loss(self, x, y):
        loss_d = self.dssim(x, y)
        loss_m = self.l1(x, y)
        g_x = self.gram_matrix(x)
        g_y = self.gram_matrix(y)
        loss_g = self.mse(g_x, g_y)
        loss = (loss_d+loss_m)/2.0+loss_g*100.0/2.0
        return loss

def load(path):
    model = RESOAUGNET().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def predict_one(img, model):
    img = img/255.0
    img = np.array([img], dtype=np.float32)
    img = img.transpose(0, 3, 1, 2)
    img = torch.from_numpy(img).float().to(device)
    pred = model(img)[0].permute(1,2,0).cpu().detach().numpy()*255.0
    pred = pred.astype(np.uint8)
    return pred

if __name__=='__main__':
    '''
    cfg.model.g_optim.learning_rate = 0.003
    cfg.model.g_optim.beta_1 = 0
    cfg.model.g_optim.beta_2 = 0.99
    cfg.model.g_optim.eps = 1e-8
    '''
    x = torch.randint(0, 255, (5,3,256,256)).float()/255
    x = x.to(device)
    dis = MyDiscriminator()
    gen = RESOAUGNET()
    GanLoss = MyLogisticGAN(dis)
    pred = gen(x)
    gen_loss = GanLoss.gen_loss(x, pred)
    pred = pred.detach()
    dis_loss = GanLoss.dis_loss(x, pred)
    print(gen_loss, dis_loss)