import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Loss_legacy import DSSIMLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _PixelShuffler(nn.Module):
    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)  # channel first
        return out

class _UpScale(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', nn.Conv2d(input_features, output_features * 4, kernel_size=3, padding=1, padding_mode='reflect'))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())

class SAE_RESNEXT_ENCODER(nn.Module):
    def __init__(self):
        super(SAE_RESNEXT_ENCODER, self).__init__()
        self.resnext50 = models.resnext50_32x4d(pretrained=True)
        self.conv1 = self.resnext50.conv1
        self.bn1 = self.resnext50.bn1
        self.relu = self.resnext50.relu
        self.maxpool = self.resnext50.maxpool
        self.L1 = self.resnext50.layer1
        self.L2 = self.resnext50.layer2
        self.L3 = self.resnext50.layer3
        self.L4 = self.resnext50.layer4
        # self.fc1 = nn.Linear(24576, 1024*4*3)
        # self.fc2 = nn.Linear(1024*4*3, 1024*4*3)
        self.fc_conv = nn.Conv2d(2048, 1024, 1)
        self.fc2_conv = nn.Conv2d(1024, 1024, 1)
        self.up50 = _UpScale(1024, 512)
        self.up51 = _UpScale(512, 256)
        self.up52 = _UpScale(256, 128)
        self.up53 = _UpScale(128, 64)
        self.up54 = _UpScale(64, 32)
        self.up10 = _UpScale(256, 128)
        self.up11 = _UpScale(128, 32)
        self.up30 = _UpScale(1024, 512)
        self.up31 = _UpScale(512, 256)
        self.up32 = _UpScale(256, 128)
        self.up33 = _UpScale(128, 32)
        self.relu = nn.LeakyReLU(0.2)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.top_conv = nn.Conv2d(32, 3, 1)
        self.activ = nn.Sigmoid()
        self.bn = nn.InstanceNorm2d(32)
        self.dssim = DSSIMLoss()
        self.noisex = torch.randn([1, 64, 64, 64]).float().to(device)
        self.noisex = self.noisex*0.05
        self.noisex.requires_grad = True
        self.noise1 = torch.randn([1, 256, 64, 64]).float().to(device)
        self.noise1 = self.noise1*0.05
        self.noise1.requires_grad = True
        self.noise3 = torch.randn([1, 1024, 16, 16]).float().to(device)
        self.noise3 = self.noise3*0.05
        self.noise3.requires_grad = True
        self.noise5 = torch.randn([1, 1024, 8, 8]).float().to(device)
        self.noise5 = self.noise5*0.05
        self.noise5.requires_grad = True
        self.insnx = nn.InstanceNorm2d(64)
        self.insn1 = nn.InstanceNorm2d(256)
        self.insn3 = nn.InstanceNorm2d(1024)
        self.insn5 = nn.InstanceNorm2d(1024)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.insnx(x+self.noisex)
        l1_out_x = self.L1(x)
        l1_out_x = self.insn1(l1_out_x+self.noise1)
        l2_out_x = self.L2(l1_out_x)
        l3_out_x = self.L3(l2_out_x)
        l3_out_x = self.insn3(l3_out_x+self.noise3)
        l4_out_x = self.L4(l3_out_x)
        # l4_out = l4_out.flatten(start_dim=1)
        # fc1_out = self.fc1(l4_out)
        l5_out_x = self.fc2_conv(self.fc_conv(l4_out_x))
        l5_out_x = self.insn5(l5_out_x+self.noise5)
        l1_out = self.up11(self.up10(l1_out_x))
        l3_out = self.up33(self.up32(self.up31(self.up30(l3_out_x))))
        l5_out = self.up54(self.up53(self.up52(self.up51(self.up50(l5_out_x)))))
        top_out = self.bn(self.relu(l1_out+l3_out+l5_out))
        top_out = self.activ(self.top_conv(top_out))
        return top_out
    
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

if __name__=='__main__':
    img = torch.randn([5, 3, 256, 256])
    model = SAE_RESNEXT_ENCODER()
    outs = model(img)
    for o in outs:
        print(o.shape)
