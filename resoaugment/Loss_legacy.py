import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_patches(img, kernel_n, stride=1):
    N, C, H, W = img.shape
    kernel_size = torch.zeros(kernel_n, kernel_n)
    kernel_list = []
    for i in range(kernel_n):
        for j in range(kernel_n):
            k = kernel_size.clone()
            k[i][j] = 1
            kernel_list.append(k)
    kernel = torch.stack(kernel_list)
#     print("kernel:", kernel, kernel.shape)
    kernel = kernel.unsqueeze(1)         
#     print(kernel, kernel.shape)
    weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
    
    # 单独提取每一个通道
    out_0 = F.conv2d(img[:, :1, ...], weight, padding=int((kernel_n - 1) / 2))
    out_1 = F.conv2d(img[:, 1:2, ...], weight, padding=int((kernel_n - 1) / 2))
    out_2 = F.conv2d(img[:, 2:3, ...], weight, padding=int((kernel_n - 1) / 2))
    # out_3 = F.conv2d(img[:, 3:4, ...], weight, padding=int((kernel_n - 1) / 2))
    
    out_list = []
    for n in range(N):
        out_n_list = []
        # 将一张图像4个通道[a, b, c, d]中的tensor按
        # [a0, b0, c0, d0, a1, b1, c1, d1, ..., an, bn, cn, dn]的顺序重组
        for v in range(out_0.shape[1]):
#             print("out_0[n][v]:", out_0[n][v])
            out_n_list.append(out_0[n][v])
            out_n_list.append(out_1[n][v])
            out_n_list.append(out_2[n][v])
            # out_n_list.append(out_3[n][v])
        out_n = torch.stack(out_n_list)
        # print("out_n:", out_n)
        out_list.append(out_n)
    out = torch.stack(out_list)
    out = out.permute(0, 2, 3, 1)
    return out

class DSSIMLoss(nn.Module):
    def __init__(self, k_1=0.01, k_2=0.03, kernel_size=3, max_value=1.0):
        super(DSSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.k_1 = k_1
        self.k_2 = k_2
        self.max_value = max_value
        self.c_1 = (self.k_1 * self.max_value) ** 2
        self.c_2 = (self.k_2 * self.max_value) ** 2
    
    def forward(self, y_pred, y_true):
        patches_true = extract_patches(y_true, self.kernel_size)
        patches_pred = extract_patches(y_pred, self.kernel_size)
        # Get mean
        u_true = torch.mean(patches_true, axis=-1)
        u_pred = torch.mean(patches_pred, axis=-1)
        # Get variance
        var_true = torch.var(patches_true, axis=-1)
        var_pred = torch.var(patches_pred, axis=-1)
        # Get standard deviation
        covar_true_pred = torch.mean(
            patches_true * patches_pred, axis=-1) - u_true * u_pred

        ssim = (2 * u_true * u_pred + self.c_1) * (
            2 * covar_true_pred + self.c_2)
        denom = (torch.square(u_true) + torch.square(u_pred) + self.c_1) * (
            var_pred + var_true + self.c_2)
        ssim /= denom  # no need for clipping, c_1 + c_2 make the denorm non-zero
        dssim = (1.0 - ssim) / 2.0
        return torch.mean(dssim)

class MultiFeatLoss(nn.Module):
    def __init__(self):
        super(MultiFeatLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.DSSIM = DSSIMLoss()
    
    def get_multi_maps(self, big_pic):
        maps = []
        size = 24
        for _ in range(5):
            y = F.upsample(big_pic, size=(size, size), mode='bilinear')
            maps.append(y)
            size *= 2
        return maps

    def forward(self, x, y):
        maps = self.get_multi_maps(y)
        loss1 = self.L1(x[0], maps[0])
        loss2 = self.DSSIM(x[1], maps[1])
        loss3 = self.L1(x[2], maps[2])
        loss4 = self.DSSIM(x[3], maps[3])
        loss5l = self.L1(x[4], maps[4])
        loss5d = self.DSSIM(x[4], maps[4])
        total_loss = (loss1+loss2+loss3+loss4+loss5d+loss5l)/6.0
        return total_loss

if __name__ == "__main__":
    imgs = torch.randn([2, 3, 10, 7])
    pred = torch.randn([2, 3, 10, 7])
    Loss = DSSIMLoss()
    out = Loss(imgs, pred)
    print(out, out.shape)
