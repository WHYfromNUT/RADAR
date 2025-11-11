import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet18
from Swin import Swintransformer
from models.ASPP import ASPP
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from functools import partial
from math import log


bn_mom = 0.1
bce_loss = nn.BCELoss(reduction='mean')
def loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0,len(preds)):

        if(preds[i].shape[2]!=target.shape[2] or preds[i].shape[3]!=target.shape[3]):

            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target)
        if(i==0):
            loss0 = loss
    return loss0, loss

def dice_loss(predict, target):
    if (predict.shape[2] != target.shape[2] or predict.shape[3] != target.shape[3]):
        target = F.interpolate(target, size=predict.size()[2:], mode='bilinear', align_corners=True)
    # print(target.shape)
    # target = target.cpu()
    # target = target.numpy()
    # target = cv2.Laplacian(target, cv2.CV_8U)
    # target = torch.tensor(target)
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()




def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, flag=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)
        self.flag = flag

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.flag == 1:
            x = self.relu(x)
        return x

class MyBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(MyBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    # ratio = 16
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.ratio = ratio

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
        out = max_out  # 32 1 1
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out  # 1 44 44
        x = self.conv1(x)
        return self.sigmoid(x)


# like Mobile-Former 双向桥接模块
class BCR(nn.Module):
    def __init__(self, inplanes):
        super(BCR, self).__init__()

        self.upsample= nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.B = MyBottleNeck(inplanes, inplanes)
        self.f1_spatial = SpatialAttention()

        self.ASPP1 = ASPP(inplanes, inplanes)
        self.ASPP2 = ASPP(inplanes, inplanes)
        self.f2_channel = ChannelAttention(inplanes)

        self.conv_cat1 = BasicConv2d(inplanes * 2, inplanes, 3, 1, padding=1, flag=1)
        self.conv_cat2 = BasicConv2d(inplanes * 2, inplanes, 3, 1, padding=1, flag=1)
        self.conv_out = BasicConv2d(inplanes * 2, inplanes, 3, 1, padding=1, flag=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)

    def forward(self, f1, f2): # f1 local feature; f2 global feature
        if f1.size() != f2.size():
            f2 = self.upsample(f2)

        temp_2 = f2.mul(self.f2_channel(f2))
        f1 = self.conv_cat1(torch.cat((f1, temp_2), dim=1))
        f_B1 = self.B(f1)
        f1_out = f_B1

        temp_1 = f_B1.mul(self.f1_spatial(f_B1))
        f2_out = self.conv_cat2(torch.cat((temp_1, f2), dim=1))
        out = self.conv_out(torch.cat((self.ASPP1(f1_out),self.ASPP2(f2_out)), dim=1))

        return out


class GSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y): #x:res ; y: Trans
        if x.size() != y.size():
            y = F.interpolate(y, x.size()[2:], mode='bilinear', align_corners=False)
        batch_size = x.shape[0]
        chanel = x.shape[1]
        sc = x

        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        sc1 = x
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        sc2 = y
        y = self.lny(y)

        B, N, C = x.shape
        x_q = self.q1(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_q = self.q2(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_kv = self.kv1(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        y_kv = self.kv2(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        x_q = x_q[0]
        y_q = y_q[0]
        x_k, x_v = x_kv[0], x_kv[1]
        y_k, y_v = y_kv[0], y_kv[1]

        attn1 = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        x1 = (attn1 @ y_v).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x1 = (x1 + sc1)
        x1 = x1.permute(0, 2, 1)
        x1 = x1.view(batch_size, chanel, *sc.size()[2:])

        attn2 = (y_q @ x_k.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        y1 = (attn2 @ x_v).transpose(1, 2).reshape(B, N, C)
        y1 = self.proj(y1)
        y1 = (y1 + sc2)
        y1 = y1.permute(0, 2, 1)
        y1 = y1.view(batch_size, chanel, *sc.size()[2:])

        out = self.conv2(torch.cat((x1,y1),dim=1))

        return out


class EGblock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(EGblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(outplanes,1,kernel_size=1, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        p = self.conv(x)
        p = p + x
        p2 = self.conv2(p)
        p = p + p2
        out = self.sigmoid(self.conv3(p))
        return out, p




class EEF(nn.Module):
    def __init__(self, channel):
        super(EEF, self).__init__()
        self.conv2d = BasicConv2d(channel, channel, 3  ,stride=1, padding=1,flag=1)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f, eg):
        if f.size() != eg.size():
            f = F.interpolate(f, eg.size()[2:], mode='bilinear', align_corners=False)
        x = f * eg + f
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x


class S_head(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(S_head, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out

class RADAR(nn.Module):
    def __init__(self, cfg=None):
        super(RADAR, self).__init__()
        # channel reduction
        # Conv+BN+Relu
        self.resnet = resnet18()
        self.swin = Swintransformer(448)
        # self.swin.load_state_dict(torch.load('../pre/swin224.pth')['model'],strict=False)
        # self.resnet.load_state_dict(torch.load('../pre/resnet18.pth'),strict=False)
        if torch.cuda.is_available():
            self.swin.load_state_dict(torch.load('./models/pre/swin_base_patch4_window7_224_22k.pth'), strict=False)
            self.resnet.load_state_dict(torch.load('./models/pre/resnet18-5c106cde.pth'), strict=False)
            self.swin = self.swin.cuda()
            self.resnet = self.resnet.cuda()
            # self.swin.load_state_dict(torch.load('../models/pre/swin_base_patch4_window7_224_22k.pth'), strict=False)
            # self.resnet.load_state_dict(torch.load('../models/pre/resnet18-5c106cde.pth'), strict=False)
            # self.swin = self.swin.cuda()
            # self.resnet = self.resnet.cuda()
        else:
            self.swin.load_state_dict(torch.load('../pre/swin_base_patch4_window7_224_22k.pth',map_location='cpu' ),strict=False)
            self.resnet.load_state_dict(torch.load('../pre/resnet18-5c106cde.pth',map_location='cpu' ), strict=False)
        # self.swin.load_state_dict(torch.load('./pre/swin_base_patch4_window7_224_22k.pth',map_location='cpu' ),strict=False)
        # self.resnet.load_state_dict(torch.load('./pre/resnet18-5c106cde.pth',map_location='cpu' ), strict=False)

        self.cbr1 = BasicConv2d(64, 64, 3, 1, padding=1, flag=1)
        self.cbr2 = BasicConv2d(128, 64, 3, 1, padding=1, flag=1)
        self.cbr3 = BasicConv2d(256, 64, 3, 1, padding=1, flag=1)
        self.cbr4 = BasicConv2d(512, 64, 3, 1, padding=1, flag=1)
        #self.cbr5 = BasicConv2d(128, 64, 3, 1, padding=1, flag=1)
        self.cbr6 = BasicConv2d(256, 64, 3, 1, padding=1, flag=1)
        self.cbr7 = BasicConv2d(512, 64, 3, 1, padding=1, flag=1)
        self.cbr8 = BasicConv2d(512, 64, 3, 1, padding=1, flag=1)

        self.cbr9 = BasicConv2d(128, 64, 3, 1, padding=1, flag=1)
        self.cbr10 = BasicConv2d(128, 64, 3, 1, padding=1, flag=1)

        self.GSA = GSA(64)
        self.BCR3 = BCR(64)
        self.BCR2 = BCR(64)
        self.BCR1 = BCR(64)
        self.eg = EGblock(64,64)
        self.EEF = EEF(64)

        self.s_shead1 = S_head(64,1)
        self.s_shead2 = S_head(64, 1)
        self.s_shead3 = S_head(64, 1)
        self.s_shead4 = S_head(64, 1)

        self.s_shead5 = S_head(64, 1)

    def compute_loss(self, preds, targets):

        return loss_fusion(preds, targets)

    def compute_boundaryloss(self, edge, targets_edge):

        return dice_loss(edge, targets_edge)


    def forward(self, x,shape=None,mask=None):
        print('useRADAR')
        shape = x.size()[2:] if shape is None else shape #input 1x3x1024x1024
        y = F.interpolate(x, size=(448,448), mode='bilinear',align_corners=True)
        r2,r3,r4,r5 = self.resnet(x) #r2:64x256x256 r3:128x128x128 r4:256x64x64 r5:512x32x32
        s1,s2,s3,s4 = self.swin(y) #s1:128x112x112 s2:256x56x56 s3:512x28x28 s4:512x28x28

        r2 = self.cbr1(r2)
        r3 = self.cbr2(r3)
        r4 = self.cbr3(r4)
        r5 = self.cbr4(r5)
        #s1 = self.cbr5(s1)
        s2 = self.cbr6(s2)
        s3 = self.cbr7(s3)
        s4 = self.cbr8(s4)

        csf = self.GSA(r5,s4)
        c = self.BCR1(r5,csf)
        p1 = self.s_shead1(c)

        s3 = F.interpolate(s3, size=c.size()[2:], mode='bilinear', align_corners=False)
        b = self.BCR2(r4,self.cbr9(torch.cat((c,s3),dim=1)))
        p2 = self.s_shead2(b)

        s2 = F.interpolate(s2, size=b.size()[2:], mode='bilinear', align_corners=False)
        a = self.BCR3(r3, self.cbr10(torch.cat((b, s2), dim=1)))

        p3 = self.s_shead3(a)

        egle,_ = self.eg(r2)
        final_out = self.EEF(a,egle)
        p4 = self.s_shead4(final_out)


        semantic_map = self.s_shead5(csf)

        return [torch.sigmoid(p4), torch.sigmoid(p3),torch.sigmoid(p2), torch.sigmoid(p1)],[torch.sigmoid(semantic_map)],[egle], [p3, p2,p1,semantic_map]




if __name__ == '__main__':
    import time
    model = RADAR().cuda()
    input_tensor = torch.randn(2, 3, 1024, 1024).cuda()
    a = torch.randn(16, 16)
    b = torch.randn( 16, 16)

    prediction1= model(input_tensor)
    print(prediction1[0][2].shape)
