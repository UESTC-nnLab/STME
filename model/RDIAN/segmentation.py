import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .cbam import *
from .direction import *
from .BaseConv import BaseConv

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )
        self.conv = nn.Sequential(
            BaseConv(1, 4, 3, 2), # [b, 4, 320, 320]
            BaseConv(4, 16, 3, 2), # [b, 16, 160, 160]
            BaseConv(16, 64, 3, 2), # [b, 64, 80, 80]
            BaseConv(64, 256, 3, 1), # [b, 256, 80, 80]
        )
    def forward(self, x):
        x = self.block(x)
        return self.conv(x)

class SAHead(nn.Module):
    def __init__(self, num_calsses = 1, width=1.0, in_channels = 256, act='silu'):
        super().__init__()
        
        self.cls_convs = nn.Sequential(
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act),
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act)
        )
        # class
        self.cls_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=num_calsses, kernel_size=1, stride=1, padding=0)
        
        self.reg_convs = nn.Sequential(
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act),
            BaseConv(int(in_channels*width), int(in_channels*width), 3, 1, act=act)
        )
        # regression
        self.reg_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # object
        self.obj_preds = nn.Conv2d(in_channels=int(in_channels*width), out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        # inputs [b, 256, 80, 80]
        
        outputs = []
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        cls_feat = self.cls_convs(inputs)
        # [b, 256, 80, 80] -> [b, num_classes, 80, 80]
        cls_output = self.cls_preds(cls_feat)
        
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        reg_feat = self.reg_convs(inputs)
        # [b, 256, 80, 80] -> [b, 4, 80, 80]
        reg_output = self.reg_preds(reg_feat)
        
        # [b, 256, 80, 80] -> [b, 1, 80, 80]
        obj_output = self.obj_preds(reg_feat)
        
        # [b, 4, 80, 80] + [b, 1, 80, 80] + [b, num_classes, 80, 80] -> [b, 5+num_classes, 80, 80]
        output = torch.cat([reg_output, obj_output, cls_output], dim=1)
        outputs.append(output)
        
        return outputs
    
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

class NewBlock(nn.Module):
    def __init__(self, in_channels, stride,kernel_size,padding):
        super(NewBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.layer2 = conv_batch(reduced_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out        

class RDIAN(nn.Module):
    def __init__(self,num_classes=1,phi='s'):
    
        super(RDIAN, self).__init__()        
        accumulate_params = "none"
        self.conv1 = conv_batch(1, 16)
        self.conv2 = conv_batch(16, 32, stride=2)       
        self.residual_block0 = self.make_layer(NewBlock, in_channels=32, num_blocks=1, kernel_size=1,padding=0,stride=1)
        self.residual_block1 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=3,padding=1,stride=1)
        self.residual_block2 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=5,padding=2,stride=1)
        self.residual_block3 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=7,padding=3,stride=1)
        self.cbam  = CBAM(32, 32)        
        self.conv_cat = conv_batch(4*32, 32, 3, padding=1)
        self.conv_res = conv_batch(16, 32, 1, padding=0)
        self.relu = nn.ReLU(True)
        
        self.d11=Conv_d11()
        self.d12=Conv_d12()
        self.d13=Conv_d13()
        self.d14=Conv_d14()
        self.d15=Conv_d15()
        self.d16=Conv_d16()
        self.d17=Conv_d17()
        self.d18=Conv_d18()

        self.head = _FCNHead(32, 1)
        self.yoloHead = SAHead()
    def forward(self, x):
        ##weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        ##x = torch.sum(x * weights, dim=1, keepdim=True).detach()
        ##print("x:",x.shape)
        x = x[:,-1,:,:].unsqueeze(1)

        _, _, hei, wid = x.shape
        d11 = self.d11(x)
        d12 = self.d12(x)
        d13 = self.d13(x)
        d14 = self.d14(x)
        d15 = self.d15(x)
        d16 = self.d16(x)
        d17 = self.d17(x)
        d18 = self.d18(x)
        md = d11.mul(d15) + d12.mul(d16) + d13.mul(d17) + d14.mul(d18)
        md = F.sigmoid(md)
        
        out1= self.conv1(x)        
        out2 = out1.mul(md)       
        out = self.conv2(out1 + out2)
            
        c0 = self.residual_block0(out)
        c1 = self.residual_block1(out)
        c2 = self.residual_block2(out)
        c3 = self.residual_block3(out)
 
        x_cat = self.conv_cat(torch.cat((c0, c1, c2, c3), dim=1)) #[16,32,240,240]
        x_a = self.cbam(x_cat)
        
        temp = F.interpolate(x_a, size=[hei, wid], mode='bilinear')
        temp2 = self.conv_res(out1)
        x_new = self.relu( temp + temp2)
        self.x_new = x_new
        
        pred = self.head(x_new)
        ##print("pred:",pred.shape)
        return self.yoloHead(pred)
             
    def make_layer(self, block, in_channels, num_blocks, stride, kernel_size, padding):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, stride, kernel_size, padding))
        return nn.Sequential(*layers)
