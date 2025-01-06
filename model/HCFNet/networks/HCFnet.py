import torch
import torch.nn as nn
from .BaseConv import BaseConv

import torch.nn.functional as F
# from ptflops import get_model_complexity_info
from ..main_blocks import PPA, DASI, MDCR
from ..utils.registry import NET_REGISTRY


@NET_REGISTRY.register()
class SAHead(nn.Module):
    def __init__(self, num_calsses=1, width=1.0, in_channels = 256, act='silu'):
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
class HCFnet(nn.Module):
    def __init__(self,
                in_features=3,
                out_features=1,
                gt_ds = False,
                ) -> None:
        super().__init__()
        self.gt_ds = gt_ds

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.p1 = PPA(in_features=in_features,
                                filters=32)

        self.respath1 = DASI(in_features=32,
                                out_features=32,
                               )
        self.p2 = PPA(in_features=32,
                                filters=int(32 * 2))
        self.respath2 = DASI(in_features=64,
                                out_features=32 * 2,
                                )
        self.p3 = PPA(in_features=64,
                                filters=int(32 * 4))
        self.respath3 = DASI(in_features=128,
                                out_features=32 * 4,
                                )
        self.p4 = PPA(in_features=128,
                                filters=int(32 * 8))
        self.respath4 = DASI(in_features=256,
                                out_features=32 * 8,
                                )
        self.p5 = PPA(in_features=256,
                                filters=int(32 * 16))

        self.mdcr = MDCR(in_features=int(512),out_features=int(512))



        self.up1 = nn.Sequential(nn.ConvTranspose2d(512,
                                      32*8,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 8),
                                    nn.ReLU())


        self.p6 = PPA(in_features=32 * 8 * 2,
                                filters=int(32 * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256,
                                      32*4,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 4),
                                    nn.ReLU())

        self.p7 = PPA(in_features=32 * 4 * 2,
                                filters=int(32 * 4))

        self.up3 = nn.Sequential(nn.ConvTranspose2d(128,
                                      32*2,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 2),
                                    nn.ReLU())

        self.p8 = PPA(in_features=32 * 2 * 2,
                                filters=int(32 * 2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64,
                                      32,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.p9 = PPA(in_features=32 * 2,
                                filters=int(32))


        self.out = nn.Conv2d(in_channels=32,
                              out_channels=4,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.out1 =  nn.Conv2d(in_channels=512,
                      out_channels=out_features,
                      kernel_size=(1, 1),
                      padding=(0, 0)
                      )
        self.out2 = nn.Conv2d(in_channels=256,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.out3 = nn.Conv2d(in_channels=128,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.out4 = nn.Conv2d(in_channels=64,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.head = SAHead()

    def forward(self, x):
        #encoder
        x1 = self.p1(x)
        xp1 = self.maxpool(x1)
        x2 = self.p2(xp1)
        xp2 = self.maxpool(x2)
        x3 = self.p3(xp2)
        xp3 = self.maxpool(x3)
        x4 = self.p4(xp3)
        xp4 = self.maxpool(x4)
        x = self.p5(xp4)
        x = self.mdcr(x)


        x1_res = self.respath1(x1, x2, None) # 1 32 512 512
        x2_res = self.respath2(x2, x3, x1) # 1 64 256 256
        x3_res = self.respath3(x3, x4, x2) # 1 128 128 128
        x4_res = self.respath4(x4,  x, x3) # 1 256 64 64


        #decoder
        out4 = F.interpolate(self.out1(x), scale_factor=16, mode ='bilinear', align_corners=True)
        x = self.up1(x)
        x = torch.cat((x, x4_res), dim=1)
        x = self.p6(x)
        out3 = F.interpolate(self.out2(x), scale_factor=8, mode ='bilinear', align_corners=True)
        x = self.up2(x)
        x = torch.cat((x, x3_res), dim=1)
        x = self.p7(x)
        out2 = F.interpolate(self.out3(x), scale_factor=4, mode ='bilinear', align_corners=True)
        x = self.up3(x)
        x = torch.cat((x, x2_res), dim=1)
        x = self.p8(x)
        out1 = F.interpolate(self.out4(x), scale_factor=2, mode ='bilinear', align_corners=True)
        x = self.up4(x)
        x = torch.cat((x, x1_res), dim=1)
        x = self.p9(x)
        out = self.out(x)
        b,c,h,w = out.shape
        out = out.view(b,256,64,64)
        #print(out.shape)
        return self.head(out)




