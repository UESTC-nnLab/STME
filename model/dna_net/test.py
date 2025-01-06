import torch
import torch.nn as nn
from .BaseConv import BaseConv
from .DNANet import DNANet

class DNAHead(nn.Module):
    def __init__(self, num_classes, in_channels = 256, act='silu'):
        super().__init__()
        # 分类的两层卷积
        self.cls_convs = nn.Sequential(
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act),
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act)
        )
        # 分类的预测
        self.cls_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        
        # 回归的两层卷积
        self.reg_convs = nn.Sequential(
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act),
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act)
        )
        # 回归的预测
        self.reg_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # 是否有检测对象预测层
        self.obj_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        # inputs [b, 256, 32, 32]
        outputs = []
        # 分类提取器 [b, 256, 32, 32] -> [b, 256, 32, 32]
        cls_feat = self.cls_convs(inputs)
        # 分类的输出 [b, 256, 32, 32] -> [b, num_classes, 32, 32]
        cls_output = self.cls_preds(cls_feat)
        
        # 回归特征提取 [b, 256, 32, 32] -> [b, 256, 32, 32]
        reg_feat = self.reg_convs(inputs)
        # 特征点的回归系数 [b, 256, 32, 32] -> [b, 4, 32, 32]
        reg_output = self.reg_preds(reg_feat)
        
        # 判断特征点是否有对应的物体(利用回归特征提取) [b, 256, 32, 32] -> [b, 1, 32, 32]
        obj_output = self.obj_preds(reg_feat)
        
        # 将结果整合到一起，0到3为回归结果，4为是否有物体的结果，其余为种类置信得分
        # [b, 4, 32, 32] + [b, 1, 32, 32] + [b, num_classes, 32, 32] -> [b, 5+num_classes, 32, 32]
        output = torch.cat([reg_output, obj_output, cls_output], dim=1)
        outputs.append(output)
        
        return outputs


class DNABody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()

        self.backbone = DNANet(num_classes, 3)
        # self.conv = nn.Sequential(
        #     BaseConv(4, 16, 3, 2), # [b, 16, 320, 320]
        #     BaseConv(16, 64, 3, 2), # [b, 64, 160, 160]
        #     BaseConv(64, 256, 3, 2), #[b, 256, 80, 80]
        # )
        self.conv = nn.Sequential(
            BaseConv(4, 2, 3, 1), # [b, 16, 320, 320]
            BaseConv(2, 1, 3, 1), # [b, 64, 160, 160]
        )
        self.head = DNAHead(num_classes)

    def forward(self, x):
        b,_,_,_ = x.shape
        net_out = self.backbone(x)
        # x = torch.cat([net_out[0], net_out[1], net_out[2], net_out[3]], dim=1).view(b, 256, 80, 80) # [b, 4, 640, 640] -> [b, 256, 80, 80]
        x = torch.cat([net_out[0], net_out[1], net_out[2], net_out[3]], dim=1) # [b, 4, 640, 640]
        x = self.conv(x)
        # outputs = self.head(x)

        return x
        
if __name__ == "__main__":
    a = torch.rand((4,3,640,640))
    a = DNABody(1, 's')(a)
    print(a[0].shape)