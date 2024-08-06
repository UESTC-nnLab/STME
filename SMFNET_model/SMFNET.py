import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
#from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from .channel_encoder import build_mychannel_encoder
from .backbone3d import Backbone3D


class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  
        
        
        return P3_out

class Feature_Backbone(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features=in_features

    def forward(self, input):
        out_features            =  self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        return [feat1,feat2,feat3]

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv =BaseConv #if depthwise else BaseConv
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        #self.conv2=nn.Identity()
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
    
class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        n=1
        #--------------------------------------------------#
        #   主干部分的初次卷积
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   大的残差边部分的初次卷积
        #--------------------------------------------------#
        self.conv2  = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=act) #in_channel
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # self.deepfeature=nn.Sequential(BaseConv(hidden_channels, hidden_channels//2, 1, stride=1, act=act),
        #       BaseConv(hidden_channels//2, hidden_channels, 3, stride=1, act=act))
        #-----------------------------------------------#
        #module_list = [Bottleneck(hidden_channels, hidden_channels, True, 1.0, depthwise, act=act) for _ in range(n)]
        #self.deepfeature      = nn.Sequential(*module_list)
        self.conv3  = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act) #2*hidden_channel

        #--------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        #--------------------------------------------------#
        # module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        # self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1是主干部分
        #-------------------------------#
        #x_1 = self.conv1(x)
        x=self.conv1(x)
        #-------------------------------#
        #   x_2是大的残差边部分
        #-------------------------------#
        #x_2 = self.conv2(x)
        x=self.conv2(x)
        #-----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        #-----------------------------------------------#
        #x_1 = self.deepfeature(x_1)
        #-----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        #-----------------------------------------------#
        #x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        #-----------------------------------------------#
        return self.conv3(x)

class Feature_Fusion(nn.Module):
    def __init__(self, in_channels = [128,128,128], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(2*int(in_channels[2]), int(in_channels[1]), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = FusionLayer(
            int(2 * in_channels[1]), 
            int(in_channels[1]),
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(2*in_channels[1]), int(in_channels[0]), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = FusionLayer(
            int(2 * in_channels[0]),
            int(in_channels[0]),
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = input # self.backbone.forward(input)
        [feat1, feat2, feat3]   =out_features# [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        #P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(feat3)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #pdb.set_trace()
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P4=self.lateral_conv0(P5_upsample)
        #P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        #P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out=  self.reduce_conv1(P4_upsample) 
        #P3_out      = self.C3_p3(P4_upsample)  
        
        
        return P3_out
class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))#128-> 256 通道整合
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
import pdb
model_config = {
  
        'backbone_2d': 'yolo_free_nano',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        # ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '1.0x', #1.0x
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        'head_dim': 128,#64
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': True,
    

}

def build_backbone_3d(cfg, pretrained=False):
    backbone = Backbone3D(cfg, pretrained)
    return backbone, backbone.feat_dim
mcfg=model_config

class STNetwork(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=10):
        super(STNetwork, self).__init__()
        #self.in_features=['stage1','stage2','stage3']
        self.num_frame = num_frame
        self.backbone = Feature_Backbone(0.33,0.50) #Feature_Extractor(0.33,0.50) 
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            mcfg, pretrained=mcfg['pretrained_3d'] and True)
        self.cls_channel_encoders1=build_mychannel_encoder(mcfg, 128+bk_dim_3d, mcfg['head_dim'])
        self.cls_channel_encoders2=build_mychannel_encoder(mcfg, 256+bk_dim_3d, mcfg['head_dim'])
        self.cls_channel_encoders3=build_mychannel_encoder(mcfg, 512+bk_dim_3d, mcfg['head_dim'])
        self.feature_fusion=Feature_Fusion()
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        feat = []
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""
        [feat1,feat2,feat3]=self.backbone(inputs[:,:,-1,:,:])#[64,32,16]
       
        featt=self.backbone_3d(inputs) #[4,464,16,16]
        feat_3d_up3=featt #
        feat_3d_up2 = F.interpolate(featt, scale_factor=2 ** (2 -1))#[4,464,32,32]
        feat_3d_up1 =F.interpolate(featt,scale_factor=2 ** (2 -0))##[4,464,64,64]
        
        feat_all1=self.cls_channel_encoders1(feat1, feat_3d_up1)#4,128,64,64]
        feat_all2=self.cls_channel_encoders2(feat2, feat_3d_up2)#[4,128,32,32]
        feat_all3=self.cls_channel_encoders3(feat3, feat_3d_up3)#[4,128,16,16]
        '''feature fusion'''
        feat_all=self.feature_fusion([feat_all1,feat_all2,feat_all3])#[b,c,h,w]
        feat.append(feat_all)
        outputs  = self.head(feat)
        motion_loss=0
        if self.training:
            return  outputs, motion_loss  
        else:
            return  outputs
        


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)  






# def print_model_size(model):
#     # 计算模型参数的总数量
#     param_num = sum(p.numel() for p in model.parameters())
#     # 计算模型参数所占的总内存大小（假设参数为float32，即4字节）
#     param_size = param_num * 4 / (1024**2)  # 单位转换为MB
#     print(f"模型参数数量：{param_num}")
#     print(f"模型大小：{param_size:.2f} MB")
from thop import profile
def get_model_complexity(model, input_size=(3,5,512,512)):
    """计算并打印模型的FLOPs和Params"""
    # 计算FLOPs和Params
    input = torch.randn(2, *input_size)  # 假设输入是一个batch size为1的张量，根据您的模型调整尺寸
    flops, params = profile(model, inputs=(input,))
    
    # 将FLOPs转换为GFLOPs
    gflops = flops / 1e9
    
    # 将Params转换为MParams
    mparams = params / 1e6
    
    return gflops, mparams
#这个版本是多级时序特征融合
if __name__ == "__main__":
    
    # from yolo_training import YOLOLoss
    net = STNetwork(num_classes=1, num_frame=5)
    
    gflops, mparams = get_model_complexity(net)

    print(f"GFLOPs: {gflops:.2f}")
    print(f"MParams: {mparams:.2f}")  
    

    # print_model_size(net)
    bs = 4
    a = torch.randn(bs, 3, 5, 512, 512)
    out = net(a)
    pdb.set_trace()
    print(out[0].shape)
    print(out[1].shape)




    # for item in out:
    #     print(item.size())
        
    # yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    # target = torch.randn([bs, 1, 5]).cuda()
    # target = nn.Softmax()(target)
    # target = [item for item in target]

    # loss = yolo_loss(out, target)
    # print(loss)
