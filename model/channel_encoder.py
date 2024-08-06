import torch
import torch.nn as nn
import torch.nn.functional as F
def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)

def get_norm2d(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(dim)


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    return conv

class Conv2d(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 g=1,
                 act_type='',          # activation
                 norm_type='',         # normalization
                 depthwise=False):
        super(Conv2d, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            assert c1 == c2, "In depthwise conv, the in_dim (c1) should be equal to out_dim (c2)."
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=g, bias=add_bias))
            if norm_type:
                convs.append(get_norm2d(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)

class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out
    

import torch.nn.functional as F
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,21),padding=(0,10),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(21,1),padding=(10,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs #(1,64,32,32)
        #pdb.set_trace()
        x_init = self.dconv5_5(inputs) #(1,64,32,32)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x) #(1,64,32,32)
        out = spatial_att * inputs
        out = self.conv(out) #(1,64,32,32)
        return out
    
class MyChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            #SPAM(),
            #SpatialAttention(in_channels=out_dim, out_channels=out_dim),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )
        self.fuse_time=nn.Sequential(
            RepBlock(464,464) #464

        )
    def forward(self, x1, x2):
        """
            x: [B, C, H, W]
        """
        x2=self.fuse_time(x2)
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x
    


def build_mychannel_encoder(cfg, in_dim, out_dim):
    encoder = MyChannelEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

    return encoder
