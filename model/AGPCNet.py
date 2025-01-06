import torch
import torch.nn as nn


import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        c1 = self.layer2(x)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return c1, c2, c3

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

__all__ = ['NonLocalBlock', 'GCA_Channel', 'GCA_Element', 'AGCB_Element', 'AGCB_Patch', 'CPM']


class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out


class GCA_Channel(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Channel, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca


class GCA_Element(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Element, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        batch_size, C, height, width = x.size()

        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        else:
            raise NotImplementedError
        return gca


class AGCB_Patch(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_Element(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Element, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Element(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            # attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = context * gca
        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_NoGCA(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32):
        super(AGCB_NoGCA, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class CPM(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(CPM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list = nn.ModuleList(
                [AGCB_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        elif block_type == 'element':
            self.scale_list = nn.ModuleList(
                [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list)):
            blocks.append(self.scale_list[i](reduced))
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
        return out

__all__ = ['AsymFusionModule']


class AsymFusionModule(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):
        super(AsymFusionModule, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_high, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(planes_low, planes_out, 3, 1, 1),
            nn.BatchNorm2d(planes_out),
            nn.ReLU(True),
        )

    def forward(self, x_high, x_low):
        x_high = self.plus_conv(x_high)
        pa = self.pa(x_low)
        ca = self.ca(x_high)

        feat = x_low + x_high
        feat = self.end_conv(feat)
        feat = feat * ca
        feat = feat * pa
        return feat


__all__ = ['agpcnet']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class AGPCNet(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


class AGPCNet_Pro(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet_Pro, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


def agpcnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop):
    return AGPCNet(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att, drop=drop)

class SiLU(nn.Module):
    """SiLU激活函数"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def get_activation(name="silu", inplace=True):
    # inplace为True，将会改变输入的数据 (降低显存)，否则不会改变原输入，只会产生新的输出
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """带归一化和激活函数的标准卷积并且保证宽高不变"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核大小
        :param stride: 步长
        :param groups: 是否分组卷积
        :param bias: 偏置
        :param act: 所选激活函数
        """
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class AGPCNetHead(nn.Module):
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

class AGPCBody(nn.Module):
    def __init__(self, num_classes=1, phi='s'):
        super().__init__()
        self.backbone = agpcnet(backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1)
        self.head = AGPCNetHead(num_classes)
        # self.conv = BaseConv(64, 256, 3, 1)
        self.conv = nn.Sequential(
            BaseConv(1, 4, 3, 2), # [b, 4, 320, 320]
            BaseConv(4, 16, 3, 2), # [b, 64, 160 ,160]
            BaseConv(16, 64, 3, 2), # [b, 64, 80, 80]
            BaseConv(64, 256, 3, 1)
            )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.backbone(x)  # [b, 1, 640, 640]
        # x = x.view(b, 64, 80, 80)
        x = self.conv(x)
        # x = x.view(b, 256, 32, 32)
        outputs = self.head(x)

        return outputs

if __name__ == "__main__":
    # 原是512 现用640 
    model_path='/home/pengshuang/Public/Exper_Results/Exper_Results/IRDST/AGPC/ep045-loss3.626-val_loss3.504.pth'
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(model_path, map_location=device)
    a = torch.randn([4,3,640,640])
    model= AGPCBody(1, 's')
    model.load_state_dict(ck)
    model.eval()
    a=model(a)
    print(a[0].shape)