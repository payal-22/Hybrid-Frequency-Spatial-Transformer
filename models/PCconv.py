from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch
import torch.nn as nn
import util.util as util
from util.Selfpatch import Selfpatch


# SE MODEL
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


class Convnorm(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', activ='leaky'):
        super().__init__()
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)

        if sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1)
        if activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out[0])
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out = self.bn(out)
            out = self.activation(out)
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)

        elif self.innorm:
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)
        elif self.outer:
            out = self.conv(out)
            out = self.bn(out)
        else:
            out = self.conv(out)
            out = self.bn(out)
            if hasattr(self, 'activation'):
                out = self.activation(out)
        return out



class ConvDown(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=False):
        super(ConvDown, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.bn = nn.InstanceNorm2d(output_size)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return self.act(out)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(inner_nc, inner_nc, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(inner_nc),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = self.down(out_csa)

        return out_32


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)

    def forward(self, inputt):
        input = inputt
        output = self.input_conv(input)
        return output


class PCconv(nn.Module):
    def __init__(self, embed_dim0):
        super(PCconv, self).__init__()
        self.down_128 = ConvDown(embed_dim0, 2*embed_dim0, 4, 2, padding=1)

        self.down2_1 = ConvDown(4*embed_dim0, 2*embed_dim0, 1, 1)
        self.down3_1 = ConvDown(6*embed_dim0, 2*embed_dim0, 1, 1)
        self.fuse = ConvDown(4*embed_dim0, 2*embed_dim0, 1, 1)
        self.equalizations = ConvDown(2*embed_dim0, 2*embed_dim0, 1,1)
        #self.base= BASE(2*embed_dim0)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(2*embed_dim0, 2*embed_dim0, innorm=True)]
            seuqence_5 += [PCBActiv(2*embed_dim0, 2*embed_dim0, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(2*embed_dim0, 2*embed_dim0, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, skip1, skip2,p1,p2):

        x_1 = self.activation(skip1)
        x_2 = self.activation(skip2)
        x_3 = self.activation(p1)
        x_4 = self.activation(p2)

        # Change the shape of each layer and intergrate low-level/high-level features
        x_1 = self.down_128(x_1)
        x_3 = self.down_128(x_3)

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_3], 1)
        x_ST = torch.cat([x_2, x_4], 1)
        # x_DE = x_1
        # x_ST = x_2

        x_ST = self.down2_1(x_ST)
        x_DE = self.down2_1(x_DE)


        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3, x_DE_5, x_DE_7], 1)
        x_DE_fi = self.down3_1(x_DE_fuse)

        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3, x_ST_5, x_ST_7], 1)
        x_ST_fi = self.down3_1(x_ST_fuse)

        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat_fuse = self.fuse(x_cat)



        return x_cat_fuse
