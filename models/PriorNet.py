import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 调用Net函数 输入：先验图像p  输出：2 feature
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self,embed_dim0,embed_dim1):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, embed_dim0, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(embed_dim0),
            ResidualBlock(embed_dim0),
            ResidualBlock(embed_dim0),
            ResidualBlock(embed_dim0),
            ResidualBlock(embed_dim0)
        )

        self.conv2x = ConvLayer(embed_dim0, embed_dim1, kernel_size=3, stride=2)
        self.dense1 = nn.Sequential(
            ResidualBlock(embed_dim1),
            ResidualBlock(embed_dim1),
            ResidualBlock(embed_dim1),
            ResidualBlock(embed_dim1),
            ResidualBlock(embed_dim1)
        )


    def forward(self, p):
        res1x = self.conv_input(p)
        x1 = self.dense0(res1x) + res1x

        res2x = self.conv2x(x1)
        x2 =self.dense1(res2x) + res2x


        return x1, x2

