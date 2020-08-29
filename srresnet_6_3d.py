# -*- coding: utf-8 -*-
import functools
import numpy as np

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(0.2, True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    '''
    given appearance estimation, deep features, and residual maps as input,
    output the final estimation
    '''

    def __init__(self, n=2, k_nn=3, bn=False):
        super(Residual, self).__init__()
        dim = 64
        self.r1 = nn.Sequential(Conv2d(n * 3, 32, 3, NL='prelu', same_padding=True, bn=bn))
        self.r2 = nn.Sequential(Conv2d(n * 3, 16, 5, NL='prelu', same_padding=True, bn=bn))
        self.r3 = nn.Sequential(Conv2d(n * 3, 8, 7, NL='prelu', same_padding=True, bn=bn))
        self.residual_predict = nn.Sequential(
            Conv2d(56, 16, 7, NL='prelu', same_padding=True, bn=bn),
            Conv2d(16, 8, 5, NL='prelu', same_padding=True, bn=bn),
            Conv2d(8, 3, 3, NL='nrelu', same_padding=True, bn=bn))

        self.rm1 = nn.Sequential(Conv2d(6, 16, 1, NL='relu', same_padding=True, bn=bn))
        self.rm2 = nn.Sequential(Conv2d(6, 8, 3, NL='relu', same_padding=True, bn=bn))
        self.rm3 = nn.Sequential(Conv2d(6, 4, 5, NL='relu', same_padding=True, bn=bn))
        self.residual_merge = nn.Sequential(Conv2d(28, 3, 3, NL='relu', same_padding=True, bn=bn))

        self.ram1 = nn.Sequential(Conv2d(6, 16, 1, NL='relu', same_padding=True, bn=bn))
        self.ram2 = nn.Sequential(Conv2d(6, 8, 3, NL='relu', same_padding=True, bn=bn))
        self.ram3 = nn.Sequential(Conv2d(6, 4, 5, NL='relu', same_padding=True, bn=bn))
        self.res_app_merge = nn.Sequential(Conv2d(28, 3, 3, NL='relu', same_padding=True, bn=bn))

        # self._initialize_weights()

    def forward(self, suport_1, suport_2, app_prediction):

        pair_1 = torch.cat((app_prediction, suport_1), 1)
        pair_2 = torch.cat((app_prediction, suport_2), 1)
        # predict residual maps
        x1_1 = self.r1(pair_1)
        x1_2 = self.r1(pair_2)
        x2_1 = self.r2(pair_1)
        x2_2 = self.r2(pair_2)
        x3_1 = self.r3(pair_1)
        x3_2 = self.r3(pair_2)
        x1 = torch.cat((x1_1, x2_1, x3_1),1)
        x2 = torch.cat((x1_2, x2_2, x3_2),1)
        # pairs = self.residual_predict(torch.cat((x1,x2,x3),1))
        # calc residual based density estimation
        # residual_predictions = pairs + support_gt.squeeze(0).unsqueeze(1)
        pairs_1 = self.residual_predict(x1)
        pairs_2 = self.residual_predict(x2)
        residual_predictions = torch.cat((pairs_1,pairs_2),1)
        # merge residual mals

        x1 = self.rm1(residual_predictions)
        x2 = self.rm2(residual_predictions)
        x3 = self.rm3(residual_predictions)
        final_residual_prediction = self.residual_merge(torch.cat((x1, x2, x3), 1))
        # merge residual and appearance maps
        x1 = self.ram1(torch.cat((final_residual_prediction, app_prediction), 1))
        x2 = self.ram2(torch.cat((final_residual_prediction, app_prediction), 1))
        x3 = self.ram3(torch.cat((final_residual_prediction, app_prediction), 1))
        final_prediction = self.res_app_merge(torch.cat((x1, x2, x3), 1))
        # final_prediction = self.res_app_merge(torch.cat((app_prediction,final_residual_prediction),1))
        return final_prediction


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class _Residual_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(_Residual_Block, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        GRU = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
            GRU.append(ConvGRUCell(input_size=width, hidden_size=width, kernel_size=3))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.ConvGRU = nn.ModuleList(GRU)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = self.ConvGRU[i](sp, sp) + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.num = num
        self.relu = nn.ReLU(inplace=True)
        # self.conv_input_1= nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3, padding=1)
        # self.conv_input_2= nn.Conv2d(in_channels=2,out_channels=16,kernel_size=3, padding=1)
        self.conv_input_3 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        self.res_1 = _Residual_Block(inplanes=16, planes=4)
        self.res_2 = _Residual_Block(inplanes=16, planes=4)
        self.res_3 = _Residual_Block(inplanes=16, planes=4)

        self.conv_up_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.res_4 = _Residual_Block(inplanes=64, planes=16)
        self.res_5 = _Residual_Block(inplanes=64, planes=16)
        self.res_6 = _Residual_Block(inplanes=64, planes=16)

        self.conv_up_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.res_7 = _Residual_Block(inplanes=128, planes=32)
        self.res_8 = _Residual_Block(inplanes=128, planes=32)
        self.res_9 = _Residual_Block(inplanes=128, planes=32)

        ###############上面为encoder,下面为decoder

        self.res_10 = _Residual_Block(inplanes=128, planes=32)
        self.res_11 = _Residual_Block(inplanes=128, planes=32)
        self.res_12 = _Residual_Block(inplanes=128, planes=32)
        self.conv_down_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.res_13 = _Residual_Block(inplanes=64, planes=16)
        self.res_14 = _Residual_Block(inplanes=64, planes=16)
        self.res_15 = _Residual_Block(inplanes=64, planes=16)
        self.conv_down_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.res_16 = _Residual_Block(inplanes=32, planes=8)
        self.res_17 = _Residual_Block(inplanes=32, planes=8)
        self.res_18 = _Residual_Block(inplanes=32, planes=8)
        # self.conv_out_1=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=3, padding=1)
        # self.conv_out_2=nn.Conv2d(in_channels=32,out_channels=2,kernel_size=3, padding=1)

        self.conv_out_3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

        # resdual regression
        self.conv_1_1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0)
        self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        # self.conv_1_3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
        self.rb_1 = Residual()

    def forward(self, input):
        # r,g,b=torch.split(input,1,1)
        output = self.conv_input_3(input)
        output = self.res_3(self.res_2(self.res_1(output)))

        output = self.relu(self.conv_up_1(output))
        output = self.res_6(self.res_5(self.res_4(output)))

        output = self.relu(self.conv_up_2(output))
        output = self.res_9(self.res_8(self.res_7(output)))

        output = self.res_12(self.res_11(self.res_10(output)))
        output_1 = self.conv_down_1(output)
        output = self.relu(output_1)

        output = self.res_15(self.res_14(self.res_13(output)))
        output_2 = self.conv_down_2(output)
        output = self.relu(output_2)

        output = self.res_18(self.res_17(self.res_16(output)))
        output = self.conv_out_3(output) + input

        # features
        output_1 = self.conv_1_1(output_1)
        output_2 = self.conv_1_2(output_2)

        # resudal regression
        output = self.rb_1(output_1, output_2, output)

        return output

