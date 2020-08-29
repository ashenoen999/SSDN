# -*- coding: utf-8 -*-


from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
# from CNN_LSTM import VGG
import numpy as np
import torch.nn as nn
import pdb
import os
from math import log10
import sys
from srresnet_6_3d import Net
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, default="weight/indoor.pth",help='model file to use')
# parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')

# parser.add_argument('--input_file', type=str, required=True, help='input image to use')
# parser.add_argument('--input_image_1', type=str, required=True, help='input image to use')
# parser.add_argument('--input_image_2', type=str, required=True, help='input image to use')
# parser.add_argument('--input_image_4', type=str, required=True, help='input image to use')
# parser.add_argument('--input_image_5', type=str, required=True, help='input image to use')

opt = parser.parse_args()
criterionG = nn.MSELoss()
x1 = 0
y1 = 0
w = 1280
h = 720
# 1280,720

file_name_out = 'out_cuda3'

# if os.path.isdir(file_name_out)== False:
#     os.mkdir(file_name_out)
file = './images'
# gt_file = 'C:\\Users\\allen\\Desktop\\experments_data\\paper_pic\\ex\\GT'
model = torch.load(opt.model)
different_file = os.listdir(os.path.join(file))
# different_file.sort()

for j in range(len(different_file)):
    # img = os.path.join(opt.input_file, "HSTS/real-world", different_file[j])
    img = os.path.join(file, different_file[j])
    # GT_img = os.path.join(gt_file, different_file[j])
    # GT_img = Image.open(GT_img).convert('RGB')
    # GT_img = Variable(ToTensor()(GT_img))
    # print(opt)
    img = Image.open(img).convert('RGB')
    # img = img.crop(((x1, y1, x1 + w, y1 + h)))

    # model = torch.load(opt.model)
    input = Variable(ToTensor()(img)).view(1, 3, img.size[1], img.size[0])

    if opt.cuda:
        model = model.cuda()
        input = input.cuda()
    try:
        with torch.no_grad():
            # r_out = model(input)
            out = model(input)
    except RuntimeError:
        print("out of memoery")
        continue


    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_cy = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = Image.fromarray(np.uint8(out_img_y[1]), mode='L')
    out_img_cr = Image.fromarray(np.uint8(out_img_y[2]), mode='L')
    out_img = Image.merge('RGB', [out_img_cy, out_img_cb, out_img_cr])

    out_img.save(os.path.join("./output/", different_file[j]))
    print('output image saved to ', os.path.join("/output/", different_file[j]))


