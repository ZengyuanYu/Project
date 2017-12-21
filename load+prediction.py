#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/12/9
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage import io,transform
import glob
import os
import numpy as np
import time
import cv2

from model import CNN

path = "D:/test"
dict = {0:'菊花',1:'玫瑰',2:'向日葵',3:'郁金香',4:'何首乌',5:'决明子'}
w=128
h=128
c=3

#获得数据并转化为Tensor
predict_data = dsets.ImageFolder(root=path,
                                  #transform=transforms.ToTensor()
                                  transform=transforms.Compose([
                                    #改变图片的size，标准化,
                                      transforms.Scale(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                      #transforms.Normalize((0.5, 0.5, 0.5),
                                                      #     (0.5, 0.5, 0.5)),
                                  ])
                                  )

predict_loder = torch.utils.data.DataLoader(dataset=predict_data,
                                            batch_size=1,
                                            shuffle = False,
                                           )
cnn = torch.load('01cnn_gloable.pkl')

for images, labels in predict_loder:
    images = Variable(images).cuda()
    labels = labels.cuda()
    output = cnn(images)
    value, predict = torch.max(output.data, 1)
    print(predict)
    print(predict.size())
    a = predict.cpu().numpy()
    print(a)
    b = a[0,0]
    print(b)
    print(dict[b])
#def read_one_image(path):
 #   img = io.imread(path)
  #  img = transform.resize(img, (w,h))
   # return torch.from_numpy(img).unsqueeze(0)

#def read_one_image_numpy(path):
 #   img = io.imread(path)
  #  img = transform.resize(img, (w,h))
   # return img
#image_numpy = read_one_image_numpy(path)
#cv2.imshow('d',image_numpy)

#image = Variable(read_one_image(path)).float().transpose(1,3).cuda()
#print('输入： ',image)

#cnn = torch.load('01cnn_gloable.pkl')
#predict = cnn(image)
#print('输出： ',predict)
#索引值
#_,index = torch.max(predict,1)
#print('预测值： ',_,'索引值： ', index)
#i = index.data[0]
#print(flower_dict[i])
#print(index.size())
#print(index.data[0])
#cv2.waitKey()