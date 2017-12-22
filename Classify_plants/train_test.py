#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/12/5
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

from model import CNN
#数据集地址
train_path='F:/yzy/data/train'
test_path='F:/yzy/data/test'

#超参数
num_epochs = 1
batch_size = 8
learning_rate = 0.001

#获得数据并转化为Tensor
train_dataset = dsets.ImageFolder(root=train_path,
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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True
                                           )

test_dataset = dsets.ImageFolder(root=test_path,
                                 #transform=transforms.ToTensor()
                                  transform=transforms.Compose([
                                      transforms.Scale(128),
                                      transforms.CenterCrop(128),
                                      transforms.ToTensor(),
                                     # transforms.Normalize((0.5, 0.5, 0.5),
                                        #                   (0.5, 0.5, 0.5)),
                                  ])
                                  )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False
                                           )

cnn = CNN()
cnn = cnn.cuda()
print(cnn)

if __name__ == '__main__':
    #误差和优化
    loss_fun = nn.CrossEntropyLoss()
    loss_fun = loss_fun.cuda()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

    #训练模型
    #cnn.train()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            #print(labels)
            labels = Variable(labels.cuda())
            #print(labels)
            #前向传播 反向传播 优化
            optimizer.zero_grad()
            # output = cnn(images)
            output = cnn.forward(images)
            loss = loss_fun.forward(output, labels)
            # loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()
            #记录loss和batchSize数据对应的分类准确数量
            #train_loss += loss.data[0]
            #_,predict = torch.max(output, 1)
            #correct_num = (predict == labels).sum()
            #train_acc = correct_num.data[0]
            if (i + 1) % 1 == 0:
        # train_loss /= len(train_dataset)
        # train_acc /= len(train_dataset)
        # print("[%d/%d] Loss: %.5f, Acc: %.2f"
        #               % (i + 1, num_epochs, train_loss, 100 * train_acc))
                print('训练次数 [%d/%d], 步数 [%d/%d], 误差： %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

#模型测试
cnn.eval()
correct = 0
total = 0
for epoch in range(num_epochs):
    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = labels.cuda()
        #labels = Variable(labels)
        output = cnn(images)
        value,predict = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
        #
        # labels = Variable(labels)
        # loss = loss_fun(images, labels)
        # print('loss %.4f' %loss.data[0]) #bug
    print('测试准确度： %d %%' % (100* correct / total))

#存储模型
#torch.save(cnn, '01cnn_gloable.pkl')
#torch.save(cnn.state_dict(), '01cnn_parament.pkl')
