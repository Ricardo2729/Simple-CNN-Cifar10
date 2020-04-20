# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:13:02 2019

@author: 15816
"""
import torch
from torch import nn
import torch.nn.functional as F

#将展平创建一个类 ，只有类才能放进sequential里
class Flatten(nn.Module):
    def __init__(self):
        super (Flatten,self).__init__()
    def forward(self,input):
        return input.view(input.size(0),1)

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1):
        super(ResidualBlock,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride = stride,padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride = 1 ,padding = 1),
            nn.BatchNorm2d(out_channels),
            )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2
            )
        self.shortcut = nn.Sequential()
        #如果输入卷积核不等于输出卷积核，在shortcut时进行1*1卷积使维度相同
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        out = self.conv(x)
        out = self.shortcut(x)+out
        out = F.relu(out)
        return out
    
class Res_Net(nn.Module):
    def __init__(self):
        super(Res_Net,self).__init__()
        self.in_channels = 64
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1 ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.layer1 = ResidualBlock(64, 128,stride = 2) 
        self.layer2 = ResidualBlock(128,256,stride = 2)
        self.layer3 = ResidualBlock(256,512,stride = 2)
        self.layer4 = ResidualBlock(512,1024,stride = 2)
        self.fc = nn.Linear(1024 , 10)

    def forward(self, x):
        x = self.layer(x) #32*32*3  => 16*16*64
        x = self.layer1(x) #16*16*64  => 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
#        print('after conv:', x.shape)
        x = F.adaptive_avg_pool2d(x, [1,1]) #不论输入什么像素，输出都是1*1*channel
#        print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(128, 3, 32, 32)
    model = Res_Net()
    out = model(x)
    print('resnet:', out.shape)

