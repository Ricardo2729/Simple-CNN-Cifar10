# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:53:05 2019

@author: 15816
"""

from torch import nn

class Le_Net(nn.Module):
    def __init__(self):
        super(Le_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,kernel_size = 5 ,stride = 1 ,padding = 0),            #(32-5)/1 + 1 = 28  28*28*6
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0 )                    #(28-2)/2+1 = 14  14*14*6
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size =5,stride=1,padding=0),                  #(14-5)/1+1 = 10, 10*10*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0 )                    #(10-2)/2+1 = 5  5*5*16
            )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2
            )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU(),
            
            nn.Linear(120, 84),
            nn.ReLU(),
            
            nn.Linear(84, 10)
            )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)
        return x
