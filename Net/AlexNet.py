# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:53:50 2019

@author: 15816
"""

from torch import nn

class Alex_Net (nn.Module):
    def __init__(self):
        super(Alex_Net,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,8,kernel_size = 3 ,stride = 1 ,padding = 1),            # (32-3+2)/1+1 = 32   32*32*8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2 , padding = 0)                 # (32-2)/2+1 = 16    16*16*8
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,24, kernel_size=3 , stride = 1, padding = 1) ,          # (16-3+2)/1+1 = 16  16*16*24
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True) ,
            nn.MaxPool2d(kernel_size = 2, stride = 2 , padding = 0)             # (16-2)/2+1 = 8   8*8*24
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(24,32, kernel_size=3 , stride = 1, padding = 1) ,         # (8-3+2)/1+1 = 8  8*8*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) ,
                
            nn.Conv2d(32 ,128, kernel_size=3 , stride = 1, padding =1) ,        # (8-3+2)/1+1 = 8  8*8*64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True) ,
            
            nn.Conv2d(128,256, kernel_size=3 , stride = 1, padding =1) ,        #(8-3+2)/1+1 = 8   8*8*128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True) ,
            nn.MaxPool2d(kernel_size = 2 ,stride=2, padding=0)                  # (8-2)/2+1 = 4   4*4*128  
            )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
            )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4*4*256, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
 
            nn.Dropout(0.5),
            nn.Linear(64, 10)
                )
    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1 , 4*4*256)
        x = self.fc(x)
        return x