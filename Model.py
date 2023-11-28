import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU()
                #torch.nn.MaxPool2d(kernel_size=2,stride=2)
                
                
            )
        self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 512, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1),
                torch.nn.Sigmoid()
                #torch.nn.Upsample(scale_factor=2, mode='nearest')
            )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def get_encoder_state_dict(self):
        return self.encoder.state_dict()
    
    def bigmanting(self, x):
        out=self.encoder(x)
        return out


class AE2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU()
 
            )
        self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 3, kernel_size=3,stride=1,padding=1),
                torch.nn.Sigmoid()
                #torch.nn.Upsample(scale_factor=2, mode='nearest')
            )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    def get_encoder_state_dict(self):
        return self.encoder.state_dict()



class segmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1)
        self.ReLU1=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(32, 64, kernel_size=3,stride=1,padding=1)
        self.ReLU2=torch.nn.ReLU()
        self.conv3=torch.nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1)
        self.ReLU3=torch.nn.ReLU()
        self.conv4=torch.nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1)
        self.ReLU4=torch.nn.ReLU()
        self.conv5=torch.nn.Conv2d(256, 1, kernel_size=1)
        self.ReLU5=torch.nn.ReLU()
        self.Sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x,intermediate_outputs):
        
        x=x+intermediate_outputs[9]
        x=self.conv1(x)
        x=self.ReLU1(x)
        
        x=x+intermediate_outputs[7]
        x=self.conv2(x)
        x=self.ReLU2(x)
        
        x=x+intermediate_outputs[5]
        x=self.conv3(x)
        x=self.ReLU3(x)
        
        x=x+intermediate_outputs[3]
        x=self.conv4(x)
        x=self.ReLU4(x)
        
        x=x+intermediate_outputs[1]
        x=self.conv5(x)
        #x=self.ReLU5(x),
        x=self.Sigmoid(x)
        return x
        
class BatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        
    def forward(self, x,features):
        norm=nn.BatchNorm2d(features)
        norm=norm.to('cuda')
        x=norm(x)
        return x

class reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,odim,targ_dim):
        conv=nn.Conv2d(odim,targ_dim,kernel_size=1)
        conv=conv.to('cuda')
        x=conv(x)
        return x
    
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.skip_residual_blocks = nn.ModuleList([ResidualBlock(in_channels, out_channels) for _ in range(depth)])
        self._reshape=reshape()
        self._segmenter=segmenter()
        self._segmenter=self._segmenter.to('cuda')
        self._BatchNorm=BatchNorm()
        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 256, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 64, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 32, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, kernel_size=3,stride=1,padding=1),
                torch.nn.ReLU()
                #torch.nn.MaxPool2d(kernel_size=2,stride=2)
                
                
        )
        self.intermediate_outputs = []
        self.residual=torch.nn.Sequential(
            torch.nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,256,kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU()
        )
        self.bottleneck=torch.nn.Sequential(
            torch.nn.Conv2d(128,16,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16,16,kernel_size=1),
            torch.nn.ReLU()
        )
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._ReLU=nn.ReLU()
        self.classifier=torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(96*96*1,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,10),
            torch.nn.LogSoftmax(dim=1)
        )
        self.hookhandles=[]
    def hook(self, module, input, output):
        output=output.to('cuda')
        self.intermediate_outputs.append(output)
        
    def forward(self,x):
        for layer in self.encoder:
            hook_handle=layer.register_forward_hook(self.hook)
            self.hookhandles.append(hook_handle)
        
        
        x=self.encoder(x)
        
        for i in range(0,len(self.intermediate_outputs),2):
            self.intermediate_outputs[i]=self._BatchNorm(self.intermediate_outputs[i],self.intermediate_outputs[i].size(1))
            self.intermediate_outputs[i]=self._ReLU(self.intermediate_outputs[i])
        
        x=self._reshape(x,16,128)
        for i in range(3):
            res_skip=x
            x=self.residual(x)
            x=self._BatchNorm(x,128)
            x=x+res_skip
        
        x=self.bottleneck(x)
        x=self._segmenter(x,self.intermediate_outputs)
        out=self.classifier(x)
        self.intermediate_outputs.clear()
        for hook_handle in self.hookhandles:
            hook_handle.remove()
        return out