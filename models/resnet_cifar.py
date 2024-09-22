import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
import numpy as np
from .ETF_classifier import ETF_Classifier

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet_pu(nn.Module):
    def __init__(self, block, num_blocks, no_class=10, batch_norm=False,try_assert=True):
        super(ResNet_pu, self).__init__()
        self.in_planes = 64

        self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2   = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4   = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier_new = ETF_Classifier(feat_in=512*block.expansion, num_classes=no_class,try_assert=try_assert)
        self.weight = nn.Parameter(torch.Tensor(512*block.expansion, no_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(512*block.expansion,512*block.expansion),
            nn.ReLU(inplace=True),
            nn.Linear(512*block.expansion, 128)
        )
        self.fc1 = nn.Linear(512*block.expansion, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)
        self.LogSoftMax = nn.LogSoftmax(dim=1)
        self.af = F.relu
        self.no_class = no_class
        self.feat_in = 512*block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def reset_etf(self):
        self.classifier_new = ETF_Classifier(feat_in=self.feat_in, num_classes=self.no_class,try_assert=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        cur_M = self.classifier_new.ori_M.cuda()
        feat_norm = self.classifier_new(out)
        logit = torch.matmul(feat_norm, cur_M)
        feat_con = self.contrastive_head(feat_norm)
        feat_con = F.normalize(feat_con,dim=1)
 
        h = self.fc1(out.detach().clone())
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        h_final =  self.LogSoftMax(h)
 
        return out, logit, feat_norm , feat_con,h_final

def resnet18_pu(**kwargs):
    return ResNet_pu(BasicBlock, [2, 2, 2, 2], **kwargs)
