import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
import numpy as np

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False, try_assert=True):
        super(ETF_Classifier, self).__init__()
        #ETF Initialization
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes,try_assert)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        if(num_classes==10):
            M = torch.load('./pretrained/etf_initial/cf10.pkl')
        elif(num_classes==100):
            M = torch.load('./pretrained/etf_initial/cf100.pkl')
        self.ori_M = M.cuda()
        self.ori_M.requires_grad_(False)
        self.LWS = LWS
        self.reg_ETF = reg_ETF
        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes,try_assert):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        if try_assert:
            assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x