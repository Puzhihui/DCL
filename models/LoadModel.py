import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
from transformers import BertTokenizer, BertModel

from config import pretrained_model, customize_model

import pdb

from .EfficientNet_my import EfficientNet
from .spectral import SpectralNorm
from .sagan_models import Self_Attn
class Efficientb4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class Efficient_bx(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.model = EfficientNet.from_pretrained(backbone)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.use_sagan = config.use_sagan
        self.use_language = config.use_language
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        elif self.backbone_arch in customize_model:
            pass
        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            linear_size = 2048
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            linear_size = 2048
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'efficientnet-b4':
            self.model = Efficientb4(num_classes=self.num_classes)
            linear_size = 1792
        if self.backbone_arch == 'efficientnet-b1':
            self.model = Efficient_bx('efficientnet-b1', num_classes=self.num_classes)
            linear_size = 1280
        if self.backbone_arch == 'efficientnet-b3':
            self.model = Efficient_bx('efficientnet-b3', num_classes=self.num_classes)
            linear_size = 1536
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(linear_size, self.num_classes, bias=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(linear_size, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(linear_size, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(linear_size, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(linear_size, self.num_classes, bias=False)

        if self.use_sagan:
            self.imsize = 128
            conv_dim = 128
            layer1 = []
            layer2 = []
            layer3 = []
            last = []

            layer1.append(SpectralNorm(nn.Conv2d(linear_size, conv_dim, 4, 2, 1)))
            layer1.append(nn.LeakyReLU(0.1))

            curr_dim = conv_dim

            layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer2.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2

            layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer3.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2

            if self.imsize == 64:
                layer4 = []
                layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
                layer4.append(nn.LeakyReLU(0.1))
                self.l4 = nn.Sequential(*layer4)
                curr_dim = curr_dim * 2
            self.l1 = nn.Sequential(*layer1)
            self.l2 = nn.Sequential(*layer2)
            self.l3 = nn.Sequential(*layer3)

            last.append(nn.Conv2d(curr_dim, 1, 1))
            self.last = nn.Sequential(*last)

            self.attn1 = Self_Attn(256, 'relu')
            self.attn2 = Self_Attn(512, 'relu')

        if self.use_language:
            self.bert_model = BertModel.from_pretrained('/data1/pzh/project/local/hugging_face/bert-base-uncased')
            self.fuse_linear = nn.Linear(linear_size+768, linear_size, bias=False)

    def forward(self, x, input_ids=None, attention_mask=None, last_cont=None):
        if self.use_language:
            text_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_outputs.last_hidden_state.mean(dim=1)  # 取最后一个隐藏层的平均值作为特征表示 (batch, 768)
            # text_features = text_outputs.last_hidden_state[0]  # 获取 [CLS] token 的嵌入
        x = self.model(x)  # 8,1792,14,14
        x_sagan = x
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.use_language:
            combined_features = torch.cat((x, text_features), dim=1)
            fused_features = self.fuse_linear(combined_features)
            x = fused_features
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            if self.use_sagan:
                out_sagan = self.l1(x_sagan)
                out_sagan = self.l2(out_sagan)
                # out_sagan = self.l3(out_sagan)
                out_sagan, p1 = self.attn1(out_sagan)
                out_sagan = self.l3(out_sagan)
                # out_sagan = self.l4(out_sagan)
                out_sagan, p2 = self.attn2(out_sagan)
                out_sagan = self.last(out_sagan)
                out.append(out_sagan)
            else:
                out.append(self.classifier_swap(x))  # x：8,1792 out：8, 4
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
