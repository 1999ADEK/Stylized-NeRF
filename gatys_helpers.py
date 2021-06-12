import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torchvision.transforms.functional as tF
import torchvision

import cv2
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
VGG19_PATH = "/content/drive/MyDrive/NeRF/nerf-pytorch/models/vgg19.pth"

mse = lambda x, y : torch.mean((x - y) ** 2)

def get_style_tensor(style_path, size=0):
    img = Image.open(style_path)
    W, H = img.size
    img_size = tuple([int((float(size) / max([H,W]))*x) 
                     for x in [H, W]]) if size else (H, W)
    img = tF.resize(img, img_size)
    img = tF.to_tensor(img)
    return img

def get_normalized_tensor(tensor):
    tensor = tF.normalize(255*tensor[:,(2, 1, 0)], [103.939, 116.779, 123.68], [1, 1, 1])
    return tensor

def load_pretrained_vgg():
    vgg = models.vgg19(pretrained=False)
    vgg.load_state_dict(torch.load(VGG19_PATH), strict=False)
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg.features.to(device)

layers = {
        '3': 'relu1_2',   # Style layers
        '8': 'relu2_2',
        '17' : 'relu3_3',
        '26' : 'relu4_3',
        '35' : 'relu5_3',
        '22' : 'relu4_2', # Content layers
    }

def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(C, H*W)
    return torch.mm(x, x.t())

class StyleTransferLoss():
    def __init__(self, w_content, w_style, w_tv, style_tensor):
        self.w_content = w_content
        self.w_style = w_style
        self.w_tv = w_tv
        self.content_layers = ['relu4_2']
        self.content_weights = {'relu4_2': 1.0} 
        self.style_layers = ['relu1_2', 
                             'relu2_2', 
                             'relu3_3', 
                             'relu4_3', 
                             'relu5_3']
        self.style_weights = {l: 0.2 for l in self.style_layers}
        self.vgg = load_pretrained_vgg()
        self.style_features = self.get_features(style_tensor.to(device).unsqueeze(0))

    def __call__(self, predict_tensor, content_tensor):
        content_features = self.get_features(content_tensor)
        predict_features = self.get_features(predict_tensor)
        content_loss = 0
        style_loss = 0
        for j in self.content_layers:
            content_loss += self.content_weights[j] * \
                            self.content_loss(predict_features[j], content_features[j])
        for j in self.style_layers:
            style_loss += self.style_weights[j] * \
                            self.style_loss(predict_features[j], self.style_features[j])
        tv_loss = self.tv_loss(predict_tensor.clone().detach())
        total_loss = self.w_content * content_loss + \
               self.w_style * style_loss + \
               self.w_tv * tv_loss
        loss = {'content': content_loss,
                'style': style_loss,
                'tv': tv_loss,
                'total': total_loss}
        return loss

    def get_features(self, tensor):
        features = {}
        x = get_normalized_tensor(tensor)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                if (name=='22'):   # relu4_2
                    features[layers[name]] = x
                elif (name=='31'): # relu5_2
                    features[layers[name]] = x
                else:
                    b, c, h, w = x.shape
                    features[layers[name]] = gram(x) / (h*w)
                    
                # Terminate forward pass
                if (name == '35'):
                    break 
        return features
    
    def style_loss(self, predict, target):
        C = predict.size(1)
        loss = mse(predict, target) / (C**2)
        return loss

    def content_loss(self, predict, target):
        return mse(predict, target)

    def tv_loss(self, x):
        x1 = x[:,:,1:,:] - x[:,:,:-1,:]
        x2 = x[:,:,:,1:] - x[:,:,:,:-1]
        loss = torch.sum(torch.abs(x1), dim=(1,2,3)) + \
               torch.sum(torch.abs(x2), dim=(1,2,3))
        return torch.mean(loss)