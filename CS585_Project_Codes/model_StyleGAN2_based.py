# '''
# Modifications & Additional Layers of StyleGAN2 Based Model.
# '''

from IPython.display import clear_output

import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from customDataLoader import *
from models import *

import sys
sys.path.insert(1, './stylegan2-ada-pytorch')

import dnnlib
import legacy
from torch_utils import misc

device = torch.device('cuda')

class EncoderTop(nn.Module):
    # '''
    # Aux. Encoder Part to be added after VGG16.
    # '''
    def __init__(self, z_dim=512, activation='lrelu'):
        super().__init__()
        
        self.z_dim = z_dim
        self.activation = activation
        
        self.fc1 = FullyConnectedLayer(512 * 7 * 7, 8 * z_dim, activation=activation)
        self.layernorm1 = nn.LayerNorm(8 * z_dim)
        self.fc2 = FullyConnectedLayer(8 * z_dim, 8 * z_dim, activation='tanh')
        self.layernorm2 = nn.LayerNorm(8 * z_dim)
        self.fc3 = FullyConnectedLayer(8 * z_dim, 8 * z_dim, activation=activation)
        self.fc4 = FullyConnectedLayer(8 * z_dim, 8 * z_dim, activation=activation)
        self.layernorm4 = nn.LayerNorm(8 * z_dim)
        self.fc5 = nn.Linear(8*z_dim, z_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.fc2(x)
        x = self.layernorm2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.layernorm4(x)
        x = self.fc5(x)
        
        return x


class VGG16_Encoder(nn.Module):
    # '''
    #     Loading VGG16 and add the Encoder Top
    # '''
    def __init__(self, model_path='./stylegan2-ada-pytorch/pretrained/vgg16.pt',
                 z_dim=512, activation='lrelu'):
        super().__init__()
        
        self.z_dim = z_dim
        self.activation = activation
        
        self.vgg16 = PerceptualModel(model_path=model_path)
        self.top_layers = EncoderTop(z_dim=z_dim, activation=activation)
    
    def forward(self, x, training=False):
        x = self.vgg16(x)
        x = self.top_layers(x)        
        return x


class Manipulation(nn.Module):
    # '''
    # Combine StyleGAN2 generator and VGG16 encoder for attribute manipulation
    # '''
    def __init__(self, encoder_path=None, copy_encoder_top=True, num_attr=40, device=torch.device('cuda')):
        super().__init__()

        network_pkl = './training-runs/00001--auto1-noaug-resumecustom/network-snapshot-000120.pkl'
        print(f'Loading networks from "{network_pkl}"...')

        with dnnlib.util.open_url(network_pkl) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].eval().requires_grad_(False)  # type: ignore

        self.z_dim = self.G.z_dim
        self.num_attr = num_attr

        self.encoder = VGG16_Encoder(z_dim=self.z_dim)
        self.encoder_top = EncoderTop()

        if encoder_path is not None:
            print(f'Loading encoder from "{encoder_path}"...')
            self.encoder.load_state_dict(torch.load(encoder_path), strict=False)
            if copy_encoder_top:
                self.encoder_top.load_state_dict(self.encoder.top_layers.state_dict().copy())

        self.encoder = self.encoder
        self.encoder_top = self.encoder_top

        _ = np.linalg.svd(np.random.normal(size=[self.z_dim] * 2))
        _ = _[0] @ _[-1][:, :self.num_attr]
        self.attr_directions = torch.nn.Parameter(torch.from_numpy(_).to(torch.float))
        self.project_attr_directions_to_orth()

        _ = np.linalg.svd(np.random.normal(size=[self.z_dim] * 2))
        _ = _[0] @ _[-1][:, :self.num_attr]
        self.attr_inverse = torch.nn.Parameter(torch.from_numpy(_).to(torch.float))

    def project_attr_directions_to_orth(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'attr_directions' in name:
                    _ = torch.linalg.svd(param.cpu().detach())
                    _ = _[0] @ torch.eye(*param.shape) @ _[-1]
                    param.copy_(_)

    def forward(self, target_images, target_labels):
        embs_t = self.encoder(target_images)

        idnt, attr_r = self.get_idnt_attr_from_images(target_images)
        embs_r = idnt + attr_r

        idnt, attr_m = self.manipulation(target_images, target_labels)
        embs_m = idnt + attr_m

        return embs_t, embs_r, embs_m

    def get_idnt_attr_from_images(self, target_images):
        embs = self.encoder(target_images)
        idnt = self.encoder_top(self.encoder.vgg16(target_images))

        idnt_normed = idnt / torch.norm(idnt, dim=1).unsqueeze(1).repeat([1, idnt.shape[-1]])
        projection = (idnt_normed * embs).sum(1).unsqueeze(1).repeat([1, idnt.shape[-1]]) * idnt_normed
        attr = embs - projection
        return idnt, attr

    def classifier(self, target_images, return_logit=False):
        _idnt, attr = self.get_idnt_attr_from_images(target_images)
        logit = attr @ self.attr_directions

        if return_logit:
            return logit
        else:
            return nn.functional.sigmoid(logit)

    def manipulation(self, target_images, target_labels, return_embs=False):
        idnt, attr_r = self.get_idnt_attr_from_images(target_images)
        norm_idnt = torch.norm(idnt, dim=1)

        weights = nn.functional.tanh(idnt @ self.attr_inverse) * (2 * target_labels - 1)
        attr = (weights @ self.attr_directions.t()) + 1e-3 * idnt

        scale = (attr * idnt).sum(1) / (norm_idnt * (1 - norm_idnt))
        attr_m = scale.unsqueeze(1) * attr

        if return_embs:
            embs = idnt + attr_m
            return embs
        else:
            return idnt, attr_r, attr_m

    def synthesis(self, input_embeddings):
        synth_images = self.G.synthesis(
            input_embeddings.unsqueeze(1).repeat([1, self.G.mapping.num_ws, 1]),
            noise_mode='const'
        )
        synth_images = (synth_images + 1) * (255 / 2)
        return synth_images
