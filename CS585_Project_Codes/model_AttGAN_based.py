#####################################################################
# Library import
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

from torchsummary import summary
import sys
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

import argparse
import datetime
import json
import os
from os.path import join

import torch.utils.data as data

import torch
import torchvision.utils as vutils
# from attgan_modified import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter

#####################################################################
# Model part
MAX_DIM = 64 * 16  # 1024


# Refactor the code to produce the wanted results
class Modifier(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Modifier, self).__init__()

        self.generator = Generator(
            enc_dim=enc_dim, enc_layers=enc_layers, enc_norm_fn=enc_norm_fn, enc_acti_fn=enc_acti_fn,
            dec_dim=dec_dim, dec_layer=dec_layers, dec_norm_fn=dec_norm_fn, dec_acti_fn=dec_acti_fn,
            n_attrs=n_attrs, shortcut_layers=shortcut_layers, inject_layers=inject_layers, img_size=img_size
        )

        self.encoder_id = Encoder(enc_dim=enc_dim, enc_layers=enc_layers, enc_norm_fn=enc_norm_fn,
                                  enc_acti_fn=enc_acti_fn)

    def _get_attr(self, x):
        emb_w = self.generator.encoder(x)
        emb_id = self.encoder_id(x)
        attr = emb_w - emb_id
        ## you can insert e.g. MLP here attr = MLP(attr)
        return attr

    def forward(self, x, a):
        return self.generator(x, a)


class Encoder(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu'):
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

    def forward(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs


class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2 ** enc_layers  # f_size = 4 for 128x128

        self.encoder = Encoder(enc_dim=enc_dim, enc_layers=enc_layers, enc_norm_fn=enc_norm_fn, enc_acti_fn=enc_acti_fn)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in // 2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)

    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                    .repeat(1, 1, self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1))
                z = torch.cat([z, a_tile], dim=1)
        return z

    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encoder(x), a)
        if mode == 'enc':
            return self.encoder(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2 ** n_layers

        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 5, 'none', 'none')
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)


# Compute Info NCE loss
def info_nce_loss(anchor, positive, temperature=0.07):
    anchor = anchor.view(anchor.size(0), -1)
    positive = positive.view(positive.size(0), -1)

    batch_size = anchor.size(0)
    scores = torch.matmul(anchor, positive.t()) / temperature
    target = torch.arange(batch_size).to(anchor.device)
    loss = F.cross_entropy(scores, target)
    return loss


class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp

        self.M = Modifier(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )

        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4,
                device='cuda' if args.gpu else 'cpu')

        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)

    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        d_fake, dc_fake = self.D(img_fake)

        # Obtain the embedding of the images
        za_temp = self.M(img_a)
        emb1 = self.M._get_attr(za_temp)
        emb2 = self.M._get_attr(d_fake)

        # Calculate InfoNCE Loss
        infonce_loss = info_nce_loss(emb1, emb2)

        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)
        gr_loss = F.l1_loss(img_recon, img_a)

        # Add the loss to the current generators training
        # g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss + self.lambda_infonce * infonce_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item(),
            'infonce_loss': infonce_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True

        img_fake = self.G(img_a, att_b_).detach()
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()

        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item(),
        }
        return errD

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
            print("G loaded")
        if 'D' in states:
            self.D.load_state_dict(states['D'])
            print("D loaded")
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
            print("Optim G loaded")
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
            print("Optim D loaded")

        print("Load pretrained successfully!")

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)


#############################################################################################
# Train part


attrs_default = [
    'Black_Hair', 'Blond_Hair', 'Straight_Hair', 'Wavy_Hair', 'Male',
]


def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/list_attr_celeba.txt')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')

    parser.add_argument('--b_distribution', dest='b_distribution', default='none',
                        choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')

    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

    return parser.parse_args(args)


args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CelebA':
    from data import CelebA

    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
if args.data == 'CelebA-HQ':
    from data import CelebA_HQ

    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

attgan = AttGAN(args)
progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float)
sample_att_b_list = [fixed_att_a]
for i in range(args.n_attrs):
    tmp = fixed_att_a.clone()
    tmp[:, i] = 1 - tmp[:, i]
    tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
    sample_att_b_list.append(tmp)

it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    writer.add_scalar('LR/learning_rate', lr, it + 1)
    for img_a, att_a in progressbar(train_dataloader):
        attgan.train()

        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()

        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)

        att_a_ = (att_a * 2 - 1) * args.thres_int
        if args.b_distribution == 'none':
            att_b_ = (att_b * 2 - 1) * args.thres_int
        if args.b_distribution == 'uniform':
            att_b_ = (att_b * 2 - 1) * \
                     torch.rand_like(att_b) * \
                     (2 * args.thres_int)
        if args.b_distribution == 'truncated_normal':
            att_b_ = (att_b * 2 - 1) * \
                     (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * \
                     (2 * args.thres_int)

        if (it + 1) % (args.n_d + 1) != 0:
            errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errD, it + 1, 'D')
        else:
            errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errG, it + 1, 'G')
            progressbar.say(epoch=epoch, iter=it + 1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])

        if (it + 1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            attgan.saveG(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
            # attgan.save(os.path.join(
            #     'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it + 1) % args.sample_interval == 0:
            attgan.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                    if i > 0:
                        att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
                    samples.append(attgan.G(fixed_img_a, att_b_))
                samples = torch.cat(samples, dim=3)
                writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it + 1)
                vutils.save_image(samples, os.path.join(
                    'output', args.experiment_name, 'sample_training',
                    'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it % it_per_epoch + 1, it_per_epoch)
                ), nrow=1, normalize=True, range=(-1., 1.))
        it += 1