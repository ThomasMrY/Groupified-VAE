"""model.py"""

import torch.nn as nn
import torch.nn.init as init
import gin
import torch
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()

@gin.configurable
class FactorVAE1(nn.Module):
    """Model proposed in Disentangling by factorising paper(Kim et al, arxiv:1802.05983, 2019)."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.omega_0 = 30
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        if group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decode_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )
        self.weight_init()
        if group:
            self.encoder[10].weight.data.uniform_(-np.sqrt(6 / 128), 
                                                np.sqrt(6 / 128))
            self.decoder[0].weight.data.uniform_(-np.sqrt(6 / decode_dim) / self.omega_0, 
                                                np.sqrt(6 / decode_dim) / self.omega_0)
            # pass

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        elif mode == 'sine':
            initializer = sine_init
        

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        elif self.group:
            cm_z = self.zcomplex(z)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def zcomplex(self, z):
        real = self.omega_0*torch.sin(2*np.pi*z/self.N)
        imag = self.omega_0*torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)

class FactorVAE2(nn.Module):
    """Model proposed in Disentangling by factorising paper(Kim et al, arxiv:1802.05983, 2019)."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        if group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decode_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        elif self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class FactorVAE3(nn.Module):
    """Encoder and Decoder architecture for 3D Faces data."""
    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(FactorVAE3, self).__init__()
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 2*z_dim, 1)
        )
        if group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decode_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        elif mode == 'sine':
            initializer = sine_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encoder(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        elif self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self.decoder(cm_z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def sine_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        m.weight.data.uniform_(-np.sqrt(6 / m.in_channels), 
                                                np.sqrt(6 / m.in_channels))
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)