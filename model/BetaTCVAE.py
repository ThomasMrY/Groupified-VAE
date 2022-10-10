import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import gin


def reparametrize(mu, logvar):
    std = logvar.exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.contiguous().view(self.size)

@gin.configurable
class BetaTCVAE(nn.Module):
    """Model proposed in Isolating Sources of Disentanglement in Variational Autoencoders paper(Chen et al, arxiv:1802.04942, 2019)."""

    def __init__(self, z_dim=10, nc=1, N=10, group = True):
        super(BetaTCVAE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        self.N = N
        self.group = group
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        if self.group:
            decode_dim = 2*z_dim
        else:
            decode_dim = z_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),          # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        
        if self.group:
            real = torch.sin(2*np.pi*z/self.N)
            imag = torch.cos(2*np.pi*z/self.N)
            cm_z = torch.cat([real,imag],dim=1)

            x_recon = self._decode(cm_z).view(x.size())
        else:
            x_recon = self._decode(z).view(x.size())
        return x_recon, mu, logvar, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)
    
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params.cuda()
    def zcomplex(self, z):
        real = torch.sin(2*np.pi*z/self.N)
        imag = torch.cos(2*np.pi*z/self.N)
        return torch.cat([real,imag],dim=1)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



def gaussian_log_density(samples, mean, log_var):
    pi = torch.tensor(np.pi)
    normalization = Variable(torch.Tensor([np.log(2 * np.pi)])).cuda()
    inv_sigma = torch.exp(-log_var)
    tmp = (samples - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2* log_var + normalization)

def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def total_correlation(net,beta, z, z_mean, z_logvar):
    """Estimate of total correlation on a batch.

    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)

    Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.

    Returns:
    Total correlation estimated on a batch.
    """
    prior_params = net._get_prior_params(z.shape[0])
    logpz = gaussian_log_density(z, prior_params[:,:,0],prior_params[:,:,1]).view(z.shape[0], -1).sum(1)
    log_qz_prob = gaussian_log_density(
        z.unsqueeze(1), z_mean.unsqueeze(0),
        z_logvar.unsqueeze(0))
    logqz_prodmarginals = (logsumexp(log_qz_prob, dim=1, keepdim=False)).sum(1)
    logqz = (logsumexp(log_qz_prob.sum(2), dim=1, keepdim=False))
    return -beta*(logqz - logqz_prodmarginals) - (logqz_prodmarginals - logpz)