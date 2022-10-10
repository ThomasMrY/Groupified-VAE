import torch
import numpy as np
import itertools
import torch.nn as nn

acrion_scale = 1

def complexfy(self,z):
    z = z[:, :self.z_dim]
    cm_z = self.zcomplex(z)
    return cm_z

def forward_action(self,z,a):
    mu = z[:, :self.z_dim].clone()
    mu[:,a] += acrion_scale
    cm_z = self.complexfy(mu)
    x_recon = torch.sigmoid(self._decode(cm_z))
    return cm_z,x_recon

def backward_action(self,z,a):
    mu = z[:, :self.z_dim].clone()
    mu[:,a] -= acrion_scale
    cm_z = self.complexfy(mu)
    x_recon = torch.sigmoid(self._decode(cm_z))
    return cm_z,x_recon

def action_order_v2(self,x_,a):
    zx = self._encode(x_)
    fcm = self.complexfy(zx)
    cm_z1,x1 = self.forward_action(zx,a)
    z1 = self._encode(x1)
    cm_z2 = self.complexfy(z1)
    fcm1,xr1 = self.backward_action(z1,a)

    cm_z3,x2 = self.backward_action(zx,a)
    z2 = self._encode(x2)
    cm_z4 = self.complexfy(z2)
    fcm2,xr2 = self.forward_action(z2,a)

    return fcm, fcm1,fcm2,cm_z1,cm_z2,cm_z3,cm_z4,xr1,xr2

def abel_action(self,x_,a,b):
    zx = self._encode(x_)
    cm_z1,x1 = self.forward_action(zx,a)
    z1 = self._encode(x1)
    cm_z2 = self.complexfy(z1)
    fcm1,xr1 = self.forward_action(z1,b)

    cm_z3,x2 = self.forward_action(zx,b)
    z2 = self._encode(x2)
    cm_z4 = self.complexfy(z2)
    fcm2,xr2 = self.forward_action(z2,a)

    return fcm1,fcm2,cm_z1,cm_z2,cm_z3,cm_z4,xr1,xr2


###################################                   Isomorphism Loss code                   ###################################


defoalt_dims = [0,1,2,3,4,5]
loss_func = nn.MSELoss()
def constrain_order(net,x,key,mean_dims):
    fcm, fcm1,fcm2,cm_z1,cm_z2,cm_z3,cm_z4,x1,x2 = net.action_order(x,key)
    return loss_func(fcm1[:,mean_dims],fcm[:,mean_dims]) + loss_func(fcm2[:,mean_dims],fcm[:,mean_dims]) + loss_func(cm_z1[:,mean_dims],cm_z2[:,mean_dims]) + loss_func(cm_z3[:,mean_dims],cm_z4[:,mean_dims])
def constrain_abel(net,x,a,b,mean_dims):
    fcm1,fcm2,cm_z1,cm_z2,cm_z3,cm_z4,x1,x2 = net.abel_action(x,a,b)
    return loss_func(fcm1[:,mean_dims],fcm2[:,mean_dims]) + loss_func(cm_z1[:,mean_dims],cm_z2[:,mean_dims]) + loss_func(cm_z3[:,mean_dims],cm_z4[:,mean_dims])
def group_constrains(net,x,mean_dims=defoalt_dims):
    for j,com in enumerate(itertools.combinations(mean_dims, 2)):
        if j == 0:
            abloss = constrain_abel(net,x,com[0],com[1],mean_dims)
        else:
            abloss += constrain_abel(net,x,com[0],com[1],mean_dims)
    for i,key in enumerate(mean_dims):
        if i == 0:
            orloss = constrain_order(net,x,key,mean_dims)
        else:
            orloss += constrain_order(net,x,key,mean_dims)
    return abloss + orloss

def check_dims(net,x):
    with torch.no_grad():
        out = net.encoder(x).squeeze()
        mean = out[:,:10].detach().cpu().numpy()
        logvar = out[:,10:].detach().cpu().numpy()
    kl = 0.5 * np.sum(np.square(mean) + np.exp(logvar) - logvar - 1, 0)
    return np.where(kl>30)

