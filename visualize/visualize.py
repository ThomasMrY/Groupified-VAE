import os
import gin
import matplotlib.pyplot as plt
import matplotlib

from data.ground_truth import named_data
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import PIL


@gin.configurable
class Visualizer():
    def __init__(self,name):
        self.name = name
        with gin.unlock_config():
            gin.bind_parameter("dataset.name", name)
        self.dataset = named_data.get_named_ground_truth_data()

    def get_latent_matrix():
        com_matrix = np.zeros((3,6*32*32))
        for k in range(6):
            for i in range(32):
                for j in range(32):
                    com_matrix[0,k*32*32+i*32 + j] = k
                    com_matrix[1,k*32*32+i*32 + j] = i
                    com_matrix[2,k*32*32+i*32 + j] = j

        shape = 0
        orien = 0
        latent_matrix = np.array([[0]*32*32*6,
                [shape]*32*32*6,
                com_matrix[0,:],
                [orien]*32*32*6,
                com_matrix[1,:],
                com_matrix[2,:]])

        latent_matrix = latent_matrix.T
        return latent_matrix

    def get_images_from_latent(self,dataset,latent_matrix):
        indices = np.array(np.dot(latent_matrix, dataset.factor_bases), dtype=np.int64)
        images = dataset.images[indices].astype(np.float32)
        if len(images.shape) < 4:
            images = np.expand_dims(images, axis=3)
        if self.name == "color_dsprites":
            color = np.random.uniform(0.5,1,[2,1, 1, 3])
            images = (np.repeat(images, 3, axis=3)*color)
        if self.name == "noisy_dsprites":
            color = np.random.uniform(0,1,[64, 64, 3])
            images = np.minimum(np.repeat(images, 3, axis=3) + color,1)
        return images


    def scatter3d(self,x,y,z, cs, colorsMap='jet'):
        cm = plt.get_cmap(colorsMap)
        cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
        scalarMap.set_array(cs)
        fig.colorbar(scalarMap)
        plt.show()



    def visualize_xys(self):
        latent_matrix = get_latent_matrix()
        images = get_images_from_latent(dataset,latent_matrix)
        scatter3d(latent_matrix[:,-1],latent_matrix[:,-2],latent_matrix[:,2],cs=np.linspace(0,0.5,32*32*6))




    def visual_traversal(self,net):
        def _representation_torch(x):
            x = Variable(torch.from_numpy(x).float()).cuda()
            x = x.permute(0,3,1,2)
            z = net.encoder(x.contiguous())[:,:net.z_dim]
            return z
        if self.name == "dsprites_full":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2])
        elif self.name == "shapes3d":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2])
        elif self.name == "color_dsprites":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2])
        elif self.name == "noisy_dsprites":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2])
        elif self.name == "cars3d":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2])
        elif self.name == "overlap144":
            latent_matrix = np.array([[0]*2,
                        [0]*2])
        elif self.name == "overlap1920":
            latent_matrix = np.array([[0]*2,
                        [0]*2,
                        [0]*2,
                        [0]*2])
        latent_matrix = latent_matrix.T
        images = self.get_images_from_latent(self.dataset,latent_matrix)
        z_ori = _representation_torch(images)[0].unsqueeze(0)
        samples = []
        gifs =[]
        interpolation = torch.arange(-2, 2, 0.4)
        loc = -1
        for row in range(net.z_dim):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row] = val
                if net.group:
                    cm_z = net.zcomplex(z)
                    sample = F.sigmoid(net.decoder(cm_z)).data
                else:
                    sample = F.sigmoid(net.decoder(z)).data
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()
        return make_grid(samples,nrow=10)
    