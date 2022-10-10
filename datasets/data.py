from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import random
import gin
import types
import h5py
import PIL




@gin.configurable
class CustomTensorDatasetfor1(Dataset):
    def __init__(self, data_tensor, noisy=False, color=False):
        self.data_tensor = data_tensor
        self.noisy = noisy
        self.color = color
        if self.color:
            self.color_tensor = np.random.uniform(0,1,[data_tensor.size(0),3, 1, 1])

    def __getitem__(self, index):
        if self.noisy:
            color = np.random.uniform(0,1,[3, 64, 64])
            return np.minimum(self.data_tensor[index].repeat(3,1,1) + color,1).float()
        elif self.color:
            return (self.data_tensor[index].repeat(3,1,1) * self.color_tensor[index]).float()
        else:
            return self.data_tensor[index]
        

    def __len__(self):
        return self.data_tensor.size(0)

@gin.configurable
class CustomTensorDatasetfor2(Dataset):
    def __init__(self, data_tensor, transform=None, noisy = False, color=False):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))
        self.noisy = noisy
        self.color = color
        if self.color:
            self.color_tensor = np.random.uniform(0,1,[data_tensor.size(0),3, 1, 1])

    def __getitem__(self, index1):
        if self.noisy:
            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            color1 = np.random.uniform(0,1,[3, 64, 64])
            img2 = self.data_tensor[index2]
            color2 = np.random.uniform(0,1,[3, 64, 64])
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return np.minimum(img1.repeat(3,1,1) + color1,1).float(), np.minimum(img2.repeat(3,1,1) + color2,1).float()
        elif self.color:
            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            img2 = self.data_tensor[index2]
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return (img1.repeat(3,1,1) * self.color_tensor[index1]).float(), (img2.repeat(3,1,1) * self.color_tensor[index2]).float()
        else:

            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            img2 = self.data_tensor[index2]
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2
        
    def __len__(self):
        return self.data_tensor.size(0)

@gin.configurable
def return_data(
    Dataset_fc = gin.REQUIRED,
    name = gin.REQUIRED,
    images = None,
    batch_size = 32
):
    dset_dir = os.path.join(
    os.environ.get("DISENTANGLEMENT_DATA", "."), "dataset_folder")
    num_workers = 2

    if name.lower() == 'dsprites_full':
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = Dataset_fc
    
    elif name.lower() == 'shapes3d':
        data = torch.from_numpy(images).permute(0,3,1,2).float()
        train_kwargs = {'data_tensor':data}
        dset = Dataset_fc
    
    elif name.lower() == "color_dsprites":
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data, 'color':True}
        dset = Dataset_fc
    
    elif name.lower() == "noisy_dsprites":
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data, 'noisy':True}
        dset = Dataset_fc

    elif name.lower() == "cars3d":
        data = torch.from_numpy(images).permute(0,3,1,2).float()
        train_kwargs = {'data_tensor':data}
        dset = Dataset_fc
    elif name.lower() == "overlap144":
        root = os.path.join(dset_dir, 'np_imgs144.npz')
        data = np.load(root)
        data = torch.from_numpy(data['imgs']/255.).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = Dataset_fc
    elif name.lower() == "overlap1920":
        root = os.path.join(dset_dir, 'np_imgs1920.npz')
        data = np.load(root)
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = Dataset_fc
        


    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    train_loader.name = name

    data_loader = train_loader

    return data_loader
