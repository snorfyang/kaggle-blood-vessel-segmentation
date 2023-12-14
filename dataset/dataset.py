import albumentations as A
import cv2
import random
import numpy as np
import gc
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


from dataset.preprocess import prepare_datadirs

class MyLoader(Dataset):
    def __init__(self, images, labels, train, mode='x'):
        self.images = images

        self.labels = labels
        self.train = train
        self.mode = mode
        self.transform = A.Compose([
                                    A.RandomBrightnessContrast(p=0.3, brightness_limit=0.05, contrast_limit=0.05),
                                    A.HorizontalFlip(p=0.3),
                                    A.VerticalFlip(p=0.3),
                                    A.Rotate(p=0.3, limit=30),
                                  ])
    
    def __len__(self):
        if self.mode == "x":
            return self.images.shape[0]
        elif self.mode == "y":
            return self.images.shape[1]
        elif self.mode == "z":
            return self.images.shape[2]

    def __getitem__(self, index):
        if self.mode == "x":
            img = self.images[index, :, :]
            msk = self.labels[index, :, :]
        
        elif self.mode == "y":
            img = self.images[:, index, :]
            msk = self.labels[:, index, :]

        elif self.mode == "z":
            img = self.images[:, :, index]
            msk = self.labels[:, :, index]
        
        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 0.0001)
        msk = np.array(msk)/255
        
        if self.train:
            transformed = self.transform(image=img, mask=msk)
            img = transformed['image']
            msk = transformed['mask']
            del transformed
        
        return torch.from_numpy(img).float().unsqueeze(0), torch.from_numpy(msk).float().unsqueeze(0)
    
def get_loader(mode, data_dir, train_bs, valid_bs):
    path = data_dir
    if 'train' in mode:
        train_data = np.load(f"{path}/kidney_1_dense.npz")
        train_images = train_data["images"]
        train_labels = train_data["labels"]
        
        valid_data = np.load(f"{path}/kidney_3_dense.npz")
        valid_images = valid_data["images"]
        valid_labels = valid_data["labels"]
        del train_data, valid_data
        gc.collect()
        train_x = MyLoader(train_images, train_labels, train=True, mode="x")
        train_y = MyLoader(train_images, train_labels, train=True, mode="y")
        train_z = MyLoader(train_images, train_labels, train=True, mode="z")
        train_x = DataLoader(train_x, batch_size=train_bs, shuffle=True, pin_memory=True)
        train_y = DataLoader(train_y, batch_size=train_bs, shuffle=True, pin_memory=True)
        train_z = DataLoader(train_z, batch_size=train_bs, shuffle=True, pin_memory=True)
        
        valid_x = MyLoader(valid_images, valid_labels, train=False, mode="x")
        valid_y = MyLoader(valid_images, valid_labels, train=False, mode="y")
        valid_z = MyLoader(valid_images, valid_labels, train=False, mode="z")
        valid_x = DataLoader(valid_x, batch_size=valid_bs, shuffle=False, pin_memory=True)
        valid_y = DataLoader(valid_y, batch_size=valid_bs, shuffle=False, pin_memory=True)
        valid_z = DataLoader(valid_z, batch_size=valid_bs, shuffle=False, pin_memory=True)
        return train_x, train_y, train_z, valid_x, valid_y, valid_z

    if 'valid' in mode:
        valid_data = np.load(f"{path}/kidney_3_dense.npz")
        valid_images = valid_data["images"]
        valid_labels = valid_data["labels"]
        del valid_data
        valid_x = MyLoader(valid_images, valid_labels, train=False, mode="x")
        valid_y = MyLoader(valid_images, valid_labels, train=False, mode="y")
        valid_z = MyLoader(valid_images, valid_labels, train=False, mode="z")
        valid_x = DataLoader(valid_x, batch_size=valid_bs, shuffle=False, pin_memory=True)
        valid_y = DataLoader(valid_y, batch_size=valid_bs, shuffle=False, pin_memory=True)
        valid_z = DataLoader(valid_z, batch_size=valid_bs, shuffle=False, pin_memory=True)
        return  valid_x, valid_y, valid_z