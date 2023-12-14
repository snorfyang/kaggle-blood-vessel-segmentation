import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_everything(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def show_images(dataloader, num_images=3):
    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 3))
    if num_images == 1:
        axes = [axes]

    for i, (images, masks) in enumerate(dataloader):
        if i >= num_images: break

        image = images[0].squeeze().numpy()
        mask = masks[0].squeeze().numpy()   

        axes[i][0].imshow(image, cmap='gray')
        axes[i][0].set_title('Image')
        axes[i][0].axis('off')

        axes[i][1].imshow(mask)
        axes[i][1].set_title('Mask')
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()

def file_to_id(f):
    s = f.split('/')
    return s[-3]+'_' + s[-1][:-4]

def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle