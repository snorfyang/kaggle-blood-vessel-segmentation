import torch
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import gc
import os
from utils.util import seed_everything, rle_encode
from dataset.dataset import get_loader
from models.ConvNeXt_U import ConvNeXt_U
from models.ResNet_U import ResNet_U
from criterion.metric import score


def tta_transforms(image):
        return [
            image,                          
            torch.flip(image, [2]),        
            torch.flip(image, [3]),        
            torch.flip(image, [2, 3]),      
        ]

def tta_inverse_transforms(preds):
        pred, pred_hflip, pred_vflip, pred_hvflip = preds
        pred_hflip = torch.flip(pred_hflip, [2])
        pred_vflip = torch.flip(pred_vflip, [3])
        pred_hvflip = torch.flip(pred_hvflip, [2, 3])
        ret = (pred + pred_hflip + pred_vflip + pred_hvflip) / 4
        del pred, pred_hflip, pred_vflip, pred_hvflip
        gc.collect()
        return ret

def valid(m):
    seed_everything(42)
    valid_x, valid_y, valid_z = get_loader('valid', '/storage', 4, 4)
    
    depth = 501
    height = 1706
    width = 1510
    
    model = ResNet_U().cuda()
    model.load_state_dict(torch.load(m))
    model.eval()

    sub = []
    
    accumulator_m = np.zeros((depth, height, width))
    accumulator_k = np.zeros((depth, height, width))
    
    def process_slice(loader, axis):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, (image, msk) in enumerate(tqdm(loader)):
                    del msk
                    gc.collect()
                    tta_images = tta_transforms(image.cuda())
                    tta_kidneys, tta_masks = [], []

                    for tta_image in tta_images:
                        masks, kidneys = model(tta_image)
                        tta_kidneys.append(kidneys)
                        tta_masks.append(masks)

                    kidneys = tta_inverse_transforms(tta_kidneys)
                    masks = tta_inverse_transforms(tta_masks)

                    for batch_index in range(kidneys.size(0)):
                        kidney_np = kidneys[batch_index].squeeze().cpu().numpy()
                        mask_np = masks[batch_index].squeeze().cpu().numpy()

                        global_index = i * loader.batch_size + batch_index
                        
                        if axis == 'x':
                            accumulator_m[global_index, :, :] += mask_np
                            accumulator_k[global_index, :, :] += kidney_np
                            
                        elif axis == 'y':
                            accumulator_m[:, global_index, :] += mask_np
                            accumulator_k[:, global_index, :] += kidney_np

                        elif axis == 'z':
                            accumulator_m[:, :, global_index] += mask_np
                            accumulator_k[:, :, global_index] += kidney_np
                        
                        del kidney_np, mask_np
                        gc.collect()
                    
    for i, j in zip([valid_x, valid_y, valid_z], ['x', 'y', 'z']):
        process_slice(i, j)
        gc.collect()
        torch.cuda.empty_cache()
        
    del valid_x, valid_y, valid_z
    gc.collect()
    
    accumulator_m /= 3
    accumulator_k /= 3
    # average_result = accumulator_m>0.1
    average_result = np.logical_and(accumulator_m>0.1, accumulator_k>0.5)
    del accumulator_k, accumulator_m
    gc.collect()

    sub = []
    for slice_idx in range(depth):
        slice_ = average_result[slice_idx, :, :]
        rle_encoded = rle_encode(slice_)
        sub.append(rle_encoded)
    
    del average_result
    gc.collect()
    
    df = pd.read_csv('/notebooks/gt.csv')
    ids = df[df['id'].str.contains("kidney_3_dense")].id.values

    submission = pd.DataFrame.from_dict({
    "id": ids,
    "rle": sub
    })

    submission.to_csv('submission.csv')
    _gt_df = pd.merge(df, submission.loc[:, ["id"]], on="id").reset_index(drop=True)
    val_score = score(submission, _gt_df)
    print(val_score)
    return val_score

if __name__ == "__main__":
    valid('convnext-unet-1ohr1hkt.pt')


    