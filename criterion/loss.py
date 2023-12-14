import segmentation_models_pytorch as smp


def get_criterion():
    criterion = smp.losses.DiceLoss('binary', from_logits=False)
    return criterion