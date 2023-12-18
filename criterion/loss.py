import segmentation_models_pytorch as smp


def get_criterion():
    criterion = smp.losses.DiceLoss('multilabel', from_logits=False)
    return criterion