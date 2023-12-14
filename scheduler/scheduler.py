from torch.optim.lr_scheduler import CosineAnnealingLR


def get_scheduler(optimizer, cfg):
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)
    return scheduler