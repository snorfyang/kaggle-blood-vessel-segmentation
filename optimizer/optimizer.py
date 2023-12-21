import torch.optim as optim


def get_optimizer(model, cfg):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-4)
    return optimizer