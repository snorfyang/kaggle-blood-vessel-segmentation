import gc
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch.cuda.amp import GradScaler, autocast

from criterion.loss import get_criterion
from criterion.metric import surface_dice
from dataset.dataset import get_loader
from models.model import get_model
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler
from utils.util import seed_everything
from valid import valid

gc.collect()
class CFG:
    seed          = 42
    model_type    = '2D'
    model_name    = 'convnext-unet'
    train_bs      = 4
    valid_bs      = 4
    lr            = 5e-4
    epochs        = 10
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_decay  = 1e-4
    note          = 'multiviews, flip, brightness, contrast, rotate'
    debug         = False
    data_dir      = '/storage'
    kidney_mask   = False

def validate(model, loaders, device, criterion):
    model.eval()
    running_loss = 0.0
    for loader in loaders:
        with torch.no_grad():
            for inputs, labels in tqdm(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                with autocast():
                    v, k = model(inputs)
                    outputs = torch.cat((v, k), dim=1)
                    loss = criterion(outputs, labels)
                running_loss += loss.item()
            
        torch.cuda.empty_cache()
        gc.collect()
    
    return running_loss / np.sum([len(i) for i in loaders])


def main():
    if not CFG.debug:
        wandb.login()
    seed_everything(CFG.seed)
    tag = [CFG.model_name]
    m = 'vessel'
    if CFG.kidney_mask:
        tag.append('kidney_mask')
        m = 'kidney'
        
    train_x, train_y, train_z, valid_x, valid_y, valid_z = get_loader('train', CFG.data_dir, CFG.train_bs, CFG.valid_bs)
    
    cfg_dict = {k: v for k, v in CFG.__dict__.items() if not k.startswith('__') and not callable(v)}
    
    if not CFG.debug:
        run = wandb.init(
        entity = "sweetdreams",
        project = "SenNet + HOA - Hacking the Human Vasculature in 3D",
        group = CFG.model_type,
        config = cfg_dict,
        tags = tag,
    )
        
    
    device = CFG.device
    model = get_model()
    optimizer = get_optimizer(model, CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = get_criterion()
    scaler = GradScaler()

    num_epochs = CFG.epochs
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for loader in [train_x, train_y, train_z]:
            for inputs, labels in tqdm(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    v, k = model(inputs)
                    outputs = torch.cat((v, k), dim=1)
                    loss = criterion(outputs, labels)
                if CFG.debug:
                    print('loss', loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                if not CFG.debug:
                    wandb.log({"train_step_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

                
            torch.cuda.empty_cache()
            gc.collect()

        train_loss = running_loss / (len(train_x) + len(train_y) + len(train_z))
        
        torch.cuda.empty_cache()
        gc.collect()
        
        valid_loss = validate(model, [valid_x], device, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        if not CFG.debug:
            wandb.log({'epoch': epoch, 'train_avg_loss': train_loss, 'valid_avg_loss': valid_loss})
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'{CFG.model_name}-{run.id}.pt')
            
            print('Best model saved!')
        
        scheduler.step()
        
    # torch.save(model.state_dict(), f'{CFG.model_name}-{run.id}-last.pt')

    if not CFG.debug: 
        artifact = wandb.Artifact('model_artifact', type='model')
        artifact.add_file(f'{CFG.model_name}-{run.id}.pt')
        # artifact.add_file(f'{CFG.model_name}-{run.id}-last.pt')
        wandb.log_artifact(artifact)

        wandb.finish()

    torch.cuda.empty_cache()
    gc.collect()
    print('Training Complete')
    return f'{CFG.model_name}-{run.id}.pt'


if __name__ == '__main__':
    model = main()
    print(f'Best model saved as {model}.')
    valid(model)

