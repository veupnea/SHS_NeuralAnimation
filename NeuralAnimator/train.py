import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model import MotionVAE, loss_function
import os
from dataloader import create_dataloaders
from config import Args
from utils import save_checkpoint, frange_cycle_linear
import numpy as np

args = Args()
run_name = "MotionVAE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)

if args.track_wandb:
    import wandb
    wandb.login(key="your_wandb_api")
    wandb.init(
        project = "Style",
        name = run_name,
        config=Args.to_dict()
    )
    path = os.path.dirname(os.path.abspath(__file__)) 
    wandb.save(os.path.join(path, "config.py"))

model = MotionVAE(args).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
betas = frange_cycle_linear(args.epochs + 1, 0, 0.02, 1, 0.75)
noise_levels = frange_cycle_linear(args.epochs + 1, 0, 0.025, 1, 0.75)

def eval(beta, noise):
    model.eval()
    loss_all, rec_loss_all, kl_loss_all = 0, 0, 0
    all_z = []

    with torch.no_grad():
        for batch in val_loader:
            last = (batch['last_full'].to(device), batch['last_partial'].to(device))
            current = (batch['current_full'].to(device), batch['current_partial'].to(device))
            fk = (batch['parents'].to(device), batch['offsets'].to(device))
                     
            pred, mu, logvar, mu_p, logvar_p, z = model(last, current, noise)
            all_z.append(z.cpu().numpy())
            loss, rec_loss, kl_loss = loss_function(pred, current[0], fk, mu, logvar, mu_p, logvar_p, beta)

            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            kl_loss_all += kl_loss.item()

    loss_all /= len(val_loader)
    rec_loss_all /= len(val_loader)
    kl_loss_all /= len(val_loader)

    # print mean and std of z
    all_z = np.concatenate(all_z, axis=0)
    print(f"\nZ mean: {np.mean(all_z):.2f}, std: {np.std(all_z):.2f}")

    return loss_all, rec_loss_all, kl_loss_all

def train():
    for epoch in tqdm(range(args.epochs + 1), desc="Training: "):
        model.train()
        beta = betas[epoch]
        noise = noise_levels[epoch]

        loss_all, rec_loss_all, kl_loss_all = 0, 0, 0
        for batch in train_loader:
            last = (batch['last_full'].to(device), batch['last_partial'].to(device))
            current = (batch['current_full'].to(device), batch['current_partial'].to(device)) 
            fk = (batch['parents'].to(device), batch['offsets'].to(device))
                     
            pred, mu, logvar, mu_p, logvar_p, z = model(last, current, noise)
            loss, rec_loss, kl_loss = loss_function(pred, current[0], fk, mu, logvar, mu_p, logvar_p, beta) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            kl_loss_all += kl_loss.item()

        loss_all /= len(train_loader)
        rec_loss_all /= len(train_loader)
        kl_loss_all /= len(train_loader)

        val_loss, val_rec_loss, val_kl_loss = eval(beta, noise)
        scheduler.step()

        # =================== Logging ======================
        print(f"Epoch: {epoch+1} | beta: {beta:.6f} | noise: {noise:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train: L: {loss_all:.4f} | Rec_L: {rec_loss_all:.4f} | KL_L: {kl_loss_all:.4f}")
        print(f"Eval:  L: {val_loss:.4f} | Rec_L: {val_rec_loss:.4f} | KL_L: {val_kl_loss:.4f}")

        if args.track_wandb:
            wandb.log({
                "train_loss": loss_all,
                "train_rec_loss": rec_loss_all,
                "train_kl_loss": kl_loss_all,
                "val_loss": val_loss,
                "val_rec_loss": val_rec_loss,
                "val_kl_loss": val_kl_loss,
                "lr": scheduler.get_last_lr()[0],
                "beta": beta
            })

        if args.track_wandb:
            if (epoch + 1) % 25 == 0:
                filename = f'{run_name}_epoch_{epoch+1}.pth'
                path = save_checkpoint(model, optimizer, epoch, args, filename)
                wandb.save(path)

train()
