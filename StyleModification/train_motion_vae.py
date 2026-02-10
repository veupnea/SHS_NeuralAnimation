import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from autoencoder import MotionVAE, motion_vae_loss_function
from dataloader import load_data
from config import Args
from utils import save_checkpoint
from tqdm import tqdm
import wandb
import numpy as np
    
args = Args()
run_name = "SHS-Motion"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, train_loader, val_loader = load_data(seq=args.seq, batch=args.batch_size)

if args.track_wandb:
    import wandb
    wandb.login(key=args.wandb_key)
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

def eval():
    model.eval()
    loss_all, rec_loss_all, kl_loss_all = 0, 0, 0
       
    all_z = []
    with torch.no_grad():
        for sample in val_loader:
            _, in_motion, partial, *_ = sample
            in_motion, partial = in_motion.to(device), partial.to(device)
            
            pred, distr = model(partial)
            rec_loss, kl_loss = motion_vae_loss_function(pred, distr, in_motion)
            loss = rec_loss + args.m_beta * kl_loss
            loss_all += loss.item()
            rec_loss_all += rec_loss.item()      
            kl_loss_all += kl_loss.item()
            all_z.append(distr[0].mean(-1).cpu().numpy())

    loss_all /= len(val_loader)
    rec_loss_all /= len(val_loader)
    kl_loss_all /= len(val_loader)
    # Plot distribution of z
    all_z = np.concatenate(all_z, axis=0)
    all_z_flat = all_z.flatten()
    mu, sigma = all_z_flat.mean(), all_z_flat.std()
    print(f"\nm: {mu:.3f}, s: {sigma:.3f}")

    return loss_all, rec_loss_all, kl_loss_all

def train():
    for epoch in tqdm(range(args.epochs + 1), desc="Training: "):
        model.train()

        loss_all, rec_loss_all, kl_loss_all = 0, 0, 0

        for sample in train_loader:
            _, in_motion, partial, *_ = sample
            in_motion, partial = in_motion.to(device), partial.to(device)

            optimizer.zero_grad()
            pred, distr = model(partial)
            rec_loss, kl_loss = motion_vae_loss_function(pred, distr, in_motion)
            loss = rec_loss + args.m_beta * kl_loss
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            kl_loss_all += kl_loss.item()

        loss_all /= len(train_loader)
        rec_loss_all /= len(train_loader)
        kl_loss_all /= len(train_loader)

        val_loss, val_rec_loss, val_kl_loss = eval()
        scheduler.step()

        # =================== Logging ======================
        print(f"Epoch: {epoch+1}")
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
                "lr": scheduler.get_last_lr()[0]
            })

        if args.track_wandb:
            if (epoch + 1) % 100 == 0:
                filename = f'{run_name}_epoch_{epoch+1}.pth'
                path = save_checkpoint(model, optimizer, epoch, args, filename)
                wandb.save(path)

train()