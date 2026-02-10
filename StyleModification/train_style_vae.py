import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from autoencoder import MotionVAE, StyleVAE, style_vae_loss_function
from dataloader import load_data
from config import Args
from utils import save_checkpoint, frange_cycle_linear
from tqdm import tqdm
import wandb

args = Args()
run_name = "SHS+CMU-Style"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load MotionAE ---
motion_vae = MotionVAE(args).to(device)
checkpoint = torch.load("./Models/SHS+CMU-Motion_epoch_1000.pth", map_location=device, weights_only=False)
motion_vae.load_state_dict(checkpoint['state_dict'])

dataset, train_loader, val_loader = load_data(seq=args.seq, batch=args.style_batch_size, motion_vae=motion_vae)

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

model = StyleVAE(motion_vae, args).to(device)
optimizer = Adam(model.parameters(), lr=args.style_lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.style_epochs, eta_min=1e-5)
betas = frange_cycle_linear(args.style_epochs + 1, 0, args.s_beta, 5, 0.5)

def eval(beta):
    model.eval()
    loss_all, rec_loss_all, cyc_loss_all, kl_loss_all, hsa_loss_all = 0, 0, 0, 0, 0
       
    with torch.no_grad():
        for sample in val_loader:
            A, B, C = sample
            _, motion_A, partial_A, _, _, _, _, emb_A = A
            _, motion_B, partial_B, _, _, _, _, emb_B = B
            _, motion_C, partial_C, _, _, _, _, _, = C
            motion_A, partial_A, motion_B, partial_B, motion_C, partial_C, emb_A, emb_B = motion_A.to(device), partial_A.to(device), motion_B.to(device), partial_B.to(device), motion_C.to(device), partial_C.to(device), emb_A.to(device), emb_B.to(device)

            ret = model(partial_A, partial_B, partial_C)
            rec_loss, cyc_loss, kl_loss, hsa_loss = style_vae_loss_function(ret, motion_A, motion_B, motion_C, emb_A, emb_B)
            loss = rec_loss + beta * kl_loss + args.gamma * cyc_loss + args.delta * hsa_loss
            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            cyc_loss_all += cyc_loss.item()      
            kl_loss_all += kl_loss.item()
            hsa_loss_all += hsa_loss.item()

    loss_all /= len(val_loader)
    rec_loss_all /= len(val_loader)
    cyc_loss_all /= len(val_loader)
    kl_loss_all /= len(val_loader)
    hsa_loss_all /= len(val_loader)

    return loss_all, rec_loss_all, cyc_loss_all, kl_loss_all, hsa_loss_all 

def train():
    for epoch in tqdm(range(args.style_epochs + 1), desc="Training: "):
        model.train()
        beta = betas[epoch]
        loss_all, rec_loss_all, cyc_loss_all, kl_loss_all, hsa_loss_all = 0, 0, 0, 0, 0

        for sample in train_loader:
            A, B, C = sample
            _, motion_A, partial_A, _, _, _, _, emb_A = A
            _, motion_B, partial_B, _, _, _, _, emb_B = B
            _, motion_C, partial_C, _, _, _, _, _ = C
            
            motion_A, partial_A, motion_B, partial_B, motion_C, partial_C, emb_A, emb_B = motion_A.to(device), partial_A.to(device), motion_B.to(device), partial_B.to(device), motion_C.to(device), partial_C.to(device), emb_A.to(device), emb_B.to(device)

            optimizer.zero_grad()
            ret = model(partial_A, partial_B, partial_C)
            rec_loss, cyc_loss, kl_loss, hsa_loss = style_vae_loss_function(ret, motion_A, motion_B, motion_C, emb_A, emb_B)
            loss = rec_loss + beta * kl_loss + args.gamma * cyc_loss + args.delta * hsa_loss
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            rec_loss_all += rec_loss.item()
            cyc_loss_all += cyc_loss.item()
            kl_loss_all += kl_loss.item()
            hsa_loss_all += hsa_loss.item()

        loss_all /= len(train_loader)
        rec_loss_all /= len(train_loader)
        cyc_loss_all /= len(train_loader)
        kl_loss_all /= len(train_loader)
        hsa_loss_all /= len(train_loader)

        val_loss, val_rec_loss, val_cyc_loss, val_kl_loss, val_hsa_loss = eval(beta)
        scheduler.step()

        # =================== Logging ======================
        print(f"Epoch: {epoch+1} | Beta: {beta:.4f}")
        print(f"Train: L: {loss_all:.4f} | Rec_L: {rec_loss_all:.4f} | KL_L: {kl_loss_all:.4f} | Cyc_L: {cyc_loss_all:.4f} | HSA_L: {hsa_loss_all:.4f}")
        print(f"Eval:  L: {val_loss:.4f} | Rec_L: {val_rec_loss:.4f} | KL_L: {val_kl_loss:.4f} | Cyc_L: {val_cyc_loss:.4f} | HSA_L: {val_hsa_loss:.4f}")

        if args.track_wandb:
            wandb.log({
                "train_loss": loss_all,
                "train_rec_loss": rec_loss_all,
                "train_kl_loss": kl_loss_all,
                "train_cyc_loss": cyc_loss_all,
                "train_hsa_loss": hsa_loss_all,
                "val_loss": val_loss,
                "val_rec_loss": val_rec_loss,
                "val_kl_loss": val_kl_loss,
                "val_cyc_loss": val_cyc_loss,
                "val_hsa_loss": val_hsa_loss,
                "lr": scheduler.get_last_lr()[0],
                "beta": beta
            })

        if args.track_wandb:
            if (epoch + 1) % 50 == 0:
                filename = f'{run_name}_epoch_{epoch+1}.pth'
                path = save_checkpoint(model, optimizer, epoch, args, filename)
                wandb.save(path)

train()