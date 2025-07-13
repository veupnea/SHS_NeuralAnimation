import torch
import torch.nn as nn
import torch.nn.functional as F 
from utils import fk_6d_torch, geodesic_so3_loss
from dataloader import PARTIAL_JOINTS, NUM_JOINTS

class PriorNetwork(nn.Module):
    def __init__(self, feat_dim, partial_dim, latent_dim, h_dim, dropout=0.1):
        super().__init__()
        self.last_conv = nn.Sequential(
            nn.Conv1d(feat_dim, h_dim // 2, kernel_size=1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=1),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.part_mlp = nn.Sequential(
            nn.Linear(partial_dim, h_dim),
            nn.ELU(),
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(h_dim * 2, h_dim)
        self.fused_mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU()
        )
        self.mu = nn.Linear(h_dim, latent_dim)
        self.logvar = nn.Linear(h_dim, latent_dim)

    def forward(self, last_full, current_partial):
        """
        last_full:       [B, J, D]        (previous full pose)
        current_partial: [B, P, d]        (positions + velocities for partial joints)
        bone_lengths:    [B, J]           (per-joint bone length magnitudes)
        """
        B, J, D = last_full.shape
        current_partial = current_partial.view(B, -1)     # [B, partial_dim]

        x_last = last_full[:,:,:3].permute(0, 2, 1)               # [B, D, J]
        e_last = self.last_conv(x_last).squeeze(-1)       # [B, h_dim]
        e_part = self.part_mlp(current_partial)           # [B, h_dim]

        cat = torch.cat([e_last, e_part], dim=1)          # [B, 2*h_dim]
        g = torch.sigmoid(self.gate(cat))                 # [B, h_dim]

        fused = g * e_last + (1 - g) * e_part             # [B, h_dim]
        fused_out = self.fused_mlp(fused)                 # [B, h_dim]

        mu = self.mu(fused_out)                           # [B, latent_dim]
        logvar = self.logvar(fused_out)                   # [B, latent_dim]
        return mu, logvar

class Encoder(nn.Module):
    def __init__(self, feat_dim, latent_dim, h_dim, dropout=0.1): #0.1
        super().__init__()
        in_channels = feat_dim * 2  # last and current pose concatenated
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels, h_dim // 4, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(h_dim // 4, h_dim // 2, kernel_size=1),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=1),
            nn.ELU(inplace=True),
            nn.AdaptiveMaxPool1d(1)  # Pool over joints
        )
        self.norm = nn.LayerNorm(h_dim)
        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)

    def forward(self, last_pose, current_pose):
        """
        last_pose:    [B, J, D]
        current_pose: [B, J, D]
        """
        x = torch.cat([last_pose, current_pose], dim=-1)
        x = x.permute(0, 2, 1)
        h = self.conv_stack(x).squeeze(-1)
        h = self.norm(h)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_dim, partial_dim, latent_dim, h_dim):
        super().__init__()
        self.part_proj = nn.Linear(partial_dim, latent_dim)
        self.input_proj = nn.Linear(2 * latent_dim, h_dim)
        self.mlp = nn.Sequential(
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.ELU(),
            nn.Linear(h_dim // 2, out_dim)
        )

    def forward(self, z, current_partial_pose):                
        B = z.shape[0]
        partial_flat = current_partial_pose.view(B, -1)
        partial_enc = self.part_proj(partial_flat)
        x = torch.cat([z, partial_enc], dim=-1)
        x = self.input_proj(x)
        return self.mlp(x)

class MotionVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.prior = PriorNetwork(args.partial_feat_dim, args.partial_dim, args.latent_dim, args.hidden_dim)
        self.encoder = Encoder(args.feat_dim, args.latent_dim, args.hidden_dim)
        self.decoder = Decoder(args.out_dim, args.partial_dim, args.latent_dim, args.hidden_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, last, current, noise): 
        last_full = last[0]
        current_full, current_partial = current[0], current[1]
        # Add noise
        if self.training:
            current_partial = current_partial + torch.randn_like(current_partial) * noise
            last_full = last_full + torch.randn_like(last_full) * noise
        # Encode
        mu, logvar = self.encoder(last_full, current_full) # Use only positions
        mu_p, logvar_p = self.prior(last_full[..., :3], current_partial) # Use only positions
        z = self.reparameterize(mu, logvar)
        # Decode
        recon = self.decoder(z, current_partial)        
        return recon, mu, logvar, mu_p, logvar_p, z
    
    def infer(self, last_full, current_partial):
        self.eval()
        device = next(self.parameters()).device  # ensure matching device
        # Add batch dimension
        last_full = last_full.unsqueeze(0).to(device)            # [1, J, D]
        current_partial_flat = current_partial.unsqueeze(0).reshape(1, -1).to(device)  # [1, P*d]
        
        with torch.no_grad():
            mu, logvar = self.prior(last_full[..., :3], current_partial_flat)  # [1, D]
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z, current_partial_flat)
            pred_root_rot = recon[0, :6]                     # [6]
            pred_joints_rot = recon[0, 6:].reshape(-1, 6)     # [J-1, 6]

        return pred_root_rot.cpu(), pred_joints_rot.cpu()

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    # elementwise KL
    kl_elem = 0.5 * (
        (var_q / var_p)
        + ((mu_q - mu_p).pow(2) / var_p)
        - 1.0
        + (logvar_p - logvar_q)
    )  # [B, D]
    # sum over latent dims, then mean over batch
    return kl_elem.sum(dim=1).mean()

def rec_loss(pred, target, fk_data):
    # Predictions
    pred_root_rot = pred[:, :6]
    pred_joints_rot = pred[:, 6:].reshape(pred.shape[0], -1, 6)  # [B, J, 6]
    # True values  
    root_pos = target[:, 0, :3]
    root_rot = target[:, 0, 3:9]
    joints_pos = target[:, :, :3]
    joints_rot = target[:, :, 3:9]
    
    # -- Root Loss --
    root_rot_loss = geodesic_so3_loss(pred_root_rot, root_rot) # Root rotation loss
        
    # -- End Efectors Joints Loss --
    parents, offsets = fk_data
    ee_indices = PARTIAL_JOINTS[1:]
    # - EE Pos Loss -
    # Compute forward kinematics to get joint positions
    pred_rot = torch.cat([pred_root_rot.unsqueeze(1), pred_joints_rot], dim=1)  # [B, J, 6]
    joints_fk_pos, _ = fk_6d_torch(pred_rot.unsqueeze(1), root_pos.unsqueeze(1), offsets, parents.squeeze(0)[0])
    pred_joint_pos = joints_fk_pos.squeeze(1)  # [B, J, 3] 
    pred_joint_pos -= root_pos.unsqueeze(1)  # Convert to local space by subtracting root position
    joints_pos_loss = F.l1_loss(pred_joint_pos[:, 1:], joints_pos[:, 1:], reduction='mean')  # Position loss    
    joints_rot_loss = geodesic_so3_loss(pred_joints_rot, joints_rot[:, 1:])  # Rotation loss
    joints_ee_pos_loss = F.mse_loss(pred_joint_pos[:, ee_indices, :], joints_pos[:, ee_indices, :], reduction='mean')  # End-effector positions
    # - EE Rot Loss -
    pred_joints_rot_ee = pred_rot[:, ee_indices]
    joints_rot_ee = joints_rot[:, ee_indices]
    joints_ee_rot_loss = geodesic_so3_loss(pred_joints_rot_ee, joints_rot_ee)
    
    # -- Total Loss --
    rec_loss = (
        2.0 * root_rot_loss +
        5.0 * joints_pos_loss +
        2.0 * joints_rot_loss + 
        10.0 * joints_ee_pos_loss +
        2.0 * joints_ee_rot_loss
    )
    return rec_loss

def loss_function(pred, current, fk, mu, logvar, mu_p, logvar_p, beta):
    kl_loss = kl_divergence(mu, logvar, mu_p, logvar_p)
    rec = rec_loss(pred, current, fk)
    loss = rec + beta * kl_loss
    return loss, rec, kl_loss