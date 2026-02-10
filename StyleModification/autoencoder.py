import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import NUM_JOINTS

# =================================
# =========== MotionAE ============
# =================================
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        h_dims  = [256, 384, 512]
        strides = [2, 2, 1]
        layers, in_ch = [], input_dim
        for out_ch, stride in zip(h_dims, strides):
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
            ))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_ch)
        self.mu_conv = nn.Conv1d(in_ch, latent_dim, kernel_size=1)
        self.logvar_conv = nn.Conv1d(in_ch, latent_dim, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.conv_stack(x)
        h = h.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super().__init__()
        self.input_proj = nn.Conv1d(latent_dim, 512, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.block2 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.block3 = nn.Sequential(
            nn.Conv1d(512, 384, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
        )
        self.output_proj = nn.Conv1d(256, output_dim, kernel_size=1)

    def forward(self, z):
        h = self.input_proj(z)
        h = self.block1(h)
        h = self.upsample1(h)
        h = self.block2(h)
        h = self.upsample2(h)
        h = self.block3(h)
        h = self.block4(h)
        out = self.output_proj(h)
        return out.permute(0, 2, 1)

class MotionVAE(nn.Module):
    def __init__(self, args):
        super(MotionVAE, self).__init__()
        self.encoder = Encoder(args.partial_dim, args.latent_dim)
        self.decoder = Decoder(args.out_motion_dim, args.latent_dim)

    def forward(self, partial):
        z, mu, logvar = self.encoder(partial)
        pred = self.decoder(z)
        return pred, (z, mu, logvar)

# =================================
# =========== StyleVAE ============
# =================================
class ContentEncoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(latent_dim, 768, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.InstanceNorm1d(768)
        self.conv2 = nn.Conv1d(768, 512, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.InstanceNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.act = nn.ELU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.ln1(self.conv1(x))))
        x = self.drop(self.act(self.ln2(self.conv2(x))))
        x = self.drop(self.act(self.conv3(x)))
        return x

class StyleEncoder(nn.Module):
    def __init__(self, latent_dim, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(latent_dim, 768, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 768, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.act = nn.ELU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mu = nn.Conv1d(512, latent_dim, kernel_size=1)
        self.logvar = nn.Conv1d(512, latent_dim, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z_latent):
        x = self.drop(self.act(self.conv1(z_latent)))
        x = self.drop(self.act(self.conv2(x)))
        x = self.drop(self.act(self.conv3(x)))
        x = self.pool(x)
        mu = self.mu(x).squeeze(-1)
        logvar = self.logvar(x).squeeze(-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class AdaIN(nn.Module):
    def __init__(self, feat_dim, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(feat_dim, affine=False)
        self.style_proj = nn.Linear(style_dim, feat_dim * 2)

    def forward(self, x, style):
        x_norm = self.norm(x)
        style_params = self.style_proj(style) 
        gamma, beta = style_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * x_norm + beta

class StyleDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 768, kernel_size=3, padding=1)
        self.adain1 = AdaIN(768, latent_dim)
        self.conv2 = nn.Conv1d(768, 1024, kernel_size=3, padding=1)
        self.adain2 = AdaIN(1024, latent_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv3 = nn.Conv1d(1024, 768, kernel_size=3, padding=1)
        self.adain3 = AdaIN(768, latent_dim)
        self.conv4 = nn.Conv1d(768, 512, kernel_size=3, padding=1)
        self.adain4 = AdaIN(512, latent_dim)
        self.act = nn.ELU(inplace=True)

    def forward(self, x, z_style):
        x = self.act(self.adain1(self.conv1(x), z_style))
        x = self.act(self.adain2(self.conv2(x), z_style))
        x = self.upsample(x)
        x = self.act(self.adain3(self.conv3(x), z_style))
        x = self.act(self.adain4(self.conv4(x), z_style))
        return x

class StyleVAE(nn.Module):
    def __init__(self, motion_vae, args):
        super().__init__()
        self.motion_vae = motion_vae
        self.motion_vae.eval()  # freeze MotionVAE
        for p in self.motion_vae.parameters():
            p.requires_grad = False

        self.content_enc = ContentEncoder(args.latent_dim)
        self.style_enc = StyleEncoder(args.latent_dim)
        self.decoder = StyleDecoder(args.latent_dim)

    def forward(self, partial_A, partial_B, partial_C):
        partials = torch.cat([partial_A, partial_B, partial_C], dim=0)
        with torch.no_grad():
            _, z_all, _ = self.motion_vae.encoder(partials)
        z_A, z_B, z_C = torch.chunk(z_all, 3, dim=0)
            
        # --- Reconstruction ---
        # For A
        z_Ac = self.content_enc(z_A)
        z_As, mu_As, logvar_As = self.style_enc(z_A)
        z_Asty = self.decoder(z_Ac, z_As)
        # For B
        z_Bc = self.content_enc(z_B)
        z_Bs, mu_Bs, logvar_Bs = self.style_enc(z_B)
        z_Bsty = self.decoder(z_Bc, z_Bs)
        
        # --- Swap ---
        z_Cc = self.content_enc(z_C)
        z_Cs, mu_Cs, logvar_Cs = self.style_enc(z_C)
        z_t = self.decoder(z_Bc, z_Cs)
    
        # --- Cycle ---
        # For B
        z_tc = self.content_enc(z_t)
        z_Bcyc = self.decoder(z_tc, z_Bs)
        # For C
        z_ts, mu_ts, logvar_ts = self.style_enc(z_t)
        z_Ccyc = self.decoder(z_Cc, z_ts)
            
        # Stack and run Motion Decoder once
        decoder_inputs = torch.cat([z_Asty, z_Bsty, z_Bcyc, z_Ccyc], dim=0)
        decoder_outputs = self.motion_vae.decoder(decoder_inputs)
        m_Asty, m_Bsty, m_Bcyc, m_Ccyc = torch.chunk(decoder_outputs, 4, dim=0)
        
        ret = (
           z_A, z_Asty, m_Asty, mu_As, logvar_As,
           z_B, z_Bsty, m_Bsty, mu_Bs, logvar_Bs,
           z_Bcyc, m_Bcyc,
           z_C, mu_Cs, logvar_Cs, z_Ccyc, m_Ccyc,
           mu_ts, logvar_ts
        )
        
        return ret
        
# =================================
# ======= Loss Functions ==========
# =================================
def compute_joints_pos_loss(pred, true):
    true_pos = true[..., :3] 
    pred_pos = pred[..., :3]
    loss = F.l1_loss(pred_pos, true_pos, reduction='mean')
    return loss

def compute_joints_rot_loss(pred, true):
    true_rot = true[..., 3:9]   
    pred_rot = pred[..., 3:9]
    loss = F.l1_loss(pred_rot, true_rot, reduction='mean')
    return loss

def compute_joints_vel_loss(pred, true):
    true_vel = true[..., 9:12]   
    pred_vel = pred[..., 9:12]
    loss = F.l1_loss(pred_vel, true_vel, reduction='mean')
    return loss

def compute_root_loss(pred, motion):
    pred_root = pred[..., :12]
    true_root = motion[..., :12]
    loss = F.l1_loss(pred_root, true_root, reduction='mean')
    return loss
    
def compute_rec_loss(pred, motion):
    root_loss = compute_root_loss(pred, motion)
    # --- Reshape for Joints ---
    batch, frames, _ = pred.shape
    pred = pred[..., 12:].reshape(batch, frames, NUM_JOINTS - 1, 12)
    motion = motion[..., 15:].reshape(batch, frames, NUM_JOINTS - 1, 15)
    pos_loss = compute_joints_pos_loss(pred, motion)
    rot_loss = compute_joints_rot_loss(pred, motion)
    vel_loss = compute_joints_vel_loss(pred, motion)
    return root_loss + rot_loss + pos_loss + vel_loss

def compute_kl_loss(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

def compute_hsa_loss(mu_A, logvar_A, mu_B, logvar_B, emb_A, emb_B):
    kl = 0.5 * (
        (logvar_B - logvar_A)
        + (torch.exp(logvar_A) + (mu_A - mu_B).pow(2)) / torch.exp(logvar_B)
        - 1
    ).sum(-1)  # [B]
    # 2) cosine similarity as soft weight
    emb_A_n = F.normalize(emb_A, dim=-1)   # [B, E]
    emb_B_n = F.normalize(emb_B, dim=-1)
    sim = (emb_A_n * emb_B_n).sum(-1)      # [B], in [-1,1]
    # 3) clamp to [0,1] so only positive sims get weight
    weight = sim.clamp(min=0.0)            # [B]
    # 4) weighted mean of kl
    # if sum of weights is zero, fall back to unweighted mean
    wsum = weight.sum()
    if wsum.item() > 0:
        return (kl * weight).sum() / wsum
    else:
        return kl.mean()

def motion_vae_loss_function(pred, distr, motion):
    _, mu, logvar = distr
    # --- Reconstruction Loss ---
    rec_loss = compute_rec_loss(pred, motion)
    # --- Latent Loss ---
    kl_loss = compute_kl_loss(mu, logvar)
    return rec_loss, kl_loss

def style_vae_loss_function(data, motion_A, motion_B, motion_C, emb_A, emb_B):
    (
        z_A, z_Asty, m_Asty, mu_As, logvar_As,
        z_B, z_Bsty, m_Bsty, mu_Bs, logvar_Bs,
        z_Bcyc, m_Bcyc,
        z_C, mu_Cs, logvar_Cs,
        z_Ccyc, m_Ccyc,
        mu_ts, logvar_ts
    ) = data

    # --- Reconstruction Loss ---
    rec_motion = torch.cat([m_Asty, m_Bsty], dim=0)
    gt_motion = torch.cat([motion_A, motion_B], dim=0)
    rec_motion_loss = compute_rec_loss(rec_motion, gt_motion)
    rec_latent_loss = F.l1_loss(torch.cat([z_Asty, z_Bsty], dim=0),
                                 torch.cat([z_A, z_B], dim=0), reduction='mean')
    rec_loss = rec_motion_loss + rec_latent_loss

    # --- Cycle Consistency Loss ---
    cyc_motion = torch.cat([m_Bcyc, m_Ccyc], dim=0)
    cyc_gt_motion = torch.cat([motion_B, motion_C], dim=0)
    cyc_latents = torch.cat([z_Bcyc, z_Ccyc], dim=0)
    target_latents = torch.cat([z_B, z_C], dim=0)
    cyc_rec_motion_loss = compute_rec_loss(cyc_motion, cyc_gt_motion)
    cyc_latent_loss = F.l1_loss(cyc_latents, target_latents, reduction='mean')
    cyc_loss = cyc_rec_motion_loss + cyc_latent_loss

    # --- KL Loss ---
    mu_all = torch.cat([mu_As, mu_Bs, mu_Cs, mu_ts], dim=0)
    logvar_all = torch.cat([logvar_As, logvar_Bs, logvar_Cs, logvar_ts], dim=0)
    kl_loss = compute_kl_loss(mu_all, logvar_all)

    # --- HSA Loss ---
    hsa_loss = compute_hsa_loss(mu_As, logvar_As, mu_Bs, logvar_Bs, emb_A, emb_B)
    return rec_loss, cyc_loss, kl_loss, hsa_loss