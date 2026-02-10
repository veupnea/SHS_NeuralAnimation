import os
import torch
import numpy as np
from autoencoder import MotionVAE, StyleVAE
from config import Args
from dataloader import load_single_file, POS_MUL, NUM_JOINTS
import pymotion.rotations.ortho6d as sixd
from scipy.ndimage import gaussian_filter1d

# Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = Args()
model_m = MotionVAE(args).to(device)
checkpoint_m = torch.load("./Models/SHS+CMU-Motion_epoch_1000.pth", map_location=device, weights_only=False)
model_m.load_state_dict(checkpoint_m['state_dict'])
model_m.eval()
model_s = StyleVAE(model_m, args).to(device)
checkpoint_s = torch.load("./Models/SHS+CMU-Style_epoch_300.pth", map_location=device, weights_only=False)
model_s.load_state_dict(checkpoint_s['state_dict'])
model_s.eval()

# Load Animation File
dir = "./Data/CMU"
filename = "15_05.bvh"
loader, bvh = load_single_file(dir=dir, filename=filename, seq=args.seq)

def run(motion):
    z, _, _ = model_m.encoder(motion)
    z_c = model_s.content_enc(z)
    z_s, _, _ = model_s.style_enc(z)
    z_s += torch.randn_like(z_s) * 0.1
    z_stylized = model_s.decoder(z_c, z_s)
    stylized_motion = model_s.motion_vae.decoder(z_stylized)
    return stylized_motion

def smooth(values, sigma=2):
    return gaussian_filter1d(values, sigma, axis=0)

window_size = args.seq
overlap = window_size // 2
init, z, blended_motion = None, None, None

with torch.no_grad():
    for i, sample in enumerate(loader):
        _, motion, partial, out_motion, root_motion, _, _, _ = sample
        motion, partial, out_motion, root_motion = motion.to(device), partial.to(device), out_motion.to(device), root_motion.to(device)
        
        if i == 0:
            init = root_motion[:, 0].cpu().numpy()
            pred_motion = run(partial)
            pred_motion[:, :, :3] += root_motion[:, 0, 0, :3]
            pred_np = pred_motion[0].cpu().numpy()
            blended_motion = pred_np  # initialize with the first window
        else:
            pred_motion = run(partial)
            pred_motion[:, :, :3] += root_motion[:, 0, 0, :3]
            pred_np = pred_motion[0].cpu().numpy()
            # Extract the overlapping parts:
            prev_overlap = blended_motion[-overlap:]
            curr_overlap = pred_np[:overlap]
            w_prev = np.linspace(1, 0, overlap)[:, None]
            w_curr = np.linspace(0, 1, overlap)[:, None]
            # Blend the overlapping frames
            blended_overlap = w_prev * prev_overlap + w_curr * curr_overlap
            # Replace the overlapping region in the current blended sequence:
            blended_motion[-overlap:] = blended_overlap
            blended_motion = np.concatenate((blended_motion, pred_np[overlap:]), axis=0)

full_motion = blended_motion
# --- Root ---
root_pos = full_motion[:, :3] * (1.0 / POS_MUL)
root_rot = sixd.to_quat(full_motion[:, 3:9].reshape(-1, 3, 2))
# --- Joints ---
joints_motion = full_motion[:, 12:].reshape(-1, NUM_JOINTS - 1, 12)
joints_rot = joints_motion[..., 3:9].reshape(-1, NUM_JOINTS - 1, 3, 2)
root_pos = smooth(root_pos, sigma=2)
joint_rot = smooth(joints_rot, sigma=2)
joint_rot = sixd.to_quat(joints_rot)
out_rot = np.concatenate([root_rot.reshape(-1, 1, 4), joint_rot], axis=1)
# --- Out ---
local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
offsets *= 1.0 / POS_MUL
out_pos = np.zeros((out_rot.shape[0], NUM_JOINTS, 3))
out_pos[:, 0, :3] = root_pos

bvh.set_data(out_rot, out_pos)
outpath = f"./OutMotions/{filename[:5]}_variation.bvh"
os.makedirs("./OutMotions", exist_ok=True)
bvh.save(outpath)