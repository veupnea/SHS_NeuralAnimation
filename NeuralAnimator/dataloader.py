from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd
from pymotion.ops.forward_kinematics import fk
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from utils import compute_velocity_np, compute_avg_bone_length
import random
random.seed(123)

ROOT_ID = 0
PARTIAL_JOINTS = [0, 3, 6, 14, 18]  # pelvis, feet, hands
NUM_JOINTS = 19
NUM_PARTIAL_JOINTS = 5
GLOBAL_MUL = 0.1
AVG_BONE_LENGTH = 3.64
FPS = 30
CMU_FILES = 250 # Number of CMU files to use for training
PATHS = ["./Data/CMU_Zup_Xfor", "./Data/SHS_Zup_Xfor"]

def read_and_process_bvh(file, is_inference=False):
    bvh = BVH()
    bvh.load(file)
    # Load motion data
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()
    
    # -- Normalize scale --
    scale_factor = AVG_BONE_LENGTH / compute_avg_bone_length(offsets)
    local_positions *= (scale_factor * GLOBAL_MUL)  # Scale positions
    offsets *= (scale_factor * GLOBAL_MUL)
    end_sites *= (scale_factor * GLOBAL_MUL)
    # -- Forward Kinematics --
    global_positions = local_positions[:, 0, :]  # root (hips)
    pos, _ = fk(local_rotations, global_positions, offsets, parents)  # global joint positions
    # -- Convert rotations to 6D --
    local_rotations = sixd.from_quat(local_rotations).reshape(-1, NUM_JOINTS, 6)
    # -- Compute velocities --
    if not is_inference:
        pos -= pos[0, ROOT_ID:ROOT_ID+1, :]  # Center root at origin
    velocities = compute_velocity_np(pos, FPS)
    # -- Concatenate full feature vector: [positions | 6D rot | velocities]
    data = np.concatenate([pos, local_rotations, velocities], axis=-1)
    return (data, parents, offsets, scale_factor), bvh
    
def chunk_motion_vectorized(bvh_data, frame_gap=1):
    data, parents, offsets, scale_factor = bvh_data

    # split into pos/rot/vel
    pos = data[:, :, :3]      # (T,19,3)
    rot = data[:, :, 3:9]     # (T,19,6)
    vel = data[:, :, 9:12]    # (T,19,3)

    # slice out previous vs current
    prev_pos = pos[:-frame_gap]      # (T-1,19,3)
    prev_rot = rot[:-frame_gap]      # (T-1,19,6)
    prev_vel = vel[:-frame_gap]      # (T-1,19,3)

    cur_pos = pos[frame_gap:]      # (T-1,19,3)
    cur_rot = rot[frame_gap:]      # (T-1,19,6)
    cur_vel = vel[frame_gap:]      # (T-1,19,3)

    # compute local coords (broadcasting over joints)
    root_pos_prev = prev_pos[:, ROOT_ID]               # (T-1,3)
    loc_prev = prev_pos - root_pos_prev[:,None,:]      # (T-1,19,3)

    root_pos_cur = cur_pos[:, ROOT_ID]
    root_pos_target = root_pos_cur - root_pos_prev  # (T-1,3)
    root_vel_cur = cur_vel[:, ROOT_ID]
    loc_cur = cur_pos - root_pos_cur[:,None,:]
    loc_cur[:, ROOT_ID] = root_pos_target
    last_full = np.concatenate([loc_prev, prev_rot], axis=-1)  # (T-1,19,12)
    current_full = np.concatenate([loc_cur, cur_rot], axis=-1)  # (T-1,19,12)

    PJ = np.array(PARTIAL_JOINTS, dtype=int)
    last_partial = np.concatenate([loc_prev[:, PJ]], axis=-1)  # (T-1,P,6)
    current_partial = np.concatenate([loc_cur[:, PJ]], axis=-1)  # (T-1,P,6)

    # to torch tensors
    t = lambda x: torch.from_numpy(x).float()
    batch = {
        'last_full':       t(last_full),         # (T-1, J, 12)
        'last_partial':    t(last_partial),      # (T-1, P, 6)
        'current_full':    t(current_full),      # (T-1, J, 12)
        'current_partial': t(current_partial),   # (T-1, P, 6)
        'root_pos_delta':  t(root_pos_target),    # (T-1, 3)
        'root_pos':        t(root_pos_cur),      # (T-1, 3)
        'root_vel':        t(root_vel_cur),      # (T-1, 3)
    }
    
    offsets = t(offsets)
    parents = torch.tensor(parents, dtype=torch.int64)
    bone_lengths = torch.norm(offsets, dim=-1)  # (J,)
    N = batch['last_full'].shape[0]                
    chunks = [
        { k: v[i] for k,v in batch.items()} for i in range(N)
    ]
    for c in chunks:
        c['parents'] = parents
        c['offsets'] = offsets
        c['bone_lengths'] = bone_lengths
    return chunks
   
class MotionDataset(Dataset):
    def __init__(self, file_paths, prev_window=1):
        self.samples = []

        for path in tqdm(file_paths, desc="Preprocessing data"):
            bvh_data, bvh_obj = read_and_process_bvh(path)
            chunks = chunk_motion_vectorized(bvh_data, prev_window)
            self.samples.extend(chunks)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def load_files(paths, extensions=(".bvh",)):
    animation_files = []
    for i, dir in enumerate(paths):
        for dirpath, _, filenames in os.walk(dir):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    full_path = os.path.join(dirpath, filename)
                    animation_files.append(full_path)
        if i == 0:
            random.shuffle(animation_files)  # Shuffle CMU files    
            animation_files = animation_files[:CMU_FILES]
            
    random.shuffle(animation_files)  # Shuffle the files
    return animation_files
        
def create_dataloaders(batch_size=64, split=0.8, shuffle=True):
    files = load_files(PATHS)
    split_idx = int(len(files) * split)
    train_files = files[:split_idx]
    eval_files = files[split_idx:]
    train_dataset = MotionDataset(train_files)
    eval_dataset = MotionDataset(eval_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader
