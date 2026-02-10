from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat
from pymotion.ops.forward_kinematics import fk
import pymotion.rotations.ortho6d as sixd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from collections import defaultdict
import random
random.seed(123)
torch.random.manual_seed(123)
from tqdm import tqdm
import json

NUM_JOINTS = 19
PARTIAL_JOINTS = [3, 6, 14, 18] #['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand'] Root always included
NUM_PARTIAL_JOINTS = len(PARTIAL_JOINTS)
FPS = 30
POS_MUL = 0.01 # Scale factor for positions
PATHS = ["./Data/CMU", "./Data/SHS"]
CMU_FILES = 50 # Number of CMU files to use for training. Should be <500

class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, motion):
        self.data = motion

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, window_in_motion, window_partial, window_out_motion, root_motion, parents, offsets, encodings = self.data[idx]
        # Convert to tensors
        id = torch.tensor(id, dtype=torch.long)
        in_motion = torch.tensor(window_in_motion, dtype=torch.float32)
        partial = torch.tensor(window_partial, dtype=torch.float32)
        out_motion = torch.tensor(window_out_motion, dtype=torch.float32)
        root_motion = torch.tensor(root_motion, dtype=torch.float32)
        parents = torch.tensor(parents, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.float32)
        # Encodings is already a tensor
        return id, in_motion, partial, out_motion, root_motion, parents, offsets, encodings

class StyleDataset(Dataset):
    def __init__(self, data, batch_size, seq_multiplier=1):
        self.data = data
        self.batch_size = batch_size
        self.seq_multiplier = seq_multiplier

        # Build: sequence ID -> indices
        self.seq_to_indices = defaultdict(list)
        for idx, (seq_id, *_) in enumerate(data):
            self.seq_to_indices[seq_id].append(idx)

        # Keep only sequences with >= 2 samples
        self.valid_seq_ids = [sid for sid, idxs in self.seq_to_indices.items() if len(idxs) >= 2]
        self.num_valid_sequences = len(self.valid_seq_ids)

        # Precompute __len__
        self.total_len = self.num_valid_sequences * self.seq_multiplier * self.batch_size

        # For z3 sampling (any different sequence)
        self.all_seq_ids = list(self.seq_to_indices.keys())

    def __len__(self):
        return self.total_len

    def __getitem__(self, _):
        # Select valid seq with ≥2 samples
        seq_id = random.choice(self.valid_seq_ids)
        z1_idx, z2_idx = random.sample(self.seq_to_indices[seq_id], 2)

        # z3 from a different sequence
        other_seq_id = random.choice(self.all_seq_ids)
        while other_seq_id == seq_id:
            other_seq_id = random.choice(self.all_seq_ids)
        z3_idx = random.choice(self.seq_to_indices[other_seq_id])

        return (
            self._get_sample(z1_idx),
            self._get_sample(z2_idx),
            self._get_sample(z3_idx)
        )

    def _get_sample(self, idx):
        id, window_in_motion, window_partial, window_out_motion, root_motion, parents, offsets, encodings = self.data[idx]
        return (
            torch.tensor(id, dtype=torch.long),
            torch.tensor(window_in_motion, dtype=torch.float32),
            torch.tensor(window_partial, dtype=torch.float32),
            torch.tensor(window_out_motion, dtype=torch.float32),
            torch.tensor(root_motion, dtype=torch.float32),
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.float32),
            encodings  # already a tensor
        )

def compute_velocity(positions, fps):
    dt = 1.0 / fps
    velocity = (positions[1:] - positions[:-1]) / dt
    # Duplicate the first velocity to match input shape
    first_velocity = velocity[0:1]
    velocity = np.concatenate([first_velocity, velocity], axis=0)
    return velocity

def compute_angular_velocity(quaternions, fps):
    dt = 1.0 / fps  # Time step
    # Compute relative quaternion difference: q_{t}^{-1} * q_{t+1}
    q_inv = quaternions[:-1].copy()  # Copy quaternions except last frame
    q_inv[..., 1:] *= -1  # Quaternion inverse (negate x, y, z)
    delta_q = quat.mul(q_inv, quaternions[1:])  # Quaternion multiplication
    # Compute the angle θ from the quaternion
    delta_q_w = np.clip(delta_q[..., 0], -1.0, 1.0)  # Clamp to valid range
    angle = 2 * np.arccos(delta_q_w)  # Extract angle in radians
    # Compute the sine of half the angle to normalize the axis
    sin_half_angle = np.sqrt(1 - delta_q_w**2 + 1e-8)  # Avoid division by zero
    # Compute normalized rotation axis
    axis = delta_q[..., 1:] / sin_half_angle[..., np.newaxis]
    # Compute angular velocity: (θ / Δt) * axis
    angular_velocity = (angle[..., np.newaxis] / dt) * axis  # Shape [frames-1, joints, 3]
    # Duplicate first angular velocity to maintain the same shape
    angular_velocity = np.concatenate([angular_velocity[:1], angular_velocity], axis=0)
    return angular_velocity

def key_func(filename):
    filename = filename.split(".")[0]  # Remove file extension
    parts = filename.split("_")
    x = int(parts[0])
    y = int(parts[1])
    return (x, y)

def mirror_motion(pos, local_rotations, offsets, joint_names, FPS):
    # Define joint swapping map
    left_right_map = {
        'LeftUpLeg': 'RightUpLeg',
        'LeftLeg': 'RightLeg',
        'LeftFoot': 'RightFoot',
        'LeftShoulder': 'RightShoulder',
        'LeftArm': 'RightArm',
        'LeftForeArm': 'RightForeArm',
        'LeftHand': 'RightHand',
    }
    name_to_index = {name: i for i, name in enumerate(joint_names)}
    mirror_indices = np.arange(len(joint_names))
    for left, right in left_right_map.items():
        li = name_to_index[left]
        ri = name_to_index[right]
        mirror_indices[li] = ri
        mirror_indices[ri] = li
    # Mirror positions (flip X-axis)
    mirrored_pos = pos.copy()
    mirrored_pos[..., 0] *= -1
    mirrored_pos = mirrored_pos[:, mirror_indices, :]
    # Mirror rotations
    mirrored_rot = local_rotations.copy()
    if mirrored_rot.shape[-1] == 4:  # quaternion
        mirrored_rot[..., [1, 2]] *= -1
        mirrored_rot = mirrored_rot[:, mirror_indices, :]
    elif mirrored_rot.shape[-1] == 6:  # 6D
        quat_rot = sixd.to_quat(mirrored_rot.reshape(-1, NUM_JOINTS, 3, 2))
        quat_rot[..., [1, 2]] *= -1
        quat_rot = quat_rot[:, mirror_indices, :]
        mirrored_rot = sixd.from_quat(quat_rot).reshape(-1, NUM_JOINTS, 6)
    else:
        raise ValueError("Unsupported rotation format.")
    # Mirror offsets if needed (optional)
    mirrored_offsets = offsets.copy()
    mirrored_offsets[:, 0] *= -1
    mirrored_offsets = mirrored_offsets[mirror_indices]
    # Compute mirrored velocities
    mirrored_vel = compute_velocity(mirrored_pos, FPS)
    mirrored_ang_vel = compute_angular_velocity(local_rotations, FPS)
    # Pack final data
    mirrored_data = np.concatenate([mirrored_pos, mirrored_rot, mirrored_vel, mirrored_ang_vel], axis=-1)
    return mirrored_data, mirrored_offsets

def load_files(paths):
    animation_files = []

    for dir in paths:
        is_cmu = "CMU" in dir # check if this is a CMU path

        for dirpath, _, filenames in os.walk(dir):
            for i, filename in enumerate(filenames):
                if is_cmu:
                    if i >= CMU_FILES:  # Limit number of CMU files
                        break
                full_path = os.path.join(dirpath, filename)
                animation_files.append(full_path)
            
    random.shuffle(animation_files)
    return animation_files

def preprocess_bvh_files(seq, motion_vae=None, overlap=0.25):
    all_data = {}
    out_data = []
    all_pos, all_rot, all_vel, all_angular_vel = [], [], [], []
    duration = seq
    if motion_vae is not None:
        overlap = 0.0  # No overlap for Style training
    step_size = int(duration * (1 - overlap))
    files = load_files(PATHS)
    files = files[:int(len(files) * 0.9)]  # Use 90% of the files for training

    # --- Load BVH FILES ---
    count = 0
    for file in tqdm(files[:], desc="Processing BVH"):
        bvh = BVH()
        bvh.load(file)
        
        local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()  
        local_positions *= POS_MUL
        offsets *= POS_MUL
        # Get global positions
        global_positions = local_positions[:, 0, :]
        pos, rotmat = fk(local_rotations, global_positions, offsets, parents)
              
        # Original Data
        velocities = compute_velocity(pos, FPS)   
        angular_velocities = compute_angular_velocity(local_rotations, FPS)
        local_rotations = sixd.from_quat(local_rotations).reshape(-1, NUM_JOINTS, 6)
        data = np.concatenate([pos, local_rotations, velocities, angular_velocities], axis=-1)
        all_pos.append(pos)
        all_rot.append(local_rotations)
        all_vel.append(velocities)
        all_angular_vel.append(angular_velocities)
        all_data[file] = (count, data, offsets)
        # Mirrored data for motion_vae training only
        if motion_vae is None:
            mirrored_data, mirrored_offsets = mirror_motion(pos, local_rotations, offsets, bvh.data['names'], FPS)
            all_data[file + '_mirrored'] = (count, mirrored_data, mirrored_offsets)
        count += 1
    
    print("Total files loaded:", count)
    
    # --- Compute motion samples ---
    for file, (id, data, offsets) in all_data.items():
        for i in range(0, len(data) - duration, step_size):
            limits = (i, i + duration)
            # Preprocess
            motion = data[limits[0]:limits[1]].copy()
            init_pose = motion[0, 0, :9].copy()
            root_motion = motion[:, :, :9].copy()
            motion[:, :, :3] -= init_pose[:3]  # Center root position of every window to 0,0,0
            partial_motion = motion.copy()[:, PARTIAL_JOINTS]
            # Root
            root_in_data = motion[:, 0, :]
            root_out_data = motion[:, 0, :12]
            # In-Motion
            joints_in_data = motion[:, 1:, :]
            window_in_motion = np.concatenate([root_in_data, joints_in_data.reshape(joints_in_data.shape[0], -1)], axis=1)           
            # Partial
            joints_partial_data = partial_motion[:, :, :12]
            window_partial = np.concatenate([root_out_data, joints_partial_data.reshape(joints_in_data.shape[0], -1)], axis=1)
            # Out-Motion
            joints_out_data = motion[:, 1:, :12]
            window_out_motion = np.concatenate([root_out_data, joints_out_data.reshape(joints_out_data.shape[0], -1)], axis=1)
            # Compute motion encoding
            encoding = torch.tensor(np.zeros((128)), dtype=torch.float32)
            if motion_vae is not None:
                with torch.no_grad():
                    encoding = motion_vae.encoder(torch.tensor(window_partial, dtype=torch.float32).unsqueeze(0).to("cuda"))[1][0] # Get mu, not z
                encoding = encoding.mean(-1)
            out_data.append((id, window_in_motion, window_partial, window_out_motion, root_motion, parents, offsets, encoding))

    return out_data

def collate_triplets(batch):
    # Unpack into 3 lists of (8-tensor tuples)
    z1_list, z2_list, z3_list = zip(*batch)
    # Helper to stack each element of the tuple across batch
    def stack_tuples(list_of_tuples):
        return tuple(torch.stack(tensors) for tensors in zip(*list_of_tuples))
    z1 = stack_tuples(z1_list)
    z2 = stack_tuples(z2_list)
    z3 = stack_tuples(z3_list)
    return z1, z2, z3

def create_data_loaders(processed_data, batch_size, train_ratio=0.8, is_motion=False):
    if is_motion:
        dataset = MotionDataset(processed_data)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        # Use regular batching for validation
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return dataset, train_loader, val_loader

    # Is Style training
    # Step 1: Split the raw data before wrapping it
    train_size = int(train_ratio * len(processed_data))
    val_size = len(processed_data) - train_size
    train_data_raw, val_data_raw = random_split(processed_data, [train_size, val_size])
    # Step 2: Wrap with custom triplet-style dataset class
    train_dataset = StyleDataset(train_data_raw, batch_size)
    val_dataset = StyleDataset(val_data_raw, batch_size)
    # Step 3: Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_triplets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_triplets)
    return train_dataset, train_loader, val_loader

def load_data(seq=1, batch=1, motion_vae=None):
    all_data = preprocess_bvh_files(seq, motion_vae)
    dataset, train_loader, val_loader = create_data_loaders(all_data, batch, is_motion=motion_vae is None)
    return dataset, train_loader, val_loader

def preprocess_bvh_file(filepath, seq, overlap=0.5):
    all_data = []
    duration = seq
    step_size = int(duration * (1 - overlap))
    bvh = BVH()
    bvh.load(filepath)
    
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh.get_data()   
    local_positions *= POS_MUL
    offsets *= POS_MUL
    # Get global positions
    global_positions = local_positions[:, 0, :]
    pos, rotmat = fk(local_rotations, global_positions, offsets, parents)  
    velocities = compute_velocity(pos, FPS)   
    angular_velocities = compute_angular_velocity(local_rotations, FPS)
    local_rotations = sixd.from_quat(local_rotations).reshape(-1, NUM_JOINTS, 6)
    data = np.concatenate([pos, local_rotations, velocities, angular_velocities], axis=-1)
    
    # --- Generate motion samples with overlapping ---
    for i in range(0, len(data) - duration, step_size):
        limits = (i, i + duration)
        # Preprocess
        motion = data[limits[0]:limits[1]].copy()
        init_pose = motion[0, 0, :9].copy()
        root_motion = motion[:, :, :9].copy()
        motion[:, :, :3] -= init_pose[:3]  # Center root position of every window to 0,0,0
        partial_motion = motion.copy()[:, PARTIAL_JOINTS]
        # Root
        root_in_data = motion[:, 0, :]
        root_out_data = motion[:, 0, :12]
        # In-Motion
        joints_in_data = motion[:, 1:, :]
        window_in_motion = np.concatenate([root_in_data, joints_in_data.reshape(joints_in_data.shape[0], -1)], axis=1)           
        # Partial
        joints_partial_data = partial_motion[:, :, :12]
        window_partial = np.concatenate([root_out_data, joints_partial_data.reshape(joints_in_data.shape[0], -1)], axis=1)
        # Out-Motion
        joints_out_data = motion[:, 1:, :12]
        window_out_motion = np.concatenate([root_out_data, joints_out_data.reshape(joints_out_data.shape[0], -1)], axis=1)
        # Compute motion encoding
        encoding = torch.tensor(np.zeros((128)), dtype=torch.float32)
        all_data.append((0, window_in_motion, window_partial, window_out_motion, root_motion, parents, offsets, encoding))
    
    return all_data, bvh

def load_single_file(dir, filename, seq):
    filepath = os.path.join(dir, filename)
    data, bvh = preprocess_bvh_file(filepath, seq)
    dataset = MotionDataset(data)
    return DataLoader(dataset, batch_size=1, shuffle=False), bvh
