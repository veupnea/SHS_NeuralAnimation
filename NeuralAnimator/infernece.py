import time
import torch
import numpy as np
from model import MotionVAE
from pymotion.ops.forward_kinematics import fk
import pymotion.rotations.ortho6d as sixd
from config import Args
from dataloader import read_and_process_bvh, chunk_motion_vectorized, GLOBAL_MUL, NUM_JOINTS

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
args = Args()
model = MotionVAE(args).to(device)
checkpoint = torch.load("./Models/MotionVAE_epoch_500.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def prepare_input_frame(root_pos, rot, offsets, parents, last):
    # Last
    last_root_pos, last_rot = last
    last_rot = last_rot.reshape(-1, last_rot.shape[0], 4)  # [T, J, 4]
    last_pos, _ = fk(last_rot, last_root_pos, offsets, parents)
    last_pos = last_pos[0]
    
    # Next
    rot = rot.reshape(-1, rot.shape[0], 4)  # [T, J, 6]    
    pos, _ = fk(rot, root_pos, offsets, parents)
    rot = sixd.from_quat(rot).reshape(NUM_JOINTS, 6)
    pos = pos[0]
    root_pos = root_pos[0]
    pos = pos - root_pos[None, :]  # Convert to local space by subtracting root position
    frame = np.concatenate([pos, rot], axis=-1)  # [J, 12]
    frame = torch.tensor(frame, dtype=torch.float32)
    return frame

def run_inference(bvh_file, out_path):
    # Load and chunk motion
    data, bvh_obj = read_and_process_bvh(bvh_file, is_inference=True)  # [T, J, D]
    chunks = chunk_motion_vectorized(data)
    parents, offsets = chunks[0]['parents'].numpy(), chunks[0]['offsets'].numpy()
    scale_factor = (1 / (data[-1] * GLOBAL_MUL))

    recon_pos = []
    recon_rot = []
    frame_times = []
    next_full = None

    with torch.no_grad():
        for i, sample in enumerate(chunks):
            current_partial = sample['current_partial']
            last_full = sample['last_full']
            root_pos = sample['root_pos'].numpy() # [3]
            delta_root = sample['root_pos_delta'].unsqueeze(0).numpy()

            # Start timing
            start_time = time.time()
            
            # Run
            if next_full is not None:
                last_full = next_full
            pred = model.infer(last_full, current_partial)
            pred_root_rot, pred_joints_rot = pred[0].numpy(), pred[1].numpy()
            all_rot = np.concatenate([pred_root_rot[None, :], pred_joints_rot], axis=0)  # [J, 6]            
            all_rot = sixd.to_quat(all_rot.reshape(all_rot.shape[0], 3, 2))  # Convert to quaternions
            
            # End timing
            end_time = time.time()
            frame_times.append((end_time - start_time) * 1000)  # ms

            # Reconstruct global position using root
            if i == 0:
                pos_global = root_pos[None, :]
            else:
                pos_global = recon_pos[-1] + delta_root
            
            if i > 0:
                last_for_next = (recon_pos[-1], recon_rot[-1])
                next_full = prepare_input_frame(pos_global, all_rot, offsets, parents, last_for_next)
            
            recon_pos.append(pos_global)
            recon_rot.append(all_rot)
            
    recon_pos = np.stack(recon_pos).squeeze(1)  # [T-1, 3]
    recon_rot = np.stack(recon_rot) # [T-1, J, 4]

    # Save to BVH
    local_rotations, local_positions, parents, offsets, end_sites, end_sites_parents = bvh_obj.get_data()
    offsets *= scale_factor
    end_sites *= scale_factor
    out_pos = np.zeros((recon_pos.shape[0], NUM_JOINTS, 3))
    out_pos[:, 0, :3] = recon_pos
    out_pos = out_pos[:-1] * scale_factor
    out_rot = recon_rot
    bvh_obj.set_data(out_rot, out_pos)
    bvh_obj.save(out_path)

if __name__ == "__main__":
    file = "./Data/SHS_Zup_Xfor/Ex1-Sub1-0.bvh"
    out_path = "./OutMotions/pred_motion.bvh"
    run_inference(file, out_path)
