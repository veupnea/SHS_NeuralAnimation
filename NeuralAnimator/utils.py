import os
import torch
import numpy as np
import pymotion.rotations.ortho6d_torch as sixd_torch
import pymotion.rotations.ortho6d as sixd

# ================== Utils ==================
def compute_avg_bone_length(offsets):
    """
    Compute average bone length from joint offsets.
    Args:
        offsets (np.ndarray): (J, 3)
    Returns:
        float: mean bone length
    """
    return np.linalg.norm(offsets, axis=1).mean()

def convert_to_unreal_coords(xyz):
    """
    Convert coordinates from CMU (+Z forward, Y-up) to Unreal (+X forward, Z-up).
    Args:
        xyz: np.ndarray of shape (..., 3)
    Returns:
        Transformed coordinates in Unreal system.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    return np.stack([z, x, y], axis=-1)  # Unreal: X=Z, Y=X, Z=Y

def compute_velocity_np(positions, fps):
    dt = 1.0 / fps
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)  # central
    velocity[-1] = (positions[-1] - positions[-2]) / dt           # backward
    return velocity

def vel_global_to_local_np(global_vel, root_rot_6d, root_id):
    """
    Convert global joint velocities to local root space using root's 6D rotation.
    
    Args:
        global_vel: np.ndarray of shape (T, J, 3)
        root_rot_6d: np.ndarray of shape (T, J, 6)
        root_id: index of the root joint
        
    Returns:
        Local-space velocity: np.ndarray of shape (T, J, 3)
    """
    T, J, _ = global_vel.shape
    # Get root rotation matrix: (T, 3, 3)
    root_rot = root_rot_6d[:, root_id]  # (T, 6)
    root_rot_mat = sixd.to_matrix(root_rot.reshape(T, 3, 2))  # Convert to rotation matrices (T, 3, 3)
    root_rot_inv = np.transpose(root_rot_mat, (0, 2, 1))  # inverse rotation: transpose
    # Rotate each velocity vector: v_local = R^-1 @ v_global
    # We vectorize this using einsum
    # global_vel: (T, J, 3), root_rot_inv: (T, 3, 3)
    vel_local = np.einsum('tij,tbj->tbi', root_rot_inv, global_vel)
    return vel_local 

def geodesic_so3_loss(pred6d, target6d):
    # both are [..., 6]
    R_pred = sixd_torch.to_matrix(pred6d.reshape(-1, 3, 2))
    R_gt   = sixd_torch.to_matrix(target6d.reshape(-1, 3, 2))
    # R_rel = R_pred^T @ R_gt
    R_rel = torch.matmul(R_pred.transpose(-2,-1), R_gt)
    # trace across last two dims:
    tr = R_rel[...,0,0] + R_rel[...,1,1] + R_rel[...,2,2]
    cos = torch.clamp((tr - 1) / 2, -1+1e-6, 1-1e-6)
    # angle in radians
    return torch.acos(cos).mean()

def fk_6d_torch(rot, global_pos, offsets, parents):
    device = rot.device
    batch, frames, n_joints, _ = rot.shape
    # Convert 6D to rotation matrices
    local_rotmats = sixd_torch.to_matrix(rot.reshape(batch, frames, n_joints, 3, 2))
    # Create homogeneous matrix (..., n_joints, 4, 4)
    mat = torch.zeros(rot.shape[:-1] + (4, 4), device=device, dtype=rot.dtype,)
    mat[..., :3, :3] = local_rotmats
    offsets = offsets.unsqueeze(1)           # (batch, 1, n_joints, 3)
    offsets = offsets.expand(-1, frames, -1, -1)  # (batch, frames, n_joints, 3)
    mat[..., :3, 3] = offsets
    mat[..., 3, 3] = 1
    # First joint is global position
    mat[..., 0, :3, 3] = global_pos
    # Forward kinematics chain
    for i, parent in enumerate(parents):
        if i == 0:
            continue
        mat[..., i, :, :] = torch.matmul(
            mat[..., parent, :, :].clone(),
            mat[..., i, :, :].clone(),
        )
    positions = mat[..., :3, 3]
    rotmats = mat[..., :3, :3]
    return positions, rotmats

def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=8, ratio=0.5):
    """
    Generate a cyclical linear annealing schedule for KL weight (beta).
    
    Args:
        n_iter (int): Total number of iterations (e.g., epochs).
        start (float): Starting value of beta (e.g., 0.0).
        stop (float): Maximum value of beta (e.g., 1.0).
        n_cycle (int): Number of cycles.
        ratio (float): Fraction of each cycle used for the ramp-up phase.
    
    Returns:
        np.ndarray: Array of beta values for each iteration.
    """
    L = np.ones(n_iter) * stop  # Initialize with the maximum value
    period = n_iter / n_cycle   # Length of each cycle
    ramp_up = period * ratio    # Ramp-up duration (half the cycle)
    step = (stop - start) / ramp_up  # Linear step size during ramp-up

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v  # Assign ramp-up values
            v += step
            i += 1
    return L

# ================= Saving ==================
def save_checkpoint(model, optimizer, epoch, args, filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_directory, "Models")
    path = os.path.join(path, filename)
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'hyperparameters': args
    }
    torch.save(checkpoint, path)
    return path