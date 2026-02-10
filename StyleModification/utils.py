import os
import torch
import pymotion.rotations.quat_torch as quat_torch
import pymotion.rotations.ortho6d_torch as sixd_torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#================= Schedulers =================
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

#============ Transformations =================
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
    for i, parent in enumerate(parents[0]):
        if i == 0:
            continue
        mat[..., i, :, :] = torch.matmul(
            mat[..., parent, :, :].clone(),
            mat[..., i, :, :].clone(),
        )
    positions = mat[..., :3, 3]
    rotmats = mat[..., :3, :3]
    return positions, rotmats

def fk_quat_torch(rot, global_pos, offsets, parents):
    device = rot.device
    batch, frames, n_joints, _ = rot.shape
    # Convert 6D to rotation matrices
    local_rotmats = quat_torch.to_matrix(rot)
    
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
    for i, parent in enumerate(parents[0]):
        if i == 0:
            continue
        mat[..., i, :, :] = torch.matmul(
            mat[..., parent, :, :].clone(),
            mat[..., i, :, :].clone(),
        )
    positions = mat[..., :3, 3]
    rotmats = mat[..., :3, :3]
    return positions, rotmats

def compute_velocity_torch(positions, fps):
    dt = 1.0 / fps
    velocity = (positions[:, 1:] - positions[:, :-1]) / dt
    # Duplicate the first velocity along time dimension
    first_velocity = velocity[:, 0:1]
    velocity = torch.cat([first_velocity, velocity], dim=1)
    return velocity

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
    """
    Visualize 3D joint positions over time with contact joints colored red.

    Args:
        positions (np.ndarray): Shape [frames, joints, 3]
        contacts (np.ndarray): Shape [frames, joints], values 0.0 or 1.0
        joint_connections (list of tuples): Optional list of (joint_idx1, joint_idx2) to draw bones
        interval (int): Delay between frames in milliseconds
    """
    frames, joints, _ = positions.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set consistent view
    ax.set_xlim(np.min(positions[..., 0]), np.max(positions[..., 0]))
    ax.set_zlim(np.min(positions[..., 1]), np.max(positions[..., 1]))
    ax.set_ylim(np.min(positions[..., 2]), np.max(positions[..., 2]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initial scatter and lines
    joint_colors = ['blue'] * joints
    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 2], positions[0, :, 1], c=joint_colors, s=30)

    def update(frame):
        # Update scatter colors and positions
        colors = ['blue' for j in range(joints)]
        if contacts is not None:
            colors = ['red' if contacts[frame, j] else 'blue' for j in range(joints)]
        scatter._offsets3d = (
            positions[frame, :, 0],
            positions[frame, :, 2],
            positions[frame, :, 1]
        )
        scatter.set_color(colors)

        return [scatter]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.show()