from dataloader import NUM_JOINTS, NUM_PARTIAL_JOINTS

class Args:
    track_wandb = False
    wandb_key = "YOUR_WANDB_KEY_HERE"
    epochs = 1000
    style_epochs = 300
    batch_size = 128
    style_batch_size = 128
    lr = 0.0001
    style_lr = 0.0001
    seq = 64 # NUMBER OF FRAMES IN WINDOW
    # MotionAE
    latent_dim = 512
    m_beta = 0.00001
    # Style VAE
    s_beta = 0.005
    gamma = 0.5
    delta = 0.01
    # Input
    motion_dim = 15 + (NUM_JOINTS - 1) * 15
    partial_dim = 12 + NUM_PARTIAL_JOINTS * 12
    out_motion_dim = 12 + (NUM_JOINTS - 1) * 12
    
    @staticmethod
    def to_dict():
        return {key: getattr(Args, key) for key in dir(Args) if not key.startswith('__') and not callable(getattr(Args, key))}