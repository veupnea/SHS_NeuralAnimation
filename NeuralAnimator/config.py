from dataloader import NUM_JOINTS, NUM_PARTIAL_JOINTS

class Args:
    track_wandb = False
    epochs = 500
    batch_size = 2048
    lr = 0.0001
    latent_dim = 32
    hidden_dim = 128
    feat_dim = 9 # [local(3), rot(6)]
    partial_feat_dim = 3 # [local(3)]
    partial_dim = NUM_PARTIAL_JOINTS * partial_feat_dim
    out_dim = NUM_JOINTS * 6
    
    @staticmethod
    def to_dict():
        return {key: getattr(Args, key) for key in dir(Args) if not key.startswith('__') and not callable(getattr(Args, key))}