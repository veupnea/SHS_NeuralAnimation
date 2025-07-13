import torch
import torch.onnx
from model import MotionVAE
from config import Args
from dataloader import NUM_JOINTS

# Define dummy input tensors matching the infer() input signature
args = Args()
batch_size = 1
num_joints = NUM_JOINTS
feat_dim = 3
partial_dim = args.partial_dim

last_full_dummy = torch.randn(batch_size, num_joints, feat_dim)
current_partial_dummy = torch.randn(batch_size, partial_dim)

# Define a wrapper module for export
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, last_full, current_partial_flat):
        mu, logvar = self.model.prior(last_full, current_partial_flat)
        z = self.model.reparameterize(mu, logvar)
        recon = self.model.decoder(z, current_partial_flat)
        return recon
    
# Instantiate the MotionVAE and wrapper
model = MotionVAE(args)
checkpoint = torch.load("./Models/MotionVAE_epoch_500.pth", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
wrapper = InferenceWrapper(model)

# Export to ONNX
onnx_path = "./MotionVAE.onnx"
dummy_input_last = last_full_dummy
dummy_input_partial = current_partial_dummy

torch.onnx.export(
    wrapper,
    (dummy_input_last, dummy_input_partial),
    onnx_path,
    input_names=['last_full', 'current_partial'],
    output_names=['reconstructed_pose'],
    dynamic_axes={
        'last_full': {0: 'batch'},
        'current_partial': {0: 'batch'},
        'reconstructed_pose': {0: 'batch'}
    },
    opset_version=16
)
