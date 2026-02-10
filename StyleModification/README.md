# Style Modification

This project allows for the modification of motion style in BVH files with partial joints. It uses a two-stage VAE approach: a `MotionVAE` to learn a latent representation of motion, and a `StyleVAE` to separate content and style from this latent representation.



## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Unzip the data:**
    Extract `CMU.zip` and `SHS.zip` into the `Data/` directory.

## Training

### 1. Train the MotionVAE

The `MotionVAE` is trained to reconstruct motion from a latent representation.

```bash
python train_motion_vae.py
```

This will save a model file in the `Models/` directory.

### 2. Train the StyleVAE

The `StyleVAE` is trained to separate content and style from the `MotionVAE`'s latent representation. It requires a pre-trained `MotionVAE` model.

Make sure the `motion_vae` path in `train_style_vae.py` points to the correct `MotionVAE` model.

```python
# train_style_vae.py
...
checkpoint = torch.load("./Models/SHS+CMU-Motion_epoch_1000.pth", map_location=device, weights_only=False)
...
```

Then run the training script:

```bash
python train_style_vae.py
```

This will save a `StyleVAE` model in the `Models/` directory.

## Inference

The `inference.py` script runs style modification on a BVH file.

1.  **Configure the input file:**
    In `inference.py`, set the `dir` and `filename` variables to point to the BVH file you want to process.

    ```python
    # inference.py
    ...
    dir = "./Data/CMU"
    filename = "15_05.bvh"
    ...
    ```

2.  **Run the script:**

    ```bash
    python inference.py
    ```

The output will be saved as a new BVH file in the `OutMotions/` directory.

## Pre-trained Models

The `Models/` directory contains the following pre-trained models:

- **SHS+CMU-Motion_epoch_1000.pth**: A `MotionVAE` model trained for 1000 epochs on the SHS and CMU datasets.
- **SHS+CMU-Style_epoch_300.pth**: A `StyleVAE` model trained for 300 epochs on the SHS and CMU datasets.

These models are used by default in the `train_style_vae.py` and `inference.py` scripts.

## Reference

The current implementation is based on the following work:

Guo, C., Mu, Y., Zuo, X., Dai, P., Yan, Y., Lu, J., & Cheng, L. (2024). Generative Human Motion Stylization in Latent Space. ArXiv, abs/2401.13505.

