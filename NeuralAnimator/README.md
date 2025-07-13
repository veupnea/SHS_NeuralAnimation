# Motion Prediction Model

This repository contains code for a motion prediction model, allowing you to train a new model or generate motion from BVH files.
### For the integration with Unreal follow the instructions in "Unreal_Integration.txt".

---

## Getting Started

1.  **Install Python Libraries**:
    Install the necessary libraries from `requirements.txt`. It's recommended to use **PyTorch with CUDA** if your system supports it for optimal performance.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Training the Model**:
    To retrain the model, execute the `train.py` script:

    ```bash
    python train.py
    ```

3.  **Generating Motion (Inference)**:
    To generate motion from a BVH file, run the `inference.py` script:

    ```bash
    python inference.py
    ```

---

## General Training Pipeline

The training process involves processing motion data and optimizing the model through various loss functions.

1.  **Load and Process Motion**: Motion data from each BVH file is loaded and prepared for training.
2.  **Chunk Motions**: Motions are segmented into 2-frame samples.
3.  **Training Step**: For each training iteration:
    * **Encoder Input**: The last full pose and current full pose are passed to the **encoder** to obtain `mu` and `logvar`.
    * **Prior Network Input**: The last full pose and current **partial pose** (root and End Effectors) are fed into the **prior network** to retrieve another set of `mu` and `logvar`.
    * **Latent Variable (`z`)**: `z` is sampled from the encoder's `mu` and `logvar`.
    * **Decoder Prediction**: `z` and the current **partial pose** are passed to the **decoder** to predict the full current pose (rotations).
4.  **Losses**:
    * **Reconstruction Loss**: Measures the difference between the predicted and actual full current poses.
    * **KL Divergence Loss**: Aligns the latent spaces of the encoder and prior network.

---

## General Inference Pipeline

During inference, the encoder is discarded, and only the prior network and decoder are used to generate motion autoregressively.

1.  **Prior Network Input**: The last full pose and current **partial pose** (root and End Effectors) are passed to the **prior network** to retrieve `mu` and `logvar`.
2.  **Latent Variable (`z`)**: `z` is sampled from `mu` and `logvar`.
3.  **Decoder Prediction**: The **decoder** generates the prediction of the next full pose.
4.  **Autoregressive Feedback**: Forward kinematics (FK) are applied to obtain positions, allowing the predicted pose to be fed back into the model for predicting the subsequent frame.

---

## Data Structure

The model processes poses with specific coordinate and rotation conventions:

* **Last Full Pose**:
    * Root position centered at (0,0,0).
    * Other joint positions converted to root space.
    * Root rotations in global space, other joint rotations in local space (6D rotations).

* **Next Full Pose**:
    * Root pose relative to the last full pose.
    * Other joint positions converted to root space.
    * Root rotations in global space, other joint rotations in local space (6D rotations).

* **Next Partial Pose**:
    * Positions for the root (global, relative to last pose).
    * Positions for hands and feet (local, relative to root).

---

## Additional Notes

* All BVH motions are converted to **Unreal Engine's coordinate system** (Z-up, X-forward).
* All BVH values are represented in **meters**.
* A **custom 19-joint skeleton** is utilized. Using more joints may reduce performance.
