import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import wget
from warnings import warn
import constants as const
from network_model import NerfNet, ImagePoseDataset, compute_sample_rays, render_rays
import skimage
import time

# Load Data
data_file = os.path.join('data', 'tiny_nerf_data.npz')
if not os.path.exists(data_file):  # get tiny nerf data
    wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz", out=data_file)

data = np.load(data_file)
images = torch.from_numpy(data['images']).type(const.DTYPE)
poses = torch.from_numpy(data['poses']).type(const.DTYPE)
focal = torch.from_numpy(data['focal']).type(const.DTYPE)  # set to foal_len = torch.from...
H, W = images.shape[1:3]  # set to height, width = images.shape[1:3]
ip_dataset = ImagePoseDataset(images, poses)
near = const.NEAR_FRUSTUM  # remove
far = const.FAR_FRUSTUM  # remove

height = H  # remove
width = W  # remove
focal_len = focal  # remove
num_samples = const.NUM_RAY_SAMPLES  # remove
N_samples = num_samples  # remove
L_embed = const.L_ENCODE  # remove


model_file = os.path.join(const.MODEL_SAVE_DIR, 'nerf_net')
model = NerfNet().type(const.DTYPE)
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))


target_img = images[0,...]
pose_in = poses[0,...]
pose_in = torch.squeeze(pose_in)
target_img = torch.squeeze(target_img)
rays_o, rays_d = compute_sample_rays(height, width, focal_len, torch.squeeze(pose_in))
model.zero_grad()  # zero gradient buffer before rendering rays
output_img = render_rays(model, rays_o, rays_d)