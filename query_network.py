import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import wget
import constants as const
import json
import imageio

from definitions import compute_sample_rays, render_rays, ImagePoseDataset, NerfNet
from matplotlib.widgets import Slider
import skimage
import time


def pose_spherical(theta, phi, radius):
    """
    Given view direction specified by spherical coordinates (theta, phi, radius) return a 4x4 camera-to-world matrix
    :param theta:
    :param phi:
    :param radius:
    :return:
    """
    theta = torch.Tensor([theta])
    phi = torch.tensor([phi])
    radius = torch.tensor([radius])

    # transforms a point along the z axis by given t
    transform_z = lambda t: torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]).type(const.DTYPE)

    # rotates a point away from the y-z plane by given angle phi
    rotate_phi = lambda phi: torch.tensor([
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1]
    ]).type(const.DTYPE)

    # rotates a point away from the x-z plane
    rotate_theta = lambda theta: torch.tensor([
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1]
    ]).type(const.DTYPE)

    # compose the transformations above to compute the camera_to_world matrix (pose)
    camera_to_world = transform_z(radius)
    camera_to_world = rotate_phi(phi/180. * np.pi) @ camera_to_world
    camera_to_world = rotate_theta(theta/180. * np.pi) @ camera_to_world
    camera_to_world = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).type(
        const.DTYPE) @ camera_to_world  # ?

    return camera_to_world


def render_view(**kwargs):
    camera_to_world = pose_spherical(**kwargs)
    rays_o, rays_d = compute_sample_rays(height, width, focal_len, camera_to_world)
    output_img = render_rays(model, rays_o, rays_d)
    img = torch.clip(output_img, min=0, max=1).cpu().detach().numpy()

    plt.figure('network render')
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':

    # Load Data
    data_file = os.path.join('data', 'tiny_nerf_data.npz')
    if not os.path.exists(data_file):  # get tiny nerf data
        wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
                      out=data_file)

    data = np.load(data_file)
    images = torch.from_numpy(data['images']).type(const.DTYPE)
    poses = torch.from_numpy(data['poses']).type(const.DTYPE)
    focal_len = torch.from_numpy(data['focal']).type(const.DTYPE)  # set to foal_len = torch.from...
    height, width = images.shape[1:3]  # set to height, width = images.shape[1:3]


    # theta, phi, radius provide these to render to perform rendering
    model = NerfNet().type(const.DTYPE)
    model_file = os.path.join(const.MODEL_SAVE_DIR, 'nerf_net')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))


    frames = []
    for th in tqdm.tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = compute_sample_rays(height, width, focal_len, c2w[:3,:4])
        rgb = render_rays(model, rays_o, rays_d,).cpu().detach().numpy()
        frames.append((255*np.clip(rgb, 0 ,1)).astype(np.uint8))

    import imageio
    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)

