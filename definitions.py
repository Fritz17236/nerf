import os
from _warnings import warn

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import tqdm
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from torch import nn
import wget
from warnings import warn
import constants as const
import skimage
import time
import datetime
import json
import imageio
import trimesh
import pyrender
import mcubes

# classes depend on this function so it must be declared up here
def positional_encode(data_in: torch.tensor, l_encode: int = const.L_ENCODE) -> torch.Tensor:
    """
    Maps M network_input N input tensor to M network_input (N + N * 2L) tensor in higher dimension by applying sin/cosine transforms to M data
    :param data_in: Tensor of shape (M, N)
    :param l_encode: integer specifying number of cosine and sine frequencies to compute
    :return out: tensor of shape (M, N + N * 2L), tensor with high-dimensional data appended
    """
    if type(data_in) != torch.Tensor:
        warn("Bad data type passed to positional_encode, expected tensor but was: {0}".format(type(data_in)))
        return torch.Tensor()

    if len(data_in.shape) != 2:
        warn("positional_encode expects 2-D input, but tensor had shape {0}".format(data_in.shape))
        return torch.Tensor()

    out = data_in
    for i in range(l_encode):
        for fn in [torch.cos, torch.sin]:
            out = torch.cat((out, fn(torch.sin(2 ** i * torch.pi * data_in))), dim=-1)

    assert (out.shape[0] == data_in.shape[0] and out.shape[1] == data_in.shape[1] + 2 * data_in.shape[
        1] * l_encode), "wrong output shapes: in {0}, out {1}".format(
        data_in.shape, out.shape)
    return out


class RayQueryDataset(torch.utils.data.Dataset):
    """
    Custom Torch Dataset container for storing ray queries to be passed to the network
    """

    def __init__(self, ray_queries: torch.Tensor):
        self.ray_queries = ray_queries

    def __len__(self):
        return self.ray_queries.shape[0]

    def __getitem__(self, index):
        return self.ray_queries[index, :]


class ImagePoseDataset(torch.utils.data.Dataset):
    """
    Dataset container for pose, image pairs representing input and output model data
    """

    def __init__(self, imgs: torch.Tensor, view_poses: torch.Tensor):
        self.imgs = imgs
        self.poses = view_poses

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index, ...], self.poses[index, ...]


class NerfNet(nn.Module):
    def __init__(self, embed=positional_encode):
        super(NerfNet, self).__init__()

        self.embed = embed

        self.flatten = nn.Flatten()
        # create a sequential stack that takes (network_input,y,z) input in batches, process first 4 hidden layers

        layer_list = [nn.Linear(3 + 3 * 2 * const.L_ENCODE, 256), nn.ReLU()]

        for i in range(3):
            layer_list += [nn.Linear(256, 256, bias=False), nn.ReLU()]
        self.stack_pre_inject = nn.Sequential(*layer_list)

        # inject input again and process 4 remaining layers
        layer_list = [nn.Linear(3 + 3 * 2 * const.L_ENCODE + 256, 256), nn.ReLU()]
        for i in range(3):
            layer_list += [nn.Linear(256, 256, bias=False), nn.ReLU()]
        self.stack_post_inject = nn.Sequential(*layer_list)

        # TODO: implement directional (not just density) encoding in network
        if const.DIRECTIONAL_ENCODING:
            self.out_layer = nn.Sequential(*[nn.ReLU(), nn.Linear(255 + (3 + 3 * 2 * const.L_ENCODE), 128, bias=False), nn.Linear(128, 3)])
        else:
            self.out_layer = nn.Sequential(*[nn.ReLU(), nn.Linear(256, 128, bias=False), nn.Linear(128, 4)])

    def forward(self, network_input):
        # if not network_input.requires_grad:
        #     network_input.requires_grad = True
        if const.DIRECTIONAL_ENCODING:
            loc_input = network_input[:, :3]
            dir_input = network_input[:, 3:]

            loc_input = self.embed(loc_input)
            loc_input = self.flatten(loc_input)
            dir_input = self.embed(dir_input)
            dir_input = self.flatten(dir_input)
            #out = torch.utils.checkpoint.checkpoint_sequential(self.stack_pre_inject, 2, loc_input)
            out = self.stack_pre_inject(loc_input)

            reinjected_input = torch.cat([out, loc_input], dim=-1)
            out = self.stack_post_inject(reinjected_input)
            #   out = torch.utils.checkpoint.checkpoint_sequential(self.stack_post_inject, 2, reinjected_input)

            sigma = out[:,0:1]
            out = torch.cat([out[:,1:], dir_input], dim=-1)
            rgb = self.out_layer(out)
            # rgb = torch.utils.checkpoint.checkpoint_sequential(self.out_layer, 2, out_cat_dir)
            del out
            return torch.cat([rgb, sigma], dim=-1)

        else:
            network_input = self.embed(network_input)
            network_input = self.flatten(network_input)
            out = self.stack_pre_inject(network_input)
            inject = torch.cat([out, network_input], dim=1)
            out = self.stack_post_inject(inject)
        return self.out_layer(out)


def compute_sample_rays(height: int, width: int, focal_len: float, c2w: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Compute the sample rays for a camera with a given height, width, focal length,
    and direction/postion  matrix.
    :param height: height of the image the camera produces (in pixels)
    :param width: width of the image the camera procudes (in pixels)
    :param focal_len:  focal length of the camera, distance from the camera view to the image plane
    :param c2w: 4x4 matrix specifying the location & direction of the camera
    :return: rays_d, rays_o, list of ray origins and directions passing from the camera view through the image
    plane a distance focal length from the camera view.
    """
    # generate grid of pixel indices
    i, j = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')

    if (const.USE_CUDA):
        i = i.cuda()
        j = j.cuda()
    # perspective projection from camera location onto image plane
    dirs = torch.stack([(i - width * .5) / focal_len, -(j - height * .5) / focal_len, -torch.ones_like(i)], dim=-1)

    # create new axis to broadcast to, multiply by c2w to fill broadcast, sum result to get ray directions
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], dim=-1)

    # create copies of ray origin to match the size / number of ray directions
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.size())
    return rays_o, rays_d


def batchify(to_apply, chunk_size=const.RAY_CHUNK_SIZE):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk_size is None:
        return to_apply

    def ret(inputs):
        return torch.concat([to_apply(inputs[i:i+chunk_size:]) for i in range(0, inputs.shape[0], chunk_size)], 0)

    return ret


def render_rays(model: NerfNet(), rays_o: torch.Tensor, rays_d: torch.Tensor,
                near: float = const.NEAR_FRUSTUM, far: float = const.FAR_FRUSTUM,
                num_samples: int = const.NUM_RAY_SAMPLES):
    """
    Render a given set of rays by querying a given neural network.
    :param model: neural network to query. expects input to be passed through positional_encode
    :param rays_o: (height network_input width network_input 3) tensor containing origins of query rays
    :param rays_d: (height network_input width network_input 3) tensor containing directions of query rays
    :param near: float specifying the near clipping plane of the view frustum
    :param far: float specifying the far clipping plane of the view frustum
    :param num_samples: number of points along each cast ray to query the network
    :param stratified_sampling: uniformly sample query points along each ray
    :return: rgbs
    """

    # compute 3D query points
    ts = torch.linspace(start=near, end=far, steps=num_samples).type(const.DTYPE)
    if const.STRATIFIED_SAMPLING:
        add = torch.rand(list(rays_o.size()[:-1]) + [num_samples]) * (far - near) / (num_samples)
        ts = ts[np.newaxis, np.newaxis, :] + add.type(const.DTYPE)
    query_points = rays_o[..., None, :] + rays_d[..., None, :] * ts[..., :, None]

    if const.DIRECTIONAL_ENCODING:
        directions = torch.broadcast_to(rays_d[:, :, np.newaxis, :], query_points.size())
        directions = torch.nn.functional.normalize(directions, dim=-1)
        query_points = torch.cat([query_points, directions], dim=-1)
        query_points_flattened = torch.reshape(query_points, [-1, 6])
    else:

        query_points_flattened = torch.reshape(query_points, [-1, 3])

    # Extract value of network at query points
    output = model(query_points_flattened).reshape(list(query_points.shape[:3]) + [4])

    # compute output density and rgb colors
    sigma = torch.nn.functional.relu(output[..., 3])
    rgbs = torch.sigmoid(output[..., :3])

    # perform volumetric rendering
    dists = torch.cat([ts[..., 1:] - ts[..., :-1], torch.broadcast_to(torch.tensor(1e10).type(const.DTYPE),
                                                                      ts[..., :1].shape)], -1)
    alpha = 1 - torch.exp(-sigma * dists)
    cumprod_arg = 1.0 - alpha + 1e-10
    cumprod_arg = torch.cat([torch.ones_like(cumprod_arg[..., 0:1]), cumprod_arg], dim=-1)
    weights = alpha * torch.cumprod(cumprod_arg, dim=-1)[..., :-1]
    return torch.sum(weights[..., None] * rgbs, -2)


def load_data(data_set='hotdog'):
    if data_set == 'hotdog':
        global height, width, focal_len
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(const.DATA_DIR, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []

            for frame in meta['frames'][::1]:
                fname = os.path.join(const.DATA_DIR, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        i_split = [np.arange(counts[i], counts[i + 1]) for i in
                   range(3)]  # this is a list of indices used to split the dataset
        height, width = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal_len = .5 * width / np.tan(.5 * camera_angle_x)
        images = torch.Tensor(all_imgs[0][..., :3]).type(const.DTYPE)
        images_test = torch.Tensor(all_imgs[2][..., :3]).type(const.DTYPE)

        # # downsample by factor of k
        factor = const.DOWNSAMPLE_FACTOR
        height //= factor
        width //= factor
        focal_len //= factor

        images_resized = torch.empty([images.shape[0], height, width, images.shape[-1]])
        images_resized_test = torch.empty([images_test.shape[0], height, width, images_test.shape[-1]])
        for idx in range(images.shape[0]):
            images_resized[idx, ...] = torchvision.transforms.functional.resize(images[idx, ...].T,
                                                                                size=[height, width]).T
            images_resized_test[idx, ...] = torchvision.transforms.functional.resize(images_test[idx, ...].T,
                                                                                size=[height, width]).T

        images = images_resized.type(const.DTYPE)
        images_test = images_resized_test.type(const.DTYPE)
        poses = torch.Tensor(all_poses[0]).type(const.DTYPE)
        poses_test = torch.Tensor(all_poses[2]).type(const.DTYPE)

        ip_dataset = ImagePoseDataset(images, poses)
        ip_dataset_test = ImagePoseDataset(images_test, poses_test)

        return height, width, focal_len, ip_dataset, ip_dataset_test
    elif data_set =='tiny_nerf':
        data_file = os.path.join('data', 'tiny_nerf_data.npz')
        if not os.path.exists(data_file):  # get tiny nerf data
            wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
                          out=data_file)

        data = np.load(data_file)
        images = torch.from_numpy(data['images']).type(const.DTYPE)
        poses = torch.from_numpy(data['poses']).type(const.DTYPE)
        focal_len = torch.from_numpy(data['focal']).type(const.DTYPE)  # set to foal_len = torch.from...
        height, width = images.shape[1:3]  # set to height, width = images.shape[1:3]

        ip_dataset = ImagePoseDataset(images, poses)
        return height, width, focal_len, ip_dataset




def load_network():
    model = NerfNet().type(const.DTYPE)
    model_file = os.path.join(const.MODEL_SAVE_DIR, 'nerf_net')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    return model


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


def closest_loc_sample(target_c2w, image_pose_dataset):
    """
    Given a target camera to world (c2w) matrix, find the image / pose within a dataset whose pose location is
    physically closest (euclidean distance of transformation vector)

    :param target_c2w:
    :param image_pose_dataset:
    :return:
    """

    targ_transformation = (target_c2w[:3,-1] + 4) / 8
    ref_transformations = [(pose[:3, -1] + 4) / 8 for _, pose in image_pose_dataset] #normalized to 0,1
    targ_r32 = target_c2w[2,1]
    targ_r33 = target_c2w[2,2]
    targ_r31 = target_c2w[2,0]
    targ_r21 = target_c2w[1,0]
    targ_r11 = target_c2w[0,0]

    targ_angles = (torch.Tensor([
        torch.atan2(targ_r32, targ_r33),
        torch.atan2(-targ_r31, torch.sqrt(targ_r32**2 + targ_r33**2)),
        torch.atan2(targ_r21, targ_r11)
    ]) + torch.pi) / (2 * torch.pi) # normalize to 0,1
    targ = torch.cat([targ_transformation, targ_angles.type(const.DTYPE)])
    targ_y = targ_angles[1]
    dists = []
    for j in range(len(ref_transformations)):
        ref_c2w = image_pose_dataset[j][1]
        ref_r32 = ref_c2w[2, 1]
        ref_r33 = ref_c2w[2, 2]
        ref_r31 = ref_c2w[2, 0]
        ref_r21 = ref_c2w[1, 0]
        ref_r11 = ref_c2w[0, 0]
        ref_angles = (torch.Tensor([
            torch.atan2(ref_r32, ref_r33),
            torch.atan2(-ref_r31, torch.sqrt(ref_r32**2 + ref_r33**2)),
            torch.atan2(ref_r21, ref_r11)
        ]) + torch.pi) / (2 * torch.pi) # normalize to 0,1
        ref = torch.cat([ref_transformations[j], ref_angles.type(const.DTYPE)])
        ref_y = ref_angles[1]
        dists.append(torch.linalg.norm(targ_transformation - ref_transformations[j]))

    dists = torch.Tensor(dists)
    idx_min = torch.argmin(dists)
    img_min = image_pose_dataset[idx_min][0]
    return img_min, dists[idx_min]


def make_video(model, height, width, focal_len):
    # render in image sub-blocks
    frames = []
    thetas = np.linspace(0., 360., 120, endpoint=False)
    block_size = 100  # assume image is shape  k * block_size  X k * block_size where k is an integer
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size
    for th in tqdm.tqdm(thetas):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = compute_sample_rays(800, 800, focal_len, c2w[:3, :4])
        output_img = torch.zeros((800, 800, 3))

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                rays_o_block = rays_o[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                rays_d_block = rays_d[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                output_img_block = render_rays(model, rays_o_block, rays_d_block)

                output_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                :] = output_img_block

        frames.append((255 * np.clip(output_img.cpu().detach().numpy(), 0, 1)).astype(np.uint8))

    f = const.VIDEO_OUTPUT_FILE
    imageio.mimwrite(f, frames, fps=30, quality=7)


def make_widget_frames(model, height, width, focal_len):
    thetas = np.linspace(0., 360., 120, endpoint=False)
    phis = np.linspace(0, -45, 2, endpoint=False)
    frames = np.zeros((height, width, 3, len(thetas), len(phis)))

    block_size = 100  # assume image is shape  k * block_size  X k * block_size where k is an integer
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    for idx_theta, th in enumerate(tqdm.tqdm(thetas)):
        for idx_phi, ph in enumerate(tqdm.tqdm(phis)):
            c2w = pose_spherical(th, ph, 4.)
            rays_o, rays_d = compute_sample_rays(800, 800, focal_len, c2w[:3, :4])
            output_img = torch.zeros((800, 800, 3))

            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    rays_o_block = rays_o[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                    rays_d_block = rays_d[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                    output_img_block = render_rays(model, rays_o_block, rays_d_block)

                    output_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                    :] = output_img_block

            frames[..., idx_theta, idx_phi] = output_img
    params = np.stack([thetas, phis], axis=-1)
    np.save('widget_frames', frames)
    np.save('widget_params', params)


def run_widget():
    plt.figure(0)
    plt.clf()
    frames = np.load('widget_frames.npy')
    idx_th = 0
    idx_ph = 1
    img_handle = plt.imshow(frames[..., idx_th, idx_ph])
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.axis('off')
    ax_theta = plt.axes([0.25, 0.1, 0.65, 0.03])

    theta_slider = Slider(
        ax=ax_theta,
        label=r"$\theta$",
        valmin=0,
        valmax=frames.shape[-2],
        valinit=idx_th,
        valstep=np.arange(frames.shape[-2]),
    )

    def update(val):
        img_handle.set_data(frames[..., int(theta_slider.val), idx_ph])
        plt.gcf().canvas.draw_idle()

    theta_slider.on_changed(update)
    plt.show()

def extract_mesh(model, height, width, focal_len):
    """
    Extract a mesh respresentation of the visual scene
    :param model:
    :param ehight:
    :param width:
    :param focal_len:
    :return:
    """
    ts = np.linspace(-1.2, 1.2, const.MESH_SAMPLE_RES)
    ts_z = np.linspace(-.2, 1.2, const.MESH_SAMPLE_RES)
    query_pts = np.stack(np.meshgrid(ts, ts, ts_z), -1).astype(np.float32)
    flat = torch.Tensor(query_pts.reshape([-1,3])).cpu()

    thetas = np.linspace(0., 360., flat.shape[0], endpoint=False)
    c2ws = [pose_spherical(th, -30., 4.) for th in thetas]
    view_dirs = []
    for idx, c2w in enumerate(c2ws):
        _ ,rays_d = compute_sample_rays(800, 800, focal_len, c2w[:3, :4])
        view_dirs.append(rays_d[height//2, width//2, :])

    flat = torch.cat([flat, view_dirs], dim=-1).cpu()
    out = torch.empty(size=(flat.size()[0], 4)).cpu()

    chunk_size = 1024 * 32
    with torch.inference_mode():
        for i in np.arange(0, out.shape[0], chunk_size):
            out[i : (i + chunk_size), :] = model(flat[i : (i + chunk_size), :].type(const.DTYPE)).cpu()

    out = torch.reshape(out, list(query_pts.shape[:-1]) + [-1]).cpu()
    out = torch.maximum(out, torch.Tensor([0.]))
    sigmas = torch.mean(out, dim=-1).cpu().numpy()

    threshold = 25.
    print('fraction occupied', np.mean(sigmas > threshold))
    vertices, triangles = mcubes.marching_cubes(sigmas, threshold)
    print('done', vertices.shape, triangles.shape)

    mesh = trimesh.Trimesh(vertices / const.MESH_SAMPLE_RES - .5, triangles)
    mesh.show()