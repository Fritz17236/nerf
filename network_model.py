import json
import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import tqdm
from torch import nn
import wget
import constants as const
import skimage
import time
import datetime

# TODO: implement cuda.jit
from definitions import compute_sample_rays, render_rays, ImagePoseDataset, NerfNet

if __name__ == '__main__':
    cuda = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

# region wip: load more sophisticated data
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

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)] # this is a list of indices used to split the dataset

    height, width = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal_len = .5 * width / np.tan(.5 * camera_angle_x)

    images = torch.Tensor(all_imgs[0][..., :3]).type(const.DTYPE)
    # # downsample by factor of k
    factor = const.DOWNSAMPLE_FACTOR
    height //= factor
    width //= factor
    focal_len //= factor
    images_resized = torch.empty([images.shape[0],height, width, images.shape[-1]])
    for idx in range(images.shape[0]):
        images_resized[idx, ...] = torchvision.transforms.functional.resize(images[idx,...].T, size=[height, width]).T

    images = images_resized.type(const.DTYPE)


    poses = torch.Tensor(all_poses[0]).type(const.DTYPE)
    ip_dataset = ImagePoseDataset(images, poses)



    # how is data (imgs, poses, height, width, focal_len passed to the network?)
    # what to do with render_poses  and i_split?
# endregion


    # Load Data
    # data_file = os.path.join('data', 'tiny_nerf_data.npz')
    # if not os.path.exists(data_file):  # get tiny nerf data
    #     wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
    #                   out=data_file)
    #
    # data = np.load(data_file)
    # images = torch.from_numpy(data['images']).type(const.DTYPE)
    # poses = torch.from_numpy(data['poses']).type(const.DTYPE)
    # focal = torch.from_numpy(data['focal']).type(const.DTYPE)  # set to focal_len = torch.from...
    # H, W = images.shape[1:3]  # set to height, width = images.shape[1:3]
    # ip_dataset = ImagePoseDataset(images, poses)
    # near = const.NEAR_FRUSTUM  # remove
    # far = const.FAR_FRUSTUM  # remove
    #
    # height = H  # remove
    # width = W  # remove
    # focal_len = focal  # remove
    # num_samples = const.NUM_RAY_SAMPLES  # remove
    # N_samples = num_samples  # remove
    # L_embed = const.L_ENCODE  # remove

    # setup neural net and training harness
    model_file = os.path.join(const.MODEL_SAVE_DIR, 'nerf_net')
    model = NerfNet().type(const.DTYPE)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    plt.ion()

    # enter training loop
    t_start = time.time()

    print("training start at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    for idx_epoch in range(const.NUM_TRAINING_EPOCHS):
        psnrs = []
        losses = []
        for i, (target_img, pose_in) in enumerate(tqdm.tqdm(torch.utils.data.DataLoader(ip_dataset))):
            # TODO: query coarse network for importance sampling, compute loss, step
            # TODO: use coarse output to determine samples for fine network, query network, compute loss, step
            pose_in = torch.squeeze(pose_in)
            target_img = torch.squeeze(target_img)
            rays_o, rays_d = compute_sample_rays(height, width, focal_len, torch.squeeze(pose_in))
            # TODO: break target img and sample rays into corresponding, subblocks, pass thru net and run GD on subblocks

            # pass BLOCK_SIZE subset of image & rays through render to avoid out-of-memory
            block_size = 100 # assume image is shape  k * block_size  X k * block_size where k is an integer
            num_blocks_h = height // block_size
            num_blocks_w = width // block_size
            output_img = torch.zeros_like(target_img)


            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    model.zero_grad()

                    target_img_block = target_img[i * block_size: (i+1)*block_size, j*block_size:(j+1)*block_size,:]
                    rays_o_block = rays_o[i * block_size: (i+1)*block_size, j*block_size:(j+1)*block_size,:]
                    rays_d_block = rays_d[i * block_size: (i+1)*block_size, j*block_size:(j+1)*block_size,:]
                    output_img_block = render_rays(model, rays_o_block, rays_d_block)

                    # store subset for computing PSNR
                    output_img[i * block_size: (i+1)*block_size, j*block_size:(j+1)*block_size,:] = output_img_block

                    loss = criterion(output_img_block, target_img_block)
                    loss.backward()
                    optimizer.step()

            # compute combined loss, step


            psnrs.append(
                skimage.metrics.peak_signal_noise_ratio(target_img.cpu().detach().numpy(),
                                                        output_img.cpu().detach().numpy())
            )
            losses.append(loss.item())
        seconds = time.time() - t_start
        time_str = "{0:0>2}:{1:0>2}:{2:0<2}".format(int(seconds // (60 * 60)), int((seconds // 60) % 60),  int(seconds % 60))

        print('epoch: {0:0>5}, psnr: {1:>8}, loss: {3:>5},  elapsed time: {2}'.format(idx_epoch, np.round(torch.mean(torch.tensor(psnrs)).cpu().detach().numpy(),6), time_str, np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(const.MODEL_SAVE_DIR, "nerf_net"))

    # exit(0)1

# endregion

