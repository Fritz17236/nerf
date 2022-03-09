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
from definitions import compute_sample_rays, render_rays, ImagePoseDataset, NerfNet, load_network, load_data

if __name__ == '__main__':
    cuda = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    height, width, focal_len, ip_dataset, _ = load_data()

    # instantiate nerf network, setup training loop
    model = load_network()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # enter training loop
    t_start = time.time()
    print("training start at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    for idx_epoch in range(const.NUM_TRAINING_EPOCHS):
        psnrs = []
        losses = []
        for i, (target_img, pose_in) in enumerate(tqdm.tqdm(torch.utils.data.DataLoader(ip_dataset))):
            pose_in = torch.squeeze(pose_in)
            target_img = torch.squeeze(target_img)
            rays_o, rays_d = compute_sample_rays(height, width, focal_len, torch.squeeze(pose_in))

            # pass BLOCK_SIZE subset of image & rays through render to avoid out-of-memory
            block_size = 50 # assume image is shape  k * block_size  X k * block_size where k is an integer
            num_blocks_h = height // block_size
            num_blocks_w = width // block_size
            output_img = torch.zeros_like(target_img)
            losses_block = []

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
                    losses_block.append(loss.item())

            # compute combined loss, step
            psnrs.append(
                skimage.metrics.peak_signal_noise_ratio(target_img.cpu().detach().numpy(),
                                                        output_img.cpu().detach().numpy())
            )
            losses.append(np.mean(np.asarray(losses_block)))
        seconds = time.time() - t_start
        time_str = "{0:0>2}:{1:0>2}:{2:0<2}".format(int(seconds // (60 * 60)), int((seconds // 60) % 60),  int(seconds % 60))

        print('epoch: {0:0>5}, psnr: {1:>8}, loss: {3:>5},  elapsed time: {2}'.format(idx_epoch, np.round(torch.mean(torch.tensor(psnrs)).cpu().detach().numpy(),6), time_str, np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(const.MODEL_SAVE_DIR, "nerf_net"))

    # exit(0)

# endregion

