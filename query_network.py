import numpy as np
import matplotlib.pyplot as plt
import skimage
import torch
import tqdm
import constants as const
import imageio
from definitions import *

if __name__ == '__main__':
    with torch.inference_mode():
        # Load Data
        run_widget()
        exit(0)

        height, width, focal_len, ip_dataset, ip_dataset_test = load_data()
        model = load_network()

        make_widget_frames(model, height, width, focal_len)




        exit(0)
        make_video(model, height, width, focal_len)

        # render in image sub-blocks
        dists = []
        psnrs = []
        block_size = 100  # assume image is shape  k * block_size  X k * block_size where k is an integer
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size
        for img_test, pose_test in tqdm.tqdm(ip_dataset_test):

            # get the physically (transformation) closest training sample to this c2w.
            rays_o, rays_d = compute_sample_rays(800, 800, focal_len, pose_test[:3,:4])
            output_img = torch.zeros((800, 800, 3))

            for i in range(num_blocks_h):
                for j in range(num_blocks_w):

                    rays_o_block = rays_o[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                    rays_d_block = rays_d[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size, :]
                    output_img_block = render_rays(model, rays_o_block, rays_d_block)

                    # store subset for computing PSNR
                    output_img[i * block_size: (i + 1) * block_size, j * block_size:(j + 1) * block_size,
                    :] = output_img_block

            # get closest training image by euclidean distance of pose vector
            img_min, closest_dist = closest_loc_sample(pose_test, ip_dataset)
            dists.append(closest_dist)
            psnrs.append(
                skimage.metrics.peak_signal_noise_ratio(img_test.cpu().detach().numpy(),
                                                        output_img.cpu().detach().numpy())
            )

    psnrs = np.asarray(psnrs)
    dists = np.asarray(dists)
    mask = psnrs > 20  # filter outlier images less than 20 psnr (network average is 27)
    plt.scatter(dists[mask], psnrs[mask])
    rho = np.corrcoef(dists[mask], psnrs[mask])[0,1]
    plt.title(r"Network Performance vs. Distance to Nearest Train Image $\rho={0}$".format(np.round(rho, 3)))
    plt.ylabel("PSNR(test image, ground truth)")
    plt.xlabel("||test image  params - nearest train image params||")
    plt.show()
