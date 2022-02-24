import os
import numpy as np
import matplotlib.pyplot as plt
import torch
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
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.join(const.DATA_DIR, 'transforms_{}.json'.format(s)), 'r') as fp:
#             metas[s] = json.load(fp)
#
#     all_imgs = []
#     all_poses = []
#     counts = [0]
#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#
#
#         for frame in meta['frames'][::1]:
#             fname = os.path.join(const.DATA_DIR, frame['file_path'] + '.png')
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#         imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)
#
#     i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)] # this is a list of indices used to split the dataset
#
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
#
#     height, width = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * width / np.tan(.5 * camera_angle_x)
#
#     # downsample by factor of 8
#     imgs_resized = []
#     for img in imgs:
#         resized_img = torchvision.transforms.functional.resize(torch.tensor(img).T, size=[100, 100]).T
#         imgs_resized.append(resized_img)
#     height = 100
#     width = 100
#     focal = focal / 8.

    # how is data (imgs, poses, height, width, focal_len passed to the network?)
    # what to do with render_poses  and i_split?
# endregion

    # Load Data
    data_file = os.path.join('data', 'tiny_nerf_data.npz')
    if not os.path.exists(data_file):  # get tiny nerf data
        wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
                      out=data_file)

    data = np.load(data_file)
    images = torch.from_numpy(data['images']).type(const.DTYPE)
    poses = torch.from_numpy(data['poses']).type(const.DTYPE)
    focal = torch.from_numpy(data['focal']).type(const.DTYPE)  # set to focal_len = torch.from...
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
        for i, (target_img, pose_in) in enumerate(torch.utils.data.DataLoader(ip_dataset)):
            pose_in = torch.squeeze(pose_in)
            target_img = torch.squeeze(target_img)
            rays_o, rays_d = compute_sample_rays(height, width, focal_len, torch.squeeze(pose_in))
            model.zero_grad()  # zero gradient buffer before rendering rays
            output_img = render_rays(model, rays_o, rays_d)
            loss = criterion(output_img, target_img)
            loss.backward()
            optimizer.step()
            psnrs.append(
                skimage.metrics.peak_signal_noise_ratio(target_img.cpu().detach().numpy(),
                                                        output_img.cpu().detach().numpy())
            )
        seconds = time.time() - t_start
        time_str = "{0:0>2}:{1:0>2}:{2:0<2}".format(int(seconds // (60 * 60)), int((seconds // 60) % 60),  int(seconds % 60))

        print('epoch: {0:0>5}, psnr: {1:>8}, loss: {3:>5},  elapsed time: {2}'.format(idx_epoch, np.round(torch.mean(torch.tensor(psnrs)).cpu().detach().numpy(),6), time_str, loss.item()))
        torch.save(model.state_dict(), os.path.join(const.MODEL_SAVE_DIR, "nerf_net"))

    # exit(0)

# endregion

