import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import wget
from warnings import warn
import constants as const
import skimage
import time
from numba import jit


# TODO: implement cuda.jit


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


class NerfNet(nn.Module):
    def __init__(self, embed=positional_encode):
        super(NerfNet, self).__init__()

        self.embed = embed

        self.flatten = nn.Flatten()
        # create a sequential stack that takes (network_input,y,z) input in batches, process first 4 hidden layers
        layer_list = [nn.Linear(3 + 3 * 2 * const.L_ENCODE, 256), nn.ReLU()]
        for i in range(3):
            layer_list += [nn.Linear(256, 256), nn.ReLU()]
        self.stack_pre_inject = nn.Sequential(*layer_list)

        # inject input again and process 4 remaining layers
        layer_list = [nn.Linear(3 + 3 * 2 * const.L_ENCODE + 256, 256), nn.ReLU()]
        for i in range(3):
            layer_list += [nn.Linear(256, 256), nn.ReLU()]
        self.stack_post_inject = nn.Sequential(*layer_list)

        # TODO: implement directional (not just density) encoding in network

        self.out_layer = nn.Sequential(*[nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, 4)])

    def forward(self, network_input):
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
    return rays_d, rays_o


def render_rays(model: NerfNet(), rays_o: torch.Tensor, rays_d: torch.Tensor,
                near: float = const.NEAR_FRUSTUM, far: float = const.FAR_FRUSTUM,
                num_samples: int = const.NUM_RAY_SAMPLES, stratified_sampling: bool = const.STRATIFIED_SAMPLING):
    """
    Render a given set of rays by querying a given neural network.
    :param model: neural network to query. expects input to be passed through positional_encode
    :param rays_o: (height network_input width network_input 3) tensor containing origins of query rays
    :param rays_d: (height network_input width network_input 3) tensor containing directions of query rays
    :param near: float specifying the near clipping plane of the view frustum
    :param far: float specifying the far clipping plane of the view frustum
    :param num_samples: number of points along each cast ray to query the network
    :param stratified_sampling: uniformly sample query points along each ray
    :return: rgbs, depths,
    """

    # compute 3D query points
    ts = torch.linspace(start=near, end=far, steps=num_samples).type(const.DTYPE)
    if stratified_sampling:
        add = torch.rand(list(rays_o.size()[:-1]) + [num_samples]) * (far - near) / (num_samples)
        ts = ts[np.newaxis, np.newaxis, :] + add
    query_points = rays_o[..., None, :] + rays_d[..., None, :] * ts[..., :, None]

    # Extract value of network at query points
    query_points_flattened = torch.reshape(query_points, [-1, 3])
    rq_dataset = RayQueryDataset(query_points_flattened)  # create dataset object for easier manipulation
    # data loader to abstract batching and allow multiprocessing and gpu acceleration
    # rq_dataloader = torch.utils.data.DataLoader(rq_dataset, batch_size=1024*128, shuffle=True, pin_memory=const.USE_CUDA,
    #                            num_workers=const.NUM_CPU_CORES)
    # batched_output = torch.Tensor()
    # for ray_batch in rq_dataloader:
    #     batched_output = torch.cat((batched_output, model(ray_batch)), dim=0)
    batched_output = model(query_points_flattened)
    batched_output = torch.reshape(batched_output, list(query_points.shape[:-1]) + [4])

    # compute output density and rgb colors
    sigma = torch.nn.functional.relu(batched_output[..., 3])
    rgbs = torch.sigmoid(batched_output[..., :3])

    # perform volumetric rendering
    dists = torch.cat([ts[..., 1:] - ts[..., :-1], torch.broadcast_to(torch.tensor(1e10).type(const.DTYPE),
                                                                      ts[..., :1].shape)], -1)
    alpha = 1 - torch.exp(-sigma * dists)
    cumprod_arg = 1.0 - alpha + 1e-10
    cumprod_arg = torch.cat([torch.ones_like(cumprod_arg[..., 0:1]), cumprod_arg], dim=-1)
    weights = alpha * torch.cumprod(cumprod_arg, dim=-1)[..., :-1]
    return torch.sum(weights[..., None] * rgbs, -2)


if __name__ == '__main__':
    cuda = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
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
        time_str = "{0}:{1}:{2}".format(int(seconds // (60 * 60)), int((seconds // 60) % 60),  int(seconds % 60))

        print('epoch: {0}, psnr: {1}, time: {2}'.format(idx_epoch, torch.mean(torch.tensor(psnrs)), time_str))
        torch.save(model.state_dict(), os.path.join(const.MODEL_SAVE_DIR, "nerf_net"))

    # exit(0)

# endregion

# region old code
# # Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )
#
# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )
#
# batch_size = 64
#
# # Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
#
# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break
#
# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
#
#
# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#
#
# model = NeuralNetwork().to(device)
# print(model)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#
#
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch_num, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()  # zeros gradient buffers
#         loss.backward()  #
#         optimizer.step()
#
#         if batch_num % 100 == 0:
#             loss, current = loss.item(), batch_num * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#
#
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():  # if you know yhou won't comptue gradients, this speeds up computations
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#
#
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")
#
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")
#
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))
#
# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]
#
# model.eval()  # switch network to evaluate model mode
#
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
#
# endregion
