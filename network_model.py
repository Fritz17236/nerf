import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import wget
from warnings import warn
import constants as const

# region data loading, processing, and batching

data_file = os.path.join('data', 'tiny_nerf_data.npz')
if not os.path.exists(data_file):  # get tiny nerf data
    wget.download("http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz", out=data_file)

data = np.load(data_file)
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
print(images.shape, poses.shape, focal)


def positional_encode(data_in: torch.tensor, l_encode: int = const.L_ENCODE):
    """
    Maps M x N input tensor to M x (N + N * 2L) tensor in higher dimension by applying sin/cosine transforms to M data
    :param data_in: Tensor of shape (M, N)
    :param l_encode: integer specifying number of cosine and sine frequencies to compute
    :return out: tensor of shape (M, N + N * 2L), tensor with high-dimensional data appended
    """
    if type(data_in) != torch.Tensor:
        warn("Bad data type passed to positional_encode, expected tensor but was: {0}".format(type(data_in)))
        return

    if len(data_in.shape) != 2:
        warn("positional_encode expects 2-D input, but tensor had shape {0}".format(data_in.shape))
        return
    out = data_in
    for i in range(l_encode):
        for fn in [torch.cos, torch.sin]:
            out = torch.cat((out, fn(torch.sin(2 ** i * torch.pi * data_in))), dim=-1)

    assert (out.shape[0] == data_in.shape[0] and out.shape[1] == data_in.shape[1] + 2 * data_in.shape[
        1] * l_encode), "wrong output shapes: in {0}, out {1}".format(
        data_in.shape, out.shape)
    return out


def nerf_mlp() -> torch.nn.Module:
    """
    Create a multi-layer perceptron (mlp) neural network designed according to nerf specifications
    """
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.flatten = nn.Flatten()
            # create a sequential stack that takes (x,y,z) input in batches, process first 4 hidden layers
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

            self.out_layer = nn.Sequential(*[nn.ReLU(), nn.Linear(256, 128), nn.Linear(128, 4), nn.Sigmoid()])

        def forward(self, x):
            x = self.flatten(x)
            inject = torch.cat([self.stack_pre_inject(x), x])
            out = self.stack_post_inject(inject)
            return self.out_layer(out)

    return Net()



# endregion


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # zeros gradient buffers
        loss.backward()  #
        optimizer.step()

        if batch_num % 100 == 0:
            loss, current = loss.item(), batch_num * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():  # if you know yhou won't comptue gradients, this speeds up computations
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()  # switch network to evaluate model mode

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
