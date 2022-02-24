import torch

# Configuration Parameters for NERF Network


L_ENCODE = 6    # Positional Encoding Projection Size
NEAR_FRUSTUM = 2.0  # how far to travel along a ray  before hitting the near clipping point of the view frustum
FAR_FRUSTUM = 6.0  # how far to travel along a ray  before hitting the far clipping point of the view frustum
NUM_RAY_SAMPLES = 64             # Number of samples to collect along each ray cast by querying the  network
STRATIFIED_SAMPLING = True       # Use stratified sampling (queries network at uniformly sampled points)
USE_CUDA = True                  # True if using CUDA GPU Acceleration
NUM_CPU_CORES = 6                # Number of CPU cores to use for multiprocessing
NUM_TRAINING_EPOCHS = 1024       # Number of iterations to train the neural network
MODEL_SAVE_DIR = './'            # Location to save neural networks to

DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor  # set const datatype if using CUDA

