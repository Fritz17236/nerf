import numpy as np
import torch


def print_mem():
    """
    Print the current memory state of the GPU to console
    """
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    f = r - a  # free inside reserved
    print("Memory --  Total: {0},  Reserved: {1},  Allocatted: {2},  Free: {3}".format(t,r,a,f))


def model_size(model, input_size):
    """
    Return an estimate of the total size (in GB) of using a Pytorch neural network. Copied from:
    http://jck.bio/pytorch_estimating_model_size/
    :param model:
    :param input_size:
    :return:
    """
    from pytorch_modelsize import SizeEstimator

    se = SizeEstimator(model, input_size)
    print(se.estimate_size())