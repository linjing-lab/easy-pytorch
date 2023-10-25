# Copyright (c) 2023 linjing-lab

import sys

if sys.version_info < (3, 7, 0):
    raise OSError(f'perming requires Python >=3.7, but yours is {sys.version}')

try:
    import torch
    TORCH_VERSION = torch.__version__
    if TORCH_VERSION.find('cu') != -1:
        CUDA_VERSION, DEVICE_COUNT = torch.version.cuda, torch.cuda.device_count()
    else:
        raise OSError(f'Your PyTorch version must support cuda acceleration. Please refer to https://pytorch.org/get-started/locally/ for PyTorch compatible with your windows computer.')
except ModuleNotFoundError:
    pass

from .general import Box
from .common import Regressier, Binarier, Mutipler, Ranker

GENERAL_BOX = Box

COMMON_MODELS = {
    'Regression': Regressier,
    'Binary-classification': Binarier,
    'Multi-classification': Mutipler,
    'Multi-outputs': Ranker
}

__version__ = '1.7.0'