import sys
import os
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# import nets as nets
from .nets import *

from .utils.convenient import mkdir
from .utils import common
from .utils.common import tonumpy, matmul, pool
from .utils.pytorch_loader import get_loader

import pickle as pkl
import hashlib

def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net

def GeMAP(path, gpu=0):
    iscuda = common.torch_set_gpu(gpu)
    checkpoint = common.load_checkpoint(path, iscuda)
    net = create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net