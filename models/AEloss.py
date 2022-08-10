import torch
import numpy as np
import torch.nn.functional as F


def L_reconstruction(FAEs, FDs):
    out = F.mse_loss(FAEs, FDs)
    return out


def L_inpaint(FAEg, FDs, r=1):
    out = F.mse_loss(FAEg, FDs)
    out = out * r
    return out


def L_all(FAEs, FAEg, FDs):
    out = L_reconstruction(FAEs, FDs) + L_inpaint(FAEg, FDs)
    return out
