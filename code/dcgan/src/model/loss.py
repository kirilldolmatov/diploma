import torch.nn as nn


def bce_loss(output, target):
    criterion = nn.BCELoss()
    return criterion(output, target)
    