import numpy as np
import torch
from sklearn import metrics


def fuse_weights_softmax(w1, w2):
    w = torch.stack([w1, w2], dim=0)
    return torch.softmax(w.sum(dim=0), dim=0)


def normalize_weight(weights, p=1):
    ws = np.array(weights)
    ws = np.power(ws, p)
    ws = ws / ws.max()

    return torch.tensor(ws).cuda()


def eva(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    pre = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, average='macro')

    return acc, f1, pre, rec

