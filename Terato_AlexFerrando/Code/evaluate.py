import copy
import csv
import os
import time
import datahandler

import numpy as np
import torch
import time
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def evaluate_sample(masks_true, masks_pred, masks_names, metrics):
    metrics_results = {}
    masks_pred = torch.split(masks_pred, 1, dim = 1)
    masks_true = torch.split(masks_true, 1, dim = 1)
    for i, mask_name in enumerate(masks_names):
        # Let's transform tensor into numpy arrays to evaluate.
        mask_pred = masks_pred[i].cpu().data.numpy().ravel()
        mask_true = masks_true[i].cpu().data.numpy().ravel()
        if mask_true.any() != 0:
            for name, metric in metrics.items():
                if name in ['f1_score','precision','recall']:
                    # Use a classification threshold of 0.1
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true > 0, mask_pred > 0.01)
                else:
                    metrics_results[f'{name}_{mask_name}'] = metric(mask_true.astype('uint8'), mask_pred)
    return metrics_results
