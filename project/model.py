import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.covariance import LedoitWolf
import json
from collections.abc import Mapping
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
torch.manual_seed(0)

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Intatiate the gated layer
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # nn.Linear(1280, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            nn.Linear(1280, 16)
        )

    def forward(self, x):
        x = self.model(x)
        return x