# Import dependencies

import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.neighbors import KNeighborsClassifier
import numpy as np



# This is the function that we used to train the KNN

def train_knn(features_dict):
    pill_names = list(features_dict.keys())
    X = []
    y = []

    for pill_name in pill_names:
        embed_tensor = features_dict[pill_name]
        flat_vec = embed_tensor.flatten().cpu().numpy()
        X.append(flat_vec)
        y.append(pill_name)

    X = np.array(X)
    y = np.array(y)


    knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="cosine"
)

    knn.fit(X, y)
    return X,y,knn

