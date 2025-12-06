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
import numpy as np

# This is an evaluation function that calculates loss 

def evaluate(model, loader, device, criterion):
   model.eval()
   total_loss = 0
   correct = 0
   total = 0


   with torch.no_grad():
       for imgs, labels in loader:
           imgs, labels = imgs.to(device), labels.to(device)

           outputs = model(imgs)
           loss = criterion(outputs, labels)

           total_loss += loss.item() * imgs.size(0)
           _, preds = outputs.max(1)
           correct += preds.eq(labels).sum().item()
           total += labels.size(0)


   return total_loss / total, correct / total

# This function gets the recall@1 and recall@5 to evaluate the softmax classifier (pretrained ResNet18)

def evaluate_softmax(model, test_loader, device):
    model.eval()
    total = 0
    recall1 = 0
    recall5 = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            top5 = torch.topk(logits, k=5, dim=1).indices

            recall1 += (top5[:, 0] == labels).sum().item()

            for i in range(len(labels)):
                if labels[i].item() in top5[i]:
                    recall5 += 1

            total += len(labels)

    return recall1 / total, recall5 / total

# This function evaluates KNN and tells us how many of the labels were correct

def evaluate_knn(knn, test_loader, label_encoder, device, feature_model):
    correct = 0
    total = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            feats = feature_model(imgs)
            feats = feats.cpu().numpy()

        preds = knn.predict(feats)

        true_names = label_encoder.inverse_transform(labels.numpy())

        correct += np.sum(preds == true_names)
        total += len(true_names)

    return correct / total

