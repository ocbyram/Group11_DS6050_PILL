
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models

from eval_retrieval import evaluate

def train_one_epoch(model, loader, optimizer, device, criterion):
   model.train()
   total_loss = 0
   correct = 0
   total = 0


   for imgs, labels in loader:
       imgs, labels = imgs.to(device), labels.to(device)

       optimizer.zero_grad()
       outputs = model(imgs)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()

       total_loss += loss.item() * imgs.size(0)
       _, preds = outputs.max(1)
       correct += preds.eq(labels).sum().item()
       total += labels.size(0)


   return total_loss / total, correct / total

def train_resnet(device, num_classes, train_loader, val_loader, criterion):
    model = models.resnet18(weights="IMAGENET1K_V1")

# We replaced the last layer with our own number of classes then saved the model

    model.fc = nn.Linear(512, num_classes)
    model = model.to(device)

# Finally, we optimized the parameters within the model

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

# We want to collect loses and accuracies so that we can assess the model metrics
# and create the training loss charts

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

# Chose to train for 10 epochs again! Will likely change to 20 or 30 in milestone 3.

    epochs = 10

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

# Printing our training and val losses!

        print(f"Epoch {epoch:02d}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print("-" * 20)
    return model,train_losses,val_losses,train_accuracies,val_accuracies
