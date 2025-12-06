# These are necessary dependencies that we have collected from class and the D2L textbook

import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# We all saved the images and metadata table to our Google Drives, since it is a large
# amount of data and we knew that we would be using Google Colab to create this code

# This is us connecting to Drive so that we can access the images/metadata


# This is a path to the table that has image file names along with the actual names of the pills
# This is what we will use so that we have the accurate pill names for training and testing

file_path = 'table.csv'
df = pd.read_csv(file_path)
df.head()

# Import dependencies again

import pandas as pd
import os

# Accessing specifically the NLM images since these are only of the pill and do not have
# any words in the background of the picture (the rxnav images have that)

from load_nih_pills import load_nih_pills



df_nlm, le, num_classes = load_nih_pills(df)


class PillDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Our images need to be in RGB to work!
        img = Image.open(row['full_path']).convert('RGB')
        img = self.transform(img)
        label = int(row['label_id'])
        return img, label
    

# The numbers in .Normalize() are the standard to use for ImageNet since we are feeding them
# to ResNet

# We are transforming the data to make it standard
# and then packing the images into a tensor so that the network
# processes them without an issue


# train_tfms is to transform the data and helps the model with generalization

# We rotated, flipped, and jittered as forms of data augmentation

train_tfms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.1, contrast=0.1,
        saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# test_tfms deals with the test data and
# does not include the rotation or flip transformations
# because we want the model to deal with the actual image without any artificial
# changes

test_tfms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# This is us creating the full, transformed dataset read for us to split into test/train
# and perform training

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Get unique pill IDs

pill_ids = df_nlm["pill_id"].unique()

# We split into a 70/30 for train and test with random seed of 42

train_ids, temp_ids = train_test_split(
    pill_ids, test_size=0.30, random_state=42, shuffle=True
)

# Now we split the test into 50/50
# All together, this gives us a 70/15/15 split for training/test/val

val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, random_state=42, shuffle=True
)

# This helps us to ensure that we split at the pill level
# This is critical so each pill identity/id is assigned to only one split

train_df = df_nlm[df_nlm["pill_id"].isin(train_ids)]
val_df   = df_nlm[df_nlm["pill_id"].isin(val_ids)]
test_df  = df_nlm[df_nlm["pill_id"].isin(test_ids)]

# Applying the transformations that we specified above

train_dataset = PillDataset(train_df, transform=train_tfms)
val_dataset   = PillDataset(val_df,   transform=test_tfms)
test_dataset  = PillDataset(test_df,  transform=test_tfms)

# Using dataloader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Here are the number of unique pills in each train/test/val

print("Number of unique pills:")
print("Train:", train_df["pill_id"].nunique())
print("Val  :", val_df["pill_id"].nunique())
print("Test :", test_df["pill_id"].nunique())

# Number of images in each train/test/val

print("\nNumber of images:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

train_classes = train_df['label'].nunique()
val_classes   = val_df['label'].nunique()
test_classes  = test_df['label'].nunique()

print("Unique classes per split:")
print("Train:", train_classes)
print("Val  :", val_classes)
print("Test :", test_classes)


# Now we would like to create ResNet18

# Similar to what we did in class and on homework assignment 2,
# we are using 2 3x3 convolutions followed by BatchNorm2d and ReLU

# We also used a skip connection, just like the homework and textbook

# This is the basic way that resNet18 is implemented

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(out_channels)
             )

# This is standard for our convolutions and including the identity skip connection

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out +=  self.shortcut(identity)
        out = self.relu(out)
        return out
    
    

# import dependency again!


from torch import nn


# This class actually builds the ResNet18. We will use the BasicBlock class
# from above to perform convolutions + BatchNorm2d + ReLU


class ResNet18(nn.Module):
   def __init__(self, num_classes=18, return_features=False):
       super(ResNet18, self).__init__()
       self.return_features = return_features
       self.conv1 = nn.Sequential(
           nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       )


       self.layer1 = self._make_layer(64, 64, 2, stride=1)
       self.layer2 = self._make_layer(64, 128, 2, stride=2)
       self.layer3 = self._make_layer(128, 256, 2, stride=2)
       self.layer4 = self._make_layer(256, 512, 2, stride=2)


       self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
       self.fc = nn.Linear(512, num_classes)


   def _make_layer(self, in_channels, out_channels, num_blocks, stride):
       blocks = []
       blocks.append(BasicBlock(in_channels, out_channels, stride))
       for _ in range(1, num_blocks):
           blocks.append(BasicBlock(out_channels, out_channels))
       return nn.Sequential(*blocks)


   def forward(self, x):
       x = self.conv1(x)
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)


       if self.return_features:
           return x


       x = self.avgpool(x)
       x = torch.flatten(x, 1)
       x = self.fc(x)
       return x


# Actually creating the ResNet18 model and saving it to our devices
model = ResNet18(num_classes=num_classes, return_features=False)
model = model.to(device)


# This is so that we can figure out the loss!


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Now we get to train the model!
# Again, most of this code comes from class and homework assignments
# When training we make sure to collect metrics on loss so that we are able to do error analysis and make training curves

from train_resnet18_classifier import train_one_epoch


# Setting up so we can evaluate the model with the data we set aside

from eval_retrieval import evaluate

# We chose 10 epochs since this is just a baseline model. We will likely bump it to 20 or 30 epochs in our next milestone

# For the baseline, we just want to see the accuracy for top-1 (k=1)

epochs = 10


for epoch in range(1, epochs+1):
   train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion)
   val_loss, val_acc = evaluate(model, val_loader, device, criterion)


# Getting the loss and accuracy!
   print(f"Epoch {epoch:02d}:")
   print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
   print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
   print("-" * 20)


# The ResNet18 that we built from scratch did not work the way that we thought it would, 
# with a final train accuracy of 0.0463 and val accuracy of 0.0274. We decided that we would try feature extraction 
# with RNN to see if that performed better.

#Feature Extraction ResNet18 + RNN**

#First is the Pretrained ResNet18 Model**

# Importing even more dependencies!

import torchvision.models as models
import torch.nn as nn

# Since we are doing feature extraction, we do not use the ResNet18 that we built from scratch,
# we just use the .resnet18 function. We set the weights as "IMAGENET1K_V1" so
# it would be pretrained on ImageNet

from train_resnet18_classifier import train_resnet


model, train_losses, val_losses, train_accuracies, val_accuracies = train_resnet(device, num_classes, train_loader, val_loader, criterion)
    
# Making our training/val loss and accuracy curves!

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# This is our final test accuracy

test_loss, test_acc = evaluate(model, val_loader, device, criterion)
print("Final Test Accuracy:", test_acc)

# This is how we evaluated the softmax classifier (Pretrained ResNet18 Model)
# We wanted to get the recall@1 and recall@5

# Essentially just grabbed the model and the test data and calculated if the top 5 labels
# were correct

from eval_retrieval import evaluate_softmax


# We are getting the softmax recall@1 and recall@5 and printing it out for the softmax classifier

soft_r1, soft_r5 = evaluate_softmax(model, test_loader, device)
print("Softmax Recall@1:", soft_r1)
print("Softmax Recall@5:", soft_r5)

# We are getting the softmax recall@1 and recall@5 and printing it out for the softmax classifier

soft_r1, soft_r5 = evaluate_softmax(model, test_loader, device)
print("Softmax Recall@1:", soft_r1)
print("Softmax Recall@5:", soft_r5)

# We have to process the images like we did in our ResNet18 from scratch before
# we can extract the features

preprocess = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

# Time to extract the model features!

# Converting them to RGB, processing them so they are in a tensor and normalized,
# and extracting the features

@torch.no_grad()
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    feats = feature_model(x).cpu().squeeze(0)
    return feats

# Running the model for the features then loading it as a dictionary

feature_model = models.resnet18(weights=None)
feature_model.fc = nn.Identity()
feature_model = feature_model.to(device)
feature_model.load_state_dict(model.state_dict(), strict=False)

# The output of this function is the 512-dim feature vector

feature_model.eval()

# We saved the features to a dictionary to make them accessible

save_path = "pill_features.pt"
create_new_features_dict = True

if create_new_features_dict:
    features_dict = {}

    for idx, row in df_nlm.iterrows():
        label = row["label"]
        img_path = row["full_path"]

        print("Extracting:", label)
        features_dict[label] = extract_features(img_path)

    torch.save(features_dict, save_path)
    print("Saved new features to:", save_path)

else:
    features_dict = torch.load(save_path)
    print("Loaded feature dict with", len(features_dict), "entries.")
    

import torch.optim as optim

# Have to create sequences

def rnn_sequence(embed):
    embed = embed.flatten()
    seq = embed.unsqueeze(1).unsqueeze(1)
    return seq

# Getting the inputs for RNN from the feature_dicts that we made above

rnn_inputs = {}
for pill_name, embed_tensor in features_dict.items():
    rnn_inputs[pill_name] = rnn_sequence(embed_tensor)

# This is the RNN!

rnn = nn.RNN(input_size=1,
             hidden_size=256,
             num_layers=2,
             nonlinearity='relu',
             bidirectional=True).to(device)

# Getting all of the distinct pill names from the rnn inputs (pills in feature dicts)
# So we know the number of classes

pill_names = list(rnn_inputs.keys())
num_classes = len(pill_names)

# We need this to be linear

fc = nn.Linear(256*2, num_classes).to(device)

# We did it this way instead of using the LabelEncoder like we did for the scratch ResNet18
# Since we can easily grab it from the features dict we made

pill_to_idx = {name: i for i, name in enumerate(pill_names)}

# Getting loss and optimizing

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=1e-3)

# This is similar to the training we did for the scratch ResNet18, with
# a few adjustments

def train_one_epoch(rnn, fc, rnn_inputs, optimizer, criterion):
    rnn.train()
    fc.train()
    total_loss = 0
    correct = 0
    total = 0

    # Labeling

    for pill_name, seq_features in rnn_inputs.items():
        seq_features = seq_features.to(device)
        label_idx = torch.tensor([pill_to_idx[pill_name]], device=device)

        optimizer.zero_grad()
        rnn_out, hidden = rnn(seq_features)

        forward = hidden[-2, :, :]
        backward = hidden[-1, :, :]
        rnn_vector = torch.cat([forward, backward], dim=1)

        logits = fc(rnn_vector)
        loss = criterion(logits, label_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_idx = torch.argmax(logits, dim=1).item()
        if predicted_idx == label_idx.item():
            correct += 1
        total += 1

    return total_loss / total, correct / total

# The evaluate function like the one in the from scratch ResNet18!

def evaluate(rnn, fc, rnn_inputs, criterion):
    rnn.eval()
    fc.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for pill_name, seq_features in rnn_inputs.items():
            seq_features = seq_features.to(device)
            label_idx = torch.tensor([pill_to_idx[pill_name]], device=device)

            rnn_out, hidden = rnn(seq_features)
            forward = hidden[-2, :, :]
            backward = hidden[-1, :, :]
            rnn_vector = torch.cat([forward, backward], dim=1)

            logits = fc(rnn_vector)
            loss = criterion(logits, label_idx)

            total_loss += loss.item()
            predicted_idx = torch.argmax(logits, dim=1).item()
            if predicted_idx == label_idx.item():
                correct += 1
            total += 1

    return total_loss / total, correct / total

# Training loop just like the one in the from scratch ResNet18

train_losses, train_accuracies = [], []

for epoch in range(1, 11):
    train_loss, train_acc = train_one_epoch(rnn, fc, rnn_inputs, optimizer, criterion)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

# Plotting our training curves

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.title('RNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.title('RNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#Adding K-NN to the Pretrained ResNet18 Model

# Here we are adding K-NN to the Pretrained model

# The majority of this code is sourced from class as well as the D2L textbook

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from train_knn_retrieval import train_knn

X, y, knn = train_knn(features_dict)

# Actually predicting the pill

def predict_pill(img_tensor):
    model.eval()
    with torch.no_grad():
        feats = model(img_tensor.unsqueeze(0).to(device))
        feats = feats.flatten().cpu().numpy()

    return knn.predict([feats])[0]

# We asked it to return the top 5 pill matches so that we had a greater
# range in case the first one was incorrect since these are difficult to predict

def retrieve_top_k(img_path, k=5):
    feats = extract_features(img_path).flatten().cpu().numpy()
    distances, indices = knn.kneighbors([feats], n_neighbors=k)
    return y[indices[0]]

# Predicting on the training data and getting the accuracy

preds = knn.predict(X)
accuracy = np.mean(preds == y)
print("Training accuracy:", accuracy)

# Grabbing the top 5 for an example pill (Loperamide Hydrochloride 2 MG Oral Capsule)

retrieve_top_k(r"Data/600/00093-0311-01_NLMIMAGE10_6315B1FD.jpg", k=5)

# We wanted to look at the accuracy metric as a way to initially identify how well
# This model performed

# Essentially just went throgh and predicted each with knn
# Got the true names and evaluated accuracy


# We ran this on our test accuracy since we did training accuracy above
from eval_retrieval import evaluate_knn

knn_test_acc = evaluate_knn(knn, test_loader, le, device, feature_model)
print("KNN Test Accuracy:", knn_test_acc)

# After seeing accuracy, we wanted to evaluate recall@1, recall@5, recall@10, as
# well as MRR at 1, 5, and 10 and the time the model took to run
# (These metrics were suggested by Professor Shakeri)

# Very similar to what we did above, we basically just created a function
# to loop through and calculate the important metrics
topk_list=[1,5,10]

import time

def evaluate_knn_metrics(knn, test_loader, label_encoder, topk_list=[1,5,10]):
    recall_totals = {k: 0 for k in topk_list}
    mrr_totals = {k: 0 for k in topk_list}
    total_images = 0
    times = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.numpy()
        true_names = label_encoder.inverse_transform(labels)

        with torch.no_grad():
            start_time = time.time()
            feats = feature_model(imgs).cpu().numpy()
            end_time = time.time()

        times.append((end_time - start_time) / len(imgs))

        for k in topk_list:
            distances, indices = knn.kneighbors(feats, n_neighbors=k)
            topk_preds = np.array(y)[indices]

            for i in range(len(true_names)):
                total_images += 1

                recall_totals[k] += 1 if true_names[i] in topk_preds[i] else 0

                if true_names[i] in topk_preds[i]:
                    rank = np.where(topk_preds[i] == true_names[i])[0][0] + 1
                    mrr_totals[k] += 1 / rank
                else:
                    mrr_totals[k] += 0

    avg_recall = {k: recall_totals[k] / total_images for k in topk_list}
    avg_mrr = {k: mrr_totals[k] / total_images for k in topk_list}
    avg_time = np.mean(times)

    return avg_recall, avg_mrr, avg_time

# Running the metrics on the test data!

recalls, mrrs, avg_time = evaluate_knn_metrics(knn, test_loader, le)

for k in topk_list:
    print(f"Recall@{k}: {recalls[k]:.4f}, MRR@{k}: {mrrs[k]:.4f}")
print(f"Average inference time per image: {avg_time:.4f} seconds")


# Part of the projetc was to perform an error analysis
# To do this we essentially showed the true label, predicted label, and the top 5 labels returned

# If the predicted label was incorrect for the true label, it was counted as an error

# We printed the length of our errors and results

# 598 errors out of 665 results, so only 67 were correct

def get_knn_predictions(knn, test_loader, label_encoder, k=5):
    results = []

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            feats = feature_model(imgs).cpu().numpy()

        top1_preds = knn.predict(feats)

        distances, indices = knn.kneighbors(feats, n_neighbors=k)
        topk_preds = y[indices]

        true_names = label_encoder.inverse_transform(labels.numpy())

        for i in range(len(true_names)):
            results.append({
                "true": true_names[i],
                "top1": top1_preds[i],
                "topk": topk_preds[i].tolist(),
            })

    return results

results = get_knn_predictions(knn, test_loader, le, k=5)

errors = [r for r in results if r["true"] != r["top1"]]
len(errors), len(results)


# Printing out ten examples of our error analysis

for r in errors[:10]:
    print("TRUE:", r["true"])
    print("PRED:", r["top1"])
    print("TOP-5:", r["topk"])
    print("---")
    
from collections import Counter

# Counting the number of true errors and seeing the most common pills that were errored on

true_error_counts = Counter([e["true"] for e in errors])
true_error_counts.most_common(20)

confusions = Counter([(e["true"], e["top1"]) for e in errors])
confusions.most_common(20)


confusions = Counter([(e["true"], e["top1"]) for e in errors])
confusions.most_common(20)

import pandas as pd

# Creating a dataframe of the most commonly confused pills so that we can create a confusion matrix/heatmap

conf_df = pd.DataFrame(confusions.most_common(), columns=["pair", "count"])
conf_df.to_csv("confusion_pairs.csv")

import numpy as np
import seaborn as sns

# Creating a heatmap showing confused class pairs as well as a list of the 25 most confused pairs

confusions = Counter([(e["true"], e["top1"]) for e in errors])

top_pairs = confusions.most_common(25)
true_labels = [p[0][0] for p in top_pairs]
pred_labels = [p[0][1] for p in top_pairs]
values = [p[1] for p in top_pairs]

plt.figure(figsize=(12,6))
sns.heatmap(
    np.array(values).reshape(-1,1),
    annot=True,
    yticklabels=[f"{t} → {p}" for t,p in zip(true_labels,pred_labels)],
    xticklabels=["Confusion Count"],
    cmap="Reds"
)
plt.title("Top 25 Most Confused Class Pairs")
plt.show()

#real world examples
retrieve_top_k("Data/real_world/Amoxicillin 200 MG Clavulanate 28.5 MG Chewable Tablet.JPG")
retrieve_top_k("Data/real_world/Bethanechol Chlorine 10 mg Tablet.jpg")
retrieve_top_k("Data/real_world/ferrous sulfate 325 MG Oral Tablet.jpg")
retrieve_top_k("Data/real_world/folic acid 1 MG Oral Tablet.jpg")
retrieve_top_k("Data/real_world/Prazosin 2 MG Oral Capsule.jpg")