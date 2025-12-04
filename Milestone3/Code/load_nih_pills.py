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

from google.colab import drive
drive.mount("/content/drive")

# This is a path to the table that has image file names along with the actual names of the pills
# This is what we will use so that we have the accurate pill names for training and testing

file_path = '/content/drive/MyDrive/table.csv'
df = pd.read_csv(file_path)
df.head()

# Import dependencies again

import pandas as pd
import os

# Accessing specifically the NLM images since these are only of the pill and do not have
# any words in the background of the picture (the rxnav images have that)

df = df[['name', 'nlmImageFileName']]
image_dir = "/content/drive/MyDrive/600"

df['full_path'] = df['nlmImageFileName'].apply(
    lambda x: os.path.join(image_dir, os.path.basename(str(x)))
)

df['exists'] = df['full_path'].apply(os.path.exists)

# Reporting results to make sure the dataframe exists and loaded all of the images

total = len(df)
found = df['exists'].sum()
missing = total - found

# Printing to make sure we did not lose any images in this process

print(f"Found {found} out of {total} images.")
print(f"Missing {missing} images")

# If we miss any images we want to know which ones they are

print("\nMissing files (first 10):")
print(df.loc[~df['exists'], 'nlmImageFileName'].head(10))

# Keep only the rows with existing images

df_clean = df[df['exists']].reset_index(drop=True)

# Save to CSV for later

df_clean.to_csv("nlm_images_verified.csv", index=False)
print("\n Saved verified file list to 'nlm_images_verified.csv'")

# Read in final image csv

pd.read_csv("nlm_images_verified.csv")

# Import dependencies (again)

# we are using this specifically because we want to convert the pill names into
# numeric ids, which are easier to work with

# We sorted the numeric ids and counted them to see the number of unique classes (pills)
# we have images of

from sklearn.preprocessing import LabelEncoder
df_nlm = pd.read_csv("nlm_images_verified.csv")
df_nlm["pill_id"] = df_nlm["nlmImageFileName"].str.split("_").str[0]
le = LabelEncoder()
df_nlm = df_nlm.rename(columns={'name': 'label'})
df_nlm['label_id'] = le.fit_transform(df_nlm['label'])
num_classes = len(le.classes_)
print("Classes:", num_classes)

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