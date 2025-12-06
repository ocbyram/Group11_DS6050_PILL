import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
import torchvision.models as models



def load_nih_pills(df):
    df = df[['name', 'nlmImageFileName']]
    image_dir = "600"

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
    return df_nlm,le,num_classes