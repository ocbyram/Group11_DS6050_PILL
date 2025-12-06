# Group11_DS6050_PILL
The repository for our DS6050 Final Project titled P.I.L.L.

This repository includes our full pipeline for the beginning of our P.I.L.L. Project. Our goal was to build and evaluate baseline models for pill image classification utilizing the NLM dataset. Within our notebook file, we load and verify the dataset, checking for missing files, and encode over 2100 unique pill classes. After defining our transforms and building a custom PyTorch class for our Dataset, we create a 70-15-15 training/validation/test split and train a ResNet18 model implemented from scratch. While this model did learn, its accuracy remained fairly low, telling us that perhaps training a deep CNN from scratch on a dataset with thousands of classes and limited images may be too tall a task.

Instead, we shifted to what proved to be a more effective approach, which was using a pretrained ResNet18 for feature extraction. With ImageNet weights and a replaced final layer (to match the number of classes) this model trained quickly and reached strong performance, with almost a 90% validation accuracy in just 10 epochs. After saving the pretrained model, we convert it into a feature extractor and generate 512-dim embeddings for every image, storing them in a dictionary. As next steps, we plan on feeding these embeddings into a small RNN classifier.

Repositiory Contents Include:

Data Folder: A folder containing another folder titled '300'. This includes all of the image data.

Milestone 1 Folder: A folder containing the milestone 1 proposal.

Milestone 2 folder: A folder containing a 'Results' folder (with our training curve graphs and metrics table), the jupyter notebook 'Group11_Milestone2_Code.ipynb' (the code for our baseline model and graphs), and Group11_checkpoint.pdf (our write up for milestone 2).

Final folder: A folder containing our final config.yaml, eval_retrieval.py, load_nih_pills.py, train_knn_retrieval.py, and train_resnet18_classifier.py files.

### Instructions for reproducibility:

To run the entirety of our pipeline, install all required packages and run the main.py file.

The code is however modular and functions can be imported from the different scripts and utilized as needed.

In order to train the Resnet18 you can call the following code:

```python

from train_resnet18_classifier import train_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2100

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_resnet(device, num_classes, train_loader, val_loader, criterion)

```

In order to train the knn on the resnet18 embeddings, run the following code:

```python

from sklearn.neighbors import KNeighborsClassifier
from train_knn_retrieval import train_knn

features_dict = {}

    for idx, row in df_nlm.iterrows():
        label = row["label"]
        img_path = row["full_path"]

        print("Extracting:", label)
        features_dict[label] = extract_features(img_path)

    torch.save(features_dict, save_path)
    print("Saved new features to:", save_path)

X, y, knn = train_knn(features_dict)

```
And to evalute the final knn on resnet18 embeddings:

```python

from load_nih_pills import load_nih_pills
df_nlm, le, num_classes = load_nih_pills(df)
file_path = 'table.csv'
df = pd.read_csv(file_path)

from eval_retrieval import evaluate_knn
knn_test_acc = evaluate_knn(knn, test_loader, le, device, feature_model)


```