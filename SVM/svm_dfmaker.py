from __future__ import print_function, division

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix, classification_report
import pickle

SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

dataset = "gazeraw"
pathdir = ""
modelpath = os.path.join(pathdir, f'VGG16_finalmodel_{dataset}.pt')
data_dir = os.path.join(pathdir, f"Data/{dataset}")

TRAIN = 'train'
VAL = 'validation'
TEST = 'test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=1,
        shuffle=True,
        # sampler=rus if TRAIN else None,
        num_workers=0
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

class_names = image_datasets[TRAIN].classes
print(f"Classes: {image_datasets[TRAIN].classes}")

for category in [TRAIN, TEST, VAL]:
    print(f"Gathering Dataset {category}...")
    Xs = []
    Ys = []

    for i, d in enumerate(dataloaders[category]):
        x, y = d
        Xs.append(x.numpy().flatten())
        Ys.append(y.item())

    df = pd.DataFrame({'X': Xs, 'y': Ys})

    print(f"Pickling Dataset {category}...")
    with open(os.path.join('../archive/dataframes', f'{dataset}_{category}.pkl'), "wb") as file:
        pickle.dump(df, file)

    # with open(f'{dataset}_{category}.pkl', "rb") as file:
    #     df_ = pickle.load(file)

# clf.fit(train_df['X'].to_numpy().tolist(), train_df['y'].to_numpy())



