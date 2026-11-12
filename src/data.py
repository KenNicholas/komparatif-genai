import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def get_loaders(batch_size=128, seed=42):
    tf = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=tf)
    test_full  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=tf)

    X_all = np.concatenate([
        np.stack([train_full[i][0].numpy() for i in range(len(train_full))]),
        np.stack([test_full[i][0].numpy()  for i in range(len(test_full))])
    ], axis=0)
    y_all = np.concatenate([
        np.array([train_full[i][1] for i in range(len(train_full))]),
        np.array([test_full[i][1]  for i in range(len(test_full))])
    ], axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    test_set  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set,   batch_size=batch_size, shuffle=False),
            DataLoader(test_set,  batch_size=batch_size, shuffle=False))
