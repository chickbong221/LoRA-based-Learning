import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import transforms
from torchvision.datasets import Flowers102

from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model

import wandb

def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Dataset & Dataloaders
# ----------------------------

def build_cifar100_datasets(img_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Tập train 50k ảnh
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=tf
    )

    # Tập test 10k ảnh → dùng làm val
    val_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=tf
    )

    return train_set, val_set

def build_flower_datasets(img_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Training dataset
    train_set = Flowers102(
        root="./data",
        split="train",
        download=True,
        transform=tf
    )

    # Validation dataset -> sử dụng split "test"
    val_set = Flowers102(
        root="./data",
        split="test",
        download=True,
        transform=tf
    )

    return train_set, val_set


# ----------------------------
# Evaluation
# ----------------------------

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = ce(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total