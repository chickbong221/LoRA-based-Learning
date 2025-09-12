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

from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model
from utils import *

import wandb

# ----------------------------
# Model Creation
# ----------------------------

def build_model(use_progressive_lora: bool = False, num_classes: int = 100, device: torch.device = None):
    # Create ViT-tiny configuration without loading pretrained weights
    config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=num_classes,
    )
    
    # Create model with random initialization (no pretrained weights)
    model = ViTForImageClassification(config)

    # Print number of trainable parameters after expansion
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after expansion: {trainable_params:,}")
    
    progressive_lora = None
    
    if device:
        model = model.to(device)
    
    return model, progressive_lora

def train_single_run(args, use_lora: bool, train_loader, val_loader, device, wandb_run):
    """Train model for one configuration (with fixed LoRA or full fine-tuning)"""
    
    start_time = time.time()
    
    # Build model
    if args.use_pretrained:
        model, _ = build_model_pretrain(use_progressive_lora=False, num_classes=100, device=device)
    else:
        model, _ = build_model(use_progressive_lora=False, num_classes=100, device=device)
    
    if use_lora:
        # Apply traditional LoRA with rank 8
        fixed_rank = 8
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        lora_config = LoraConfig(
            r=fixed_rank,
            lora_alpha=int(1.5 * fixed_rank),  # alpha = 3/2 * rank
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        model = get_peft_model(model, lora_config)
        print("Traditional LoRA applied. Trainable parameters:")
        model.print_trainable_parameters()
    
    # Optimizer SGD with fixed learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    ce = nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"Training {'WITH Traditional LoRA (rank=8)' if use_lora else 'WITHOUT LoRA (Full Fine-tuning)'}")
    print(f"{'='*50}")
    
    best_val_acc = 0.0
    results = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'best_val_acc': 0.0,
        'time_history': []   # <-- store accumulated time
    }
    
    for epoch in range(args.epochs):
        model.train()
        running_loss, running_acc, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = ce(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)
            total += bs
            running_loss += loss.item() * bs
            running_acc += correct

        train_loss = running_loss / total
        train_acc = running_acc / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # Accumulated training time
        elapsed_time = time.time() - start_time
        results['time_history'].append(elapsed_time)

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Store results
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['val_losses'].append(val_loss)
        results['val_accs'].append(val_acc)
        results['best_val_acc'] = best_val_acc

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | "
              f"Time {elapsed_time:.2f}s")

        # Log to wandb with proper step
        if wandb_run:
            suffix = 'lora' if use_lora else 'full_ft'
            log_data = {
                f"{suffix}/epoch": epoch,
                f"{suffix}/train_loss": train_loss,
                f"{suffix}/train_acc": train_acc,
                f"{suffix}/val_loss": val_loss,
                f"{suffix}/val_acc": val_acc,
                f"{suffix}/elapsed_time": elapsed_time,   # <-- log accumulated time
            }
            wandb_run.log(log_data)
    
    end_time = time.time()
    training_time = end_time - start_time
    results['training_time'] = training_time
    
    approach = 'Traditional LoRA' if use_lora else 'Full Fine-tuning'
    print(f"{approach} training completed in {training_time:.2f} seconds")
    
    return results

def build_model_pretrain(use_progressive_lora: bool = False, num_classes: int = 100, device: torch.device = None):
    # Load pretrained DeiT-Tiny từ Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=num_classes,   # đổi số lớp cho phù hợp dataset của bạn
        ignore_mismatched_sizes=True  # cho phép thay đổi output layer
    )

    # In số lượng tham số trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    progressive_lora = None

    if device:
        model = model.to(device)

    return model, progressive_lora