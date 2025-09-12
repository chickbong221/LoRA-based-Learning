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

def build_model(args, use_progressive_lora: bool = False, num_classes: int = 100, device: torch.device = None):
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
    
    progressive_lora = None
    
    if use_progressive_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Create progressive LoRA
        lora_config = LoraConfig(
            r=args.start_rank,
            lora_alpha=int(2 * args.start_rank),  # alpha = 2 * rank
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Initialize progressive LoRA system
        progressive_lora = ProgressiveLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        # Apply hooks for LoRA forward pass
        progressive_lora.apply_progressive_lora_hooks(model)
        
        print(f"Progressive LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, progressive_lora

class ProgressiveLoRAModule:
    """Handles progressive expansion of LoRA matrices."""

    def __init__(self, model, config: LoraConfig, target_modules, device):
        self.model = model
        self.config = config
        self.target_modules = target_modules
        self.device = device
        self.current_rank = config.r

        # Initialize the first LoRA modules with the initial rank
        self._initialize_lora_modules()

    def _initialize_lora_modules(self):
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if hasattr(module, "weight"):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]

                    # Create LoRA parameters with the initial rank
                    lora_A_trainable = nn.Parameter(torch.empty(self.current_rank, in_features, device=self.device))
                    nn.init.kaiming_uniform_(lora_A_trainable, a=math.sqrt(5))

                    lora_B_trainable = nn.Parameter(
                        torch.zeros(out_features, self.current_rank, device=self.device))

                    # Attach directly to the module. These are the initial trainable parameters.
                    setattr(module, "lora_A_trainable", lora_A_trainable)
                    setattr(module, "lora_B_trainable", lora_B_trainable)

                    # Freeze original weights
                    module.weight.requires_grad = False

    def expand_rank(self, new_rank):
        """
        Expands the rank of the LoRA matrices while preserving learned weights.
        Old weights are frozen, while new weights are added and made trainable.
        """
        print(f"Expanding LoRA rank from {self.current_rank} to {new_rank}")

        for name, module in self.model.named_modules():
            # Check for the existence of the trainable parameters
            if hasattr(module, "lora_A_trainable") and hasattr(module, "lora_B_trainable"):
                # Get the old parameters and freeze them
                old_A_trainable = getattr(module, "lora_A_trainable")
                old_B_trainable = getattr(module, "lora_B_trainable")
                
                # Move old parameters to a "frozen" attribute
                setattr(module, "lora_A_frozen", nn.Parameter(old_A_trainable.data, requires_grad=False))
                setattr(module, "lora_B_frozen", nn.Parameter(old_B_trainable.data, requires_grad=False))
                
                # Delete old attributes to prepare for creating new ones
                delattr(module, "lora_A_trainable")
                delattr(module, "lora_B_trainable")

                in_features = old_A_trainable.shape[1]
                out_features = old_B_trainable.shape[0]

                # Calculate the new rank to add
                new_trainable_rank = new_rank - self.current_rank

                # Create new trainable parameters
                new_A_trainable = nn.Parameter(torch.empty(new_rank, in_features, device=self.device))
                nn.init.kaiming_uniform_(new_A_trainable, a=math.sqrt(5))

                new_B_trainable = nn.Parameter(
                    torch.zeros(out_features, new_trainable_rank, device=self.device))

                # Attach the new parameters to the module
                setattr(module, "lora_A_trainable", new_A_trainable)
                setattr(module, "lora_B_trainable", new_B_trainable)
        
        self.current_rank = new_rank
        print(f"Rank successfully expanded to {self.current_rank}")

    def apply_progressive_lora_hooks(self, model):
        """
        Applies forward hooks to integrate LoRA computations.
        This hook safely handles both the initial and expanded states.
        """
        def create_hook():
            def hook(module, input, output):
                # Calculate the scaling factor based on the current rank
                alpha = self.config.lora_alpha * self.current_rank
                scaling = alpha / self.current_rank
                
                x = input[0]

                # Check and compute output from both matrices (frozen and trainable)
                if hasattr(module, "lora_A_frozen") and hasattr(module, "lora_A_trainable"):
                    # Combine the matrices to perform a single operation
                    lora_A_combined = torch.cat([module.lora_A_frozen, module.lora_A_trainable], dim=0)
                    lora_B_combined = torch.cat([module.lora_B_frozen, module.lora_B_trainable], dim=1)
                    
                    temp = torch.nn.functional.linear(x, lora_A_combined)
                    lora_output = torch.nn.functional.linear(temp, lora_B_combined) * scaling
                    return output + lora_output
                
                # Initial state (only the trainable matrix exists)
                elif hasattr(module, "lora_A_trainable"):
                    temp = torch.nn.functional.linear(x, module.lora_A_trainable)
                    lora_output = torch.nn.functional.linear(temp, module.lora_B_trainable) * scaling
                    return output + lora_output
                
                return output

            return hook

        for name, module in model.named_modules():
            # Only apply the hook if the module has LoRA parameters
            if hasattr(module, "lora_A_trainable") or hasattr(module, "lora_A_frozen"):
                module.register_forward_hook(create_hook())

def train_with_progressive_lora(args, train_loader, val_loader, device, wandb_run):
    """Train model with progressive LoRA rank expansion"""
    
    start_time = time.time()
    
    # Build model with progressive LoRA
    if args.use_pretrained:
        model, progressive_lora = build_model_pretrain(use_progressive_lora=True, num_classes=100, device=device, args=args)
    else:
        model, progressive_lora = build_model(use_progressive_lora=True, num_classes=100, device=device, args=args)
    
    # Optimizer AdamW with fixed learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    ce = nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"Training WITH Progressive LoRA")
    print(f"Starting rank: {args.start_rank}, Max rank: {args.max_rank}")
    print(f"Expansion every {args.expansion_epochs} epochs")
    print(f"{'='*50}")
    
    best_val_acc = 0.0
    results = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': [],
        'best_val_acc': 0.0,
        'rank_history': [],
        'time_history': []   # <-- store accumulated time
    }
    
    current_rank = args.start_rank
    loss_history = []

    min_improvement = 0.001  # Minimum loss reduction threshold
    patience = 8  # Number of epochs to wait before considering plateau
    
    for epoch in range(args.epochs):
        # Check if we need to expand rank based on loss plateau
        should_expand = False
        
        if (epoch > 0 and current_rank < args.max_rank and len(loss_history) >= patience):
            # Check if loss has plateaued
            recent_losses = loss_history[-patience:-1]
            improvement = min(recent_losses) - loss_history[-1]
            loss_plateaued = improvement < min_improvement
            
            # Check timing constraints
            epochs_since_expansion = 10
            if hasattr(args, 'last_expansion_epoch'):
                can_expand_timing = (epoch - args.last_expansion_epoch) >= epochs_since_expansion
            else:
                can_expand_timing = epoch >= epochs_since_expansion
            
            should_expand = loss_plateaued and can_expand_timing
        
        # Expand rank if conditions met
        if should_expand:
            new_rank = min(current_rank * 2, args.max_rank)
            direct_fusion_lora.expand_rank(new_rank)
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Rank expanded to {new_rank} at epoch {epoch+1} (improvement: {improvement:.6f})")
            
            current_rank = new_rank
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            args.last_expansion_epoch = epoch
        
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
        results['rank_history'].append(current_rank)

        print(f"Epoch {epoch+1}/{args.epochs} (Rank: {current_rank}) - "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | "
              f"Time {elapsed_time:.2f}s")

        # Log to wandb with proper step
        if wandb_run:
            suffix = 'progressive_lora'
            log_data = {
                f"{suffix}/epoch": epoch,
                f"{suffix}/train_loss": train_loss,
                f"{suffix}/train_acc": train_acc,
                f"{suffix}/val_loss": val_loss,
                f"{suffix}/val_acc": val_acc,
                f"{suffix}/current_rank": current_rank,
                f"{suffix}/elapsed_time": elapsed_time,
            }
            wandb_run.log(log_data)
    
    end_time = time.time()
    training_time = end_time - start_time
    results['training_time'] = training_time
    
    print(f"Progressive LoRA training completed in {training_time:.2f} seconds")
    
    return results

def build_model_pretrain(args, use_progressive_lora: bool = False, num_classes: int = 100, device: torch.device = None):
    # Load pretrained DeiT-Tiny từ Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=num_classes,   # đổi số lớp cho phù hợp dataset của bạn
        ignore_mismatched_sizes=True  # cho phép thay đổi output layer
    )
    
    progressive_lora = None
    
    if use_progressive_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Create progressive LoRA
        lora_config = LoraConfig(
            r=args.start_rank,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Initialize progressive LoRA system
        progressive_lora = ProgressiveLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        # Apply hooks for LoRA forward pass
        progressive_lora.apply_progressive_lora_hooks(model, progressive_lora)
        
        print(f"Progressive LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, progressive_lora
