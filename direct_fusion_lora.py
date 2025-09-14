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

def build_model(args, use_direct_fusion_lora: bool = False, num_classes: int = 100, device: torch.device = None):
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
    
    direct_fusion_lora = None
    
    if use_direct_fusion_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Create direct_fusion LoRA
        lora_config = LoraConfig(
            r=args.start_rank,
            lora_alpha=int(2 * args.start_rank),  # alpha = 2 * rank
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Initialize direct_fusion LoRA system
        direct_fusion_lora = direct_fusionLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        # Apply hooks for LoRA forward pass
        direct_fusion_lora.apply_direct_fusion_lora_hooks(model)
        
        print(f"direct_fusion LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, direct_fusion_lora

class direct_fusionLoRAModule:
    """Handles direct_fusion expansion of LoRA matrices."""

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

                    # Create LoRA parameters with the initial rank using Kaiming initialization
                    lora_A_trainable = nn.Parameter(
                        torch.empty(self.current_rank, in_features, device=self.device)
                    )
                    lora_B_trainable = nn.Parameter(
                        torch.zeros(out_features, self.current_rank, device=self.device))

                    # nn.init.kaiming_uniform_(lora_A_trainable, a=math.sqrt(5))
                    # nn.init.kaiming_uniform_(lora_B_trainable, a=math.sqrt(5))
                    nn.init.xavier_uniform_(lora_A_trainable, gain=1.0)
                    nn.init.xavier_uniform_(lora_B_trainable, gain=1.0)

                    # Attach directly to the module. These are the initial trainable parameters.
                    setattr(module, "lora_A_trainable", lora_A_trainable)
                    setattr(module, "lora_B_trainable", lora_B_trainable)

                    # Freeze original weights
                    module.weight.requires_grad = False

        print("Parameters with requires_grad=True:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def expand_rank(self, new_rank):
        """
        Expands the rank of the LoRA matrices by fusing old weights into the base
        model and initializing new LoRA matrices.
        """
        print(f"Expanding LoRA rank from {self.current_rank} to {new_rank}")
        
        # Calculate the scaling factor for the current LoRA matrices
        scaling = self.config.lora_alpha / self.current_rank

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A_trainable") and hasattr(module, "lora_B_trainable"):
                # Fuse the old LoRA weights into the base module's weights
                # W_fused = W_orig + (B_old @ A_old) * scaling
                with torch.no_grad():
                    # Perform the matrix multiplication B @ A
                    lora_update = torch.matmul(module.lora_B_trainable, module.lora_A_trainable)
                    # Apply scaling and add to the base weight matrix
                    module.weight.data += lora_update * scaling

                # Delete the old trainable parameters
                delattr(module, "lora_A_trainable")
                delattr(module, "lora_B_trainable")

                # Initialize new A and B matrices with the new rank
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]

                new_A_trainable = nn.Parameter(torch.empty(new_rank, in_features, device=self.device))
                new_B_trainable = nn.Parameter(torch.zeros(out_features, new_rank, device=self.device))

                # Use Kaiming uniform initialization for the new parameters
                # nn.init.kaiming_uniform_(new_A_trainable, a=math.sqrt(5))
                # nn.init.kaiming_uniform_(new_B_trainable, a=math.sqrt(5))
                nn.init.xavier_uniform_(new_A_trainable, gain=1.0)
                nn.init.xavier_uniform_(new_B_trainable, gain=1.0)

                # Attach the new parameters to the module
                setattr(module, "lora_A_trainable", new_A_trainable)
                setattr(module, "lora_B_trainable", new_B_trainable)
        
        self.current_rank = new_rank
        print(f"Rank successfully expanded to {self.current_rank}")


    def apply_direct_fusion_lora_hooks(self, model):
        """
        Applies forward hooks to integrate LoRA computations.
        This hook now only handles the new trainable LoRA matrices.
        """
        def create_hook():
            def hook(module, input, output):
                # Calculate the scaling factor based on the current rank
                scaling = self.config.lora_alpha / self.current_rank
                
                x = input[0]

                # Compute output from the single trainable matrix
                if hasattr(module, "lora_A_trainable"):
                    lora_output = (x @ module.lora_A_trainable.T @ module.lora_B_trainable.T) * scaling
                    return output + lora_output
                
                return output

            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)): # Apply to relevant layers
                module.register_forward_hook(create_hook())

def train_with_direct_fusion_lora(args, train_loader, val_loader, device, wandb_run):
    """Train model with direct_fusion LoRA rank expansion"""
    
    start_time = time.time()
    
    # Build model with direct_fusion LoRA
    if args.use_pretrained:
        model, direct_fusion_lora = build_model_pretrain(use_direct_fusion_lora=True, num_classes=100, device=device, args=args)
    else:
        model, direct_fusion_lora = build_model(use_direct_fusion_lora=True, num_classes=100, device=device, args=args)
    
    # Optimizer AdamW with fixed learning rate
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    ce = nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"Training WITH direct_fusion LoRA")
    print(f"Starting rank: {args.start_rank}, Max rank: {args.max_rank}")
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
            epochs_since_expansion = 5
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
        
        # Add current validation loss to history
        loss_history.append(val_loss)
        
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
            suffix = 'direct_fusion_lora'
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

    print(f"direct_fusion LoRA training completed in {training_time:.2f} seconds")
    return results

def build_model_pretrain(args, use_direct_fusion_lora: bool = False, num_classes: int = 100, device: torch.device = None):
    # Load pretrained DeiT-Tiny từ Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    direct_fusion_lora = None
    
    if use_direct_fusion_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Create direct_fusion LoRA
        lora_config = LoraConfig(
            r=args.start_rank,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Initialize direct_fusion LoRA system
        direct_fusion_lora = direct_fusionLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        # Apply hooks for LoRA forward pass
        direct_fusion_lora.apply_direct_fusion_lora_hooks(model, direct_fusion_lora)
        
        print(f"direct_fusion LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, direct_fusion_lora
