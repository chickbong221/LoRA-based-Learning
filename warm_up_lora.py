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

def build_model(args, use_warm_up_lora: bool = False, num_classes: int = 100, device: torch.device = None):
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
    model = ViTForImageClassification(config)
    warm_up_lora = None

    if use_warm_up_lora:
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True

        lora_config = LoraConfig(
            r=args.start_rank,
            lora_alpha=int(2 * args.start_rank),
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        warm_up_lora = warm_upLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        warm_up_lora.apply_warm_up_lora_hooks(model)
        
        print(f"warm_up LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, warm_up_lora

class warm_upLoRAModule:
    """Handles warm-up LoRA initialization and fusion."""

    def __init__(self, model, config: LoraConfig, target_modules, device):
        self.model = model
        self.config = config
        self.target_modules = target_modules
        self.device = device
        self.hooks = []

        self._initialize_lora_modules()

    def _initialize_lora_modules(self):
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if hasattr(module, "weight"):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]

                    lora_A_trainable = nn.Parameter(
                        torch.empty(self.config.r, in_features, device=self.device)
                    )
                    lora_B_trainable = nn.Parameter(
                        torch.zeros(out_features, self.config.r, device=self.device)
                    )

                    nn.init.kaiming_uniform_(lora_A_trainable, a=math.sqrt(5))

                    setattr(module, "lora_A_trainable", lora_A_trainable)
                    setattr(module, "lora_B_trainable", lora_B_trainable)

                    module.weight.requires_grad = False
    
    def fuse_and_remove_lora(self):
        """
        Fuses the current LoRA weights into the base model and deletes
        the LoRA modules and hooks. After this, the entire model is set
        to trainable for full fine-tuning.
        """
        print("Fusing LoRA weights and transitioning to full fine-tuning.")
        scaling = self.config.lora_alpha / self.config.r

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A_trainable") and hasattr(module, "lora_B_trainable"):
                with torch.no_grad():
                    lora_update = torch.matmul(module.lora_B_trainable, module.lora_A_trainable)
                    module.weight.data += lora_update * scaling

                delattr(module, "lora_A_trainable")
                delattr(module, "lora_B_trainable")

        self.remove_lora_hooks()

        for param in self.model.parameters():
            param.requires_grad = True

        print("LoRA fused. Model is now in full fine-tuning mode (all params trainable).")
        
    def remove_lora_hooks(self):
        """Removes all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("All LoRA hooks removed.")

    def apply_warm_up_lora_hooks(self, model):
        """Applies forward hooks to integrate LoRA computations."""
        def create_hook():
            def hook(module, input, output):
                scaling = self.config.lora_alpha / self.config.r
                x = input[0]
                if hasattr(module, "lora_A_trainable"):
                    temp = torch.nn.functional.linear(x, module.lora_A_trainable)
                    lora_output = torch.nn.functional.linear(temp, module.lora_B_trainable) * scaling
                    return output + lora_output
                return output
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook_handle = module.register_forward_hook(create_hook())
                self.hooks.append(hook_handle)


def train_with_warm_up_lora(args, train_loader, val_loader, device, wandb_run):
    """Train model with warm-up LoRA, then transition to full fine-tuning."""
    
    start_time = time.time()
    
    # Build model with warm-up LoRA
    if args.use_pretrained:
        # Assuming build_model_pretrain is a function that loads a pretrained model
        model, warm_up_lora = build_model_pretrain(use_warm_up_lora=True, num_classes=100, device=device, args=args)
    else:
        model, warm_up_lora = build_model(use_warm_up_lora=True, num_classes=100, device=device, args=args)

    # Optimizer for the LoRA and classifier parameters
    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_parameters, lr=args.lr)

    ce = nn.CrossEntropyLoss()
    
    print(f"\n{'='*50}")
    print(f"Training with warm-up LoRA")
    print(f"Starting rank: {args.start_rank}")
    print(f"Will transition to full fine-tuning on loss plateau.")
    print(f"{'='*50}")

    best_val_acc = 0.0
    results = {
        'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': [],
        'best_val_acc': 0.0, 'time_history': []
    }
    
    loss_history = []
    is_full_finetuning = False

    min_improvement = 0.001
    patience = 8
    
    for epoch in range(args.epochs):
        # Check if we need to transition to full fine-tuning based on loss plateau
        if not is_full_finetuning and (epoch > 0 and len(loss_history) >= patience):
            recent_losses = loss_history[-patience:-1]
            improvement = min(recent_losses) - loss_history[-1]
            loss_plateaued = improvement < min_improvement

            if loss_plateaued:
                # 1. Fuse LoRA weights into the base model and remove LoRA modules
                warm_up_lora.fuse_and_remove_lora()
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"{trainable}/{total} trainable ({100*trainable/total:.2f}%)")
                
                # 2. Re-initialize optimizer to include all model parameters
                optimizer = optim.AdamW(model.parameters(), lr=args.lr)
                
                # 3. Update state
                is_full_finetuning = True
                
                print(f"Transitioned to full fine-tuning at epoch {epoch+1}.")

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
        
        # Add current validation loss to history if still in LoRA phase
        if not is_full_finetuning:
            loss_history.append(val_loss)
        
        elapsed_time = time.time() - start_time
        results['time_history'].append(elapsed_time)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        results['train_losses'].append(train_loss)
        results['train_accs'].append(train_acc)
        results['val_losses'].append(val_loss)
        results['val_accs'].append(val_acc)
        results['best_val_acc'] = best_val_acc
        
        print(f"Epoch {epoch+1}/{args.epochs} (Mode: {'Full-Finetuning' if is_full_finetuning else f'LoRA r={args.start_rank}'}) - "
              f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:.2f}% | "
              f"Time {elapsed_time:.2f}s")
        
        if wandb_run:
            suffix = 'warm_up_lora'
            log_data = {
                f"{suffix}/epoch": epoch,
                f"{suffix}/train_loss": train_loss,
                f"{suffix}/train_acc": train_acc,
                f"{suffix}/val_loss": val_loss,
                f"{suffix}/val_acc": val_acc,
                f"{suffix}/elapsed_time": elapsed_time,
            }
            wandb_run.log(log_data)

    end_time = time.time()
    training_time = end_time - start_time
    results['training_time'] = training_time

    print(f"Training completed in {training_time:.2f} seconds")
    return results

def build_model_pretrain(args, use_warm_up_lora: bool = False, num_classes: int = 100, device: torch.device = None):
    # Load pretrained DeiT-Tiny từ Hugging Face
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    warm_up_lora = None
    
    if use_warm_up_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Keep classifier trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Create warm_up LoRA
        lora_config = LoraConfig(
            r=args.start_rank,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        
        # Initialize warm_up LoRA system
        warm_up_lora = warm_upLoRAModule(
            model, lora_config, args.lora_target_modules, device
        )
        
        # Apply hooks for LoRA forward pass
        warm_up_lora.apply_warm_up_lora_hooks(model, warm_up_lora)
        
        print(f"warm_up LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, warm_up_lora
