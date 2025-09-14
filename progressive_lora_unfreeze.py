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
        
        # Initialize progressive LoRA system with the new max_rank argument
        progressive_lora = ProgressiveLoRAModule(
            model, lora_config, args.lora_target_modules, device, max_rank=args.max_rank
        )
        
        # Apply hooks for LoRA forward pass
        progressive_lora.apply_progressive_lora_hooks(model)
        
        print(f"Progressive LoRA initialized with rank {args.start_rank}")
    
    if device:
        model = model.to(device)
    
    return model, progressive_lora

class ProgressiveLoRAModule:
    def __init__(self, model, config, target_modules, device, max_rank):
        self.model = model
        self.config = config
        self.target_modules = target_modules
        self.device = device
        self.current_rank = config.r
        self.max_rank = max_rank

        self._initialize_lora_modules_full()

    def _initialize_lora_modules_full(self):
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules) and hasattr(module, "weight"):
                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]

                # Chia 3 phần
                lora_A_training = nn.Parameter(
                    torch.empty(self.current_rank, in_features, device=self.device)
                )
                lora_B_training = nn.Parameter(
                    torch.zeros(out_features, self.current_rank, device=self.device)
                )

                # nn.init.kaiming_uniform_(lora_A_training, a=math.sqrt(5))
                nn.init.xavier_uniform_(lora_A_training, gain=1.0)
                nn.init.xavier_uniform_(lora_B_training, gain=1.0)

                remaining = self.max_rank - self.current_rank
                lora_A_not_trained = nn.Parameter(
                    torch.empty(remaining, in_features, device=self.device),
                    requires_grad=False
                )
                lora_B_not_trained = nn.Parameter(
                    torch.zeros(out_features, remaining, device=self.device),
                    requires_grad=False
                )

                # nn.init.kaiming_uniform_(lora_A_not_trained, a=math.sqrt(5))
                nn.init.xavier_uniform_(lora_A_not_trained, gain=1.0)
                nn.init.xavier_uniform_(lora_B_not_trained, gain=1.0)

                lora_A_trained = nn.Parameter(torch.empty(0, in_features, device=self.device),
                                              requires_grad=False)
                lora_B_trained = nn.Parameter(torch.empty(out_features, 0, device=self.device),
                                              requires_grad=False)

                # Attach
                module.lora_A_training = lora_A_training
                module.lora_B_training = lora_B_training
                module.lora_A_trained = lora_A_trained
                module.lora_B_trained = lora_B_trained
                module.lora_A_not_trained = lora_A_not_trained
                module.lora_B_not_trained = lora_B_not_trained

                module.weight.requires_grad = False

        print("Parameters with requires_grad=True:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

    def expand_rank(self, new_rank):
        if new_rank >= self.max_rank:
            print(f"Cannot expand to {new_rank}, max rank is {self.max_rank}.")
            return

        print(f"Expanding rank {self.current_rank} → {new_rank}")
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A_training"):
                # Hợp nhất training → trained
                module.lora_A_trained = nn.Parameter(
                    torch.cat([module.lora_A_trained.data,
                               module.lora_A_training.data], dim=0),
                    requires_grad=False
                )
                module.lora_B_trained = nn.Parameter(
                    torch.cat([module.lora_B_trained.data,
                               module.lora_B_training.data], dim=1),
                    requires_grad=False
                )

                # Lấy slice mới từ not_trained
                add_rank = new_rank - self.current_rank
                new_A_training = module.lora_A_not_trained.data[:add_rank, :]
                new_B_training = module.lora_B_not_trained.data[:, :add_rank]

                # Update lại not_trained
                module.lora_A_not_trained = nn.Parameter(
                    module.lora_A_not_trained.data[add_rank:, :],
                    requires_grad=False
                )
                module.lora_B_not_trained = nn.Parameter(
                    module.lora_B_not_trained.data[:, add_rank:],
                    requires_grad=False
                )

                # Set training
                module.lora_A_training = nn.Parameter(new_A_training, requires_grad=True)
                module.lora_B_training = nn.Parameter(new_B_training, requires_grad=True)

        self.current_rank = new_rank

    def apply_progressive_lora_hooks(self, model):
        def create_hook():
            def hook(module, input, output):
                alpha = self.config.lora_alpha
                scaling = alpha / self.max_rank
                x = input[0]

                # Ghép 3 phần theo trật tự [trained | training | not_trained]
                lora_A = torch.cat(
                    [module.lora_A_trained, module.lora_A_training, module.lora_A_not_trained],
                    dim=0
                )
                lora_B = torch.cat(
                    [module.lora_B_trained, module.lora_B_training, module.lora_B_not_trained],
                    dim=1
                )

                lora_output = (x @ lora_A.T @ lora_B.T) * scaling
                return output + lora_output

            return hook

        for name, module in model.named_modules():
            if hasattr(module, "lora_A_training"):
                module.register_forward_hook(create_hook())

def train_with_progressive_lora_unfreeze(args, train_loader, val_loader, device, wandb_run):
    """Train model with progressive LoRA rank expansion"""

    start_time = time.time()

    # Build model with progressive LoRA
    if args.use_pretrained:
        # Note: You'll need to update build_model_pretrain to also pass max_rank
        model, progressive_lora = build_model_pretrain(
            use_progressive_lora=True, 
            num_classes=100, 
            device=device, 
            args=args
        )
    else:
        # Pass the max_rank argument to the ProgressiveLoRAModule
        model, progressive_lora = build_model(
            use_progressive_lora=True, 
            num_classes=100, 
            device=device, 
            args=args,
        )

    # Optimizer AdamW with fixed learning rate. No need to re-initialize later.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    ce = nn.CrossEntropyLoss()

    print(f"\n{'='*50}")
    print(f"Training WITH Progressive LoRA")
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
        'time_history': []  # <-- store accumulated time
    }

    current_rank = args.start_rank
    loss_history = []

    min_improvement = 0.001  # Minimum loss reduction threshold
    patience_epochs = 5
    last_expansion_epoch = 0
    patience = 0
    
    for epoch in range(args.epochs):
        # Check if we need to expand rank based on loss plateau
        print_count = 0
        should_expand = False

        if (epoch > 0 and current_rank < args.max_rank and len(loss_history) >= 5):
            # Check if loss has plateaued
            improvement = min(loss_history[:-1]) - loss_history[-1]
            
            print(f"Epoch {epoch}: Recent loss improvement: {improvement:.6f}")
            loss_plateaued = improvement < min_improvement

            if loss_plateaued:
                patience += 1
            else:
                patience = 0

            should_expand = loss_plateaued and patience >= patience_epochs

        # Expand rank if conditions met
        if should_expand:
            new_rank = min(current_rank * 2, args.max_rank)
            
            # Call the expand_rank method on the progressive_lora object
            progressive_lora.expand_rank(new_rank)

            args.lr = args.lr * 0.5
            args.lr = max(args.lr, 1e-5)  # don't go below 1e-5
            min_improvement = max(min_improvement * 0.3, 0.00005)
            print(f"Learning rate decayed to {args.lr}")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Rank expanded to {new_rank} at epoch {epoch+1} (improvement: {improvement:.6f})")
            
            current_rank = new_rank
            last_expansion_epoch = epoch
            patience = 0  

        model.train()
        running_loss, running_acc, total = 0.0, 0, 0

        # print("Parameters with requires_grad=True:")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = ce(logits, labels)

            # check gradient wrt output (logits)
            # print_count += 1
            # if print_count <= 10:
            #     logits_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            #     print("||dL/dy|| (logits grad norm):", logits_grad.norm().item())

            loss.backward()
            optimizer.step()

            layer_count = 0
            print_count += 1
            if print_count <= 3:
                # print("Params updated in this step:")
                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # kiểm tra param có trong optimizer không
                        in_optimizer = any(p is param for group in optimizer.param_groups for p in group['params'])
                        if in_optimizer:
                            # accumulate norm^2
                            param_norm = param.grad.data.norm(2).item()
                            total_norm += param_norm ** 2

                            # if layer_count <= 5:
                            #     print(f"  {name:50s} | grad_norm={param_norm:.6f}")
                            #     layer_count += 1

                total_norm = total_norm ** 0.5
                print(f"Total grad norm: {total_norm:.6f}")

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)
            total += bs
            running_loss += loss.item() * bs
            running_acc += correct

            # break  # chỉ in thử 1 batch, tránh quá dài

        train_loss = running_loss / total
        train_acc = running_acc / total
        val_loss, val_acc = evaluate(model, val_loader, device)

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
