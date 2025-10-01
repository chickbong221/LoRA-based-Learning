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
from progressive_lora import train_with_progressive_lora
from full_finetuning import train_single_run
from direct_fusion_lora import train_with_direct_fusion_lora
from warm_up_lora import train_with_warm_up_lora
from progressive_lora_unfreeze import train_with_progressive_lora_unfreeze
from utils import *

import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--start_rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--max_rank", type=int, default=196, help="Maximum LoRA rank")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()
    args.lora_target_modules = ("query", "key", "value", "output.dense")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb_run = None
    if args.wandb:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb_run = wandb.init(
            project="ViT_LoRA_Comparison",
            entity="letuanhf-hanoi-university-of-science-and-technology",
            config=vars(args),
            name=f"lr_{args.lr}_pretrained" if args.use_pretrained else f"lr_{args.lr}_from_scratch",
        )

        # Định nghĩa metric riêng cho từng phương pháp
        wandb.define_metric("full_ft/*", step_metric="full_ft/epoch")
        wandb.define_metric("progressive_lora/*", step_metric="progressive_lora/epoch")
        wandb.define_metric("direct_fusion_lora/*", step_metric="direct_fusion_lora/epoch")
        wandb.define_metric("warm_up_lora/*", step_metric="warm_up_lora/epoch")

    # Data (create once, use for all runs)
    print("Loading datasets...")
    train_set, val_set = build_cifar100_datasets(img_size=224)
    # train_set, val_set = build_flower_datasets(img_size=224)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Run experiments
    print("\n" + "="*60)
    print("STARTING EXPERIMENTS")
    print("="*60)
    
    # Run 1: Full Fine-tuning (without LoRA)
    results_full_ft = train_single_run(
        use_lora=False, 
        args=args, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        wandb_run=wandb_run
    )
    
    # # Run 2: Progressive LoRA
    # results_progressive = train_with_progressive_lora(
    #     args=args, 
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     device=device, 
    #     wandb_run=wandb_run
    # )

    # Run 3: Direct Fusion LoRA
    # results_direct_fusion = train_with_direct_fusion_lora(
    #     args=args, 
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     device=device, 
    #     wandb_run=wandb_run
    # )

    # Run 4: Warm Up LoRA
    # results_warm_up = train_with_warm_up_lora(
    #     args=args, 
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     device=device, 
    #     wandb_run=wandb_run
    # )

    # Run 5: Unfreeze Progressive LoRA
    # results_full_ft = train_with_progressive_lora_unfreeze( 
    #     args=args, 
    #     train_loader=train_loader, 
    #     val_loader=val_loader, 
    #     device=device, 
    #     wandb_run=wandb_run
    # )
    
    # Final comparison
    # print(f"\n{'='*60}")
    # print("FINAL RESULTS COMPARISON")
    # print(f"{'='*60}")
    # print(f"Full Fine-tuning       - Best Val Acc: {results_full_ft['best_val_acc']*100:.2f}% | Time: {results_full_ft['training_time']:.2f}s")
    # print(f"Progressive LoRA       - Best Val Acc: {results_progressive['best_val_acc']*100:.2f}% | Time: {results_progressive['training_time']:.2f}s")
    # print(f"\nImprovement over Full Fine-tuning: {(results_progressive['best_val_acc'] - results_full_ft['best_val_acc'])*100:.2f}%")

if __name__ == "__main__":
    main()