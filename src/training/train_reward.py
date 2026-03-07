"""
Reward model training loop for RLCR v2 (Phase 1).

Binary classification: does this (diff, comment) get acted on?
Loss: Binary cross-entropy
Optimizer: AdamW with cosine LR schedule
Validation: AUROC, accuracy, precision, recall, F1
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .reward_model import RewardModel

logger = logging.getLogger(__name__)


def evaluate(
    model: RewardModel,
    val_loader,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """
    Evaluate model on validation set.
    
    Returns dict with: loss, auroc, accuracy, precision, recall, f1
    """
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask).squeeze(-1)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Handle edge case: single class in batch
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5  # Default if only one class present
    
    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "auroc": auroc,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    
    return metrics


def train_reward_model(
    model: RewardModel,
    train_loader,
    val_loader,
    config: dict,
    project_root: str,
    max_steps: Optional[int] = None,
) -> dict:
    """
    Train the reward model.
    
    Full training loop with:
    - BCE loss
    - AdamW optimizer with cosine schedule
    - Mixed precision (bf16/fp16)
    - Periodic validation with AUROC tracking
    - Best checkpoint saving
    - Wandb logging (optional)
    
    Returns:
        Final validation metrics dict
    """
    rm_config = config["reward_model"]
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected! Training will be slow.")
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer: only train LoRA params + classification head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=rm_config["learning_rate"],
        weight_decay=rm_config["weight_decay"],
    )
    
    # Scheduler
    num_epochs = rm_config["num_epochs"]
    total_steps = len(train_loader) * num_epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    
    warmup_steps = int(total_steps * rm_config["warmup_ratio"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    
    # Mixed precision
    use_amp = rm_config.get("bf16", False) or rm_config.get("fp16", False)
    amp_dtype = torch.bfloat16 if rm_config.get("bf16", False) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda" and amp_dtype == torch.float16))
    
    # Gradient accumulation
    grad_accum_steps = rm_config["gradient_accumulation_steps"]
    
    # Logging
    log_steps = rm_config["log_steps"]
    eval_steps = rm_config["eval_steps"]
    
    # Wandb
    use_wandb = rm_config.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=rm_config.get("wandb_project", "rlcr-v2"), config=rm_config)
        except ImportError:
            logger.warning("wandb not installed, skipping")
            use_wandb = False
    
    # Output directory
    output_dir = Path(project_root) / rm_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    
    # Training state
    best_auroc = 0.0
    global_step = 0
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("REWARD MODEL TRAINING")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {rm_config['batch_size']}")
    logger.info(f"  Gradient accumulation: {grad_accum_steps}")
    logger.info(f"  Effective batch size: {rm_config['batch_size'] * grad_accum_steps}")
    logger.info(f"  Learning rate: {rm_config['learning_rate']}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Eval every: {eval_steps} steps")
    logger.info(f"  Mixed precision: {amp_dtype if use_amp else 'disabled'}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if max_steps and global_step >= max_steps:
                logger.info(f"Reached max_steps={max_steps}, stopping")
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids, attention_mask).squeeze(-1)
                loss = criterion(logits, labels)
                loss = loss / grad_accum_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            # Step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, rm_config["max_grad_norm"]
                )
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # LR warmup
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = rm_config["learning_rate"] * lr_scale
                else:
                    scheduler.step()
                
                global_step += 1
            
            epoch_loss += loss.item() * grad_accum_steps
            epoch_steps += 1
            
            # Live TQDM update
            avg_loss = epoch_loss / epoch_steps
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "step": global_step,
            })
            
            # Logging
            if global_step % log_steps == 0 and global_step > 0:
                elapsed = time.time() - start_time
                
                if use_wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/step": global_step,
                        "train/elapsed_min": elapsed / 60,
                    }, step=global_step)
            
            # Validation
            if global_step % eval_steps == 0 and global_step > 0:
                val_metrics = evaluate(model, val_loader, device, criterion)
                
                logger.info(
                    f"\n[Step {global_step}] Val Metrics:\n"
                    f"  Loss:      {val_metrics['loss']:.4f}\n"
                    f"  AUROC:     {val_metrics['auroc']:.4f}\n"
                    f"  Accuracy:  {val_metrics['accuracy']:.4f}\n"
                    f"  Precision: {val_metrics['precision']:.4f}\n"
                    f"  Recall:    {val_metrics['recall']:.4f}\n"
                    f"  F1:        {val_metrics['f1']:.4f}"
                )
                
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
                
                # AUROC warnings
                auroc = val_metrics["auroc"]
                if auroc < rm_config["auroc_warning"]:
                    logger.warning(
                        f"⚠️  AUROC {auroc:.4f} < {rm_config['auroc_warning']} — "
                        f"possible data issue. Consider debugging before Phase 2."
                    )
                
                # Save best
                if auroc > best_auroc:
                    best_auroc = auroc
                    model.save_checkpoint(str(best_dir))
                    logger.info(f"  ✅ New best AUROC: {best_auroc:.4f} — checkpoint saved")
                
                model.train()
        
        if max_steps and global_step >= max_steps:
            break
        
        # End of epoch validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        logger.info(
            f"\n{'='*40}\n"
            f"EPOCH {epoch+1}/{num_epochs} COMPLETE\n"
            f"  Train Loss:  {epoch_loss / max(epoch_steps, 1):.4f}\n"
            f"  Val AUROC:   {val_metrics['auroc']:.4f}\n"
            f"  Val F1:      {val_metrics['f1']:.4f}\n"
            f"  Best AUROC:  {best_auroc:.4f}\n"
            f"{'='*40}"
        )
        
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            model.save_checkpoint(str(best_dir))
            logger.info(f"  ✅ New best AUROC: {best_auroc:.4f}")
    
    # Final summary
    elapsed = time.time() - start_time
    logger.info(
        f"\n{'='*60}\n"
        f"TRAINING COMPLETE\n"
        f"  Total steps: {global_step}\n"
        f"  Total time:  {elapsed/60:.1f} min\n"
        f"  Best AUROC:  {best_auroc:.4f}\n"
        f"  Checkpoint:  {best_dir}\n"
        f"{'='*60}"
    )
    
    # Final AUROC check
    if best_auroc >= rm_config["auroc_target"]:
        logger.info(f"✅ AUROC {best_auroc:.4f} ≥ {rm_config['auroc_target']} — ready for Phase 2!")
    else:
        logger.warning(
            f"⚠️  AUROC {best_auroc:.4f} < {rm_config['auroc_target']} — "
            f"reward model may need more data or training. "
            f"Debug before proceeding to Phase 2."
        )
    
    if use_wandb:
        wandb.finish()
    
    return val_metrics
