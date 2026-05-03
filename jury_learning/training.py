from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle
from jury_learning.model import MoralJuryDCN


def resolve_device(cfg: RunConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    if cfg.device == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: RunConfig, bundle: DataBundle) -> MoralJuryDCN:
    fd = bundle.feature_dict
    return MoralJuryDCN(
        num_users=bundle.num_users_for_embedding,
        num_response_features=len(fd["response_fts"]),
        num_group_features=len(fd["group_fts"]),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_cross_layers=cfg.num_cross_layers,
        response_encoder_hidden=cfg.response_encoder_hidden,
    )


def train_moral_model(cfg: RunConfig, model: MoralJuryDCN, bundle: DataBundle, device: torch.device) -> MoralJuryDCN:
    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, reinit=True)
        wandb.watch(model, log="gradients", log_freq=10)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    model.to(device)

    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    freeze_epoch = int(cfg.epochs * cfg.freeze_encoder_epoch_fraction)

    for epoch in range(cfg.epochs):
        if epoch == freeze_epoch:
            print("--- Phase 2: freezing response encoder, lowering LR ---")
            for param in model.response_encoder.parameters():
                param.requires_grad = False
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr_phase2

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            response_fts = batch["response_features"].to(device)
            labels = batch["label"].to(device)
            user_ids = batch["ann_id"].to(device)
            group_fts = batch["group_features"].to(device)

            optimizer.zero_grad()
            outputs = model(response_fts, user_ids, group_fts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                response_fts = batch["response_features"].to(device)
                labels = batch["label"].to(device)
                user_ids = batch["ann_id"].to(device)
                group_fts = batch["group_features"].to(device)

                outputs = model(response_fts, user_ids, group_fts).squeeze()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                val_loss += criterion(outputs, labels).item()

        epoch_train_loss = train_loss / max(len(train_loader), 1)
        epoch_train_acc = 100 * correct / max(total, 1)
        epoch_val_loss = val_loss / max(len(val_loader), 1)
        epoch_val_acc = 100 * val_correct / max(val_total, 1)

        wandb_msg = f" | wandb: {wandb_run.name}" if wandb_run is not None else ""
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%{wandb_msg}"
        )

        if cfg.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "train_accuracy": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

    print("Training complete.")
    if cfg.use_wandb:
        import wandb

        wandb.finish()

    out_path = Path(cfg.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model weights to {out_path.resolve()}")

    return model
