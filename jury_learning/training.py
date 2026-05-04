from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, MoralJuryDataset
from jury_learning.model import MoralJuryDCN, MoralJuryDCNBaseline
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Per-epoch training and validation metrics (accuracies are fractions in [0, 1])."""

    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    new_users_accuracy: list[float] = field(default_factory=list)
    new_scenarios_accuracy: list[float] = field(default_factory=list)
    new_groups_accuracy: list[float] = field(default_factory=list)
    combined_accuracy: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_loss": self.train_loss,
                "train_accuracy": self.train_accuracy,
                "val_loss": self.val_loss,
                "val_accuracy": self.val_accuracy,
                "new_users_accuracy": self.new_users_accuracy,
                "new_scenarios_accuracy": self.new_scenarios_accuracy,
                "new_groups_accuracy": self.new_groups_accuracy,
                "combined_accuracy": self.combined_accuracy,
                "learning_rate": self.learning_rate,
            }
        )

    def last(self) -> dict[str, float]:
        """Metrics from the final epoch."""
        if not self.epoch:
            return {}
        i = -1
        return {
            "epoch": int(self.epoch[i]),
            "train_loss": self.train_loss[i],
            "train_accuracy": self.train_accuracy[i],
            "val_loss": self.val_loss[i],
            "val_accuracy": self.val_accuracy[i],
            "new_users_accuracy": self.new_users_accuracy[i],
            "new_scenarios_accuracy": self.new_scenarios_accuracy[i],
            "new_groups_accuracy": self.new_groups_accuracy[i],
            "combined_accuracy": self.combined_accuracy[i],
            "learning_rate": self.learning_rate[i],
        }


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


def build_model(cfg: RunConfig, bundle: DataBundle) -> MoralJuryDCN | MoralJuryDCNBaseline:
    fd = bundle.feature_dict
    if cfg.use_user_embedding:
        return MoralJuryDCN(
            num_users=bundle.num_users_for_embedding,
            num_response_features=len(fd["response_fts"]),
            num_group_features=len(fd["group_fts"]),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_cross_layers=cfg.num_cross_layers,
            response_encoder_hidden=cfg.response_encoder_hidden,
        )
    return MoralJuryDCNBaseline(
        num_response_features=len(fd["response_fts"]),
        num_group_features=len(fd["group_fts"]),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_cross_layers=cfg.num_cross_layers,
        response_encoder_hidden=cfg.response_encoder_hidden,
    )


def _maybe_tqdm(iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable, **kwargs)


def train_moral_model(
    cfg: RunConfig,
    model: MoralJuryDCN,
    bundle: DataBundle,
    device: torch.device,
) -> tuple[MoralJuryDCN, TrainingHistory]:
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

    # Build eval loaders for held-out splits (created once, reused every epoch)
    _eval_splits = {
        "new_users":    bundle.df_new_users,
        "new_scenarios": bundle.df_new_scenarios,
        "new_groups":   bundle.df_new_groups,
        "combined":     bundle.df_combined,
    }
    _eval_loaders = {
        name: DataLoader(
            MoralJuryDataset(df, bundle.feature_dict),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
        )
        for name, df in _eval_splits.items()
    }

    freeze_epoch = int(cfg.epochs * cfg.freeze_encoder_epoch_fraction)
    history = TrainingHistory()

    for epoch in range(cfg.epochs):
        if epoch == freeze_epoch:
            if cfg.verbose:
                print("Phase 2: freezing response encoder, lowering LR.")
            for param in model.response_encoder.parameters():
                param.requires_grad = False
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr_phase2

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        batches = _maybe_tqdm(
            train_loader,
            enabled=cfg.show_progress_bar,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        for batch in batches:
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

        n_train = max(len(train_loader), 1)
        n_val = max(len(val_loader), 1)
        epoch_train_loss = train_loss / n_train
        epoch_train_acc = correct / max(total, 1)
        epoch_val_loss = val_loss / n_val
        epoch_val_acc = val_correct / max(val_total, 1)
        lr = optimizer.param_groups[0]["lr"]

        # Evaluate on all held-out splits
        split_accs: dict[str, float] = {}
        with torch.no_grad():
            for name, loader in _eval_loaders.items():
                s_correct = 0
                s_total = 0
                for batch in loader:
                    response_fts = batch["response_features"].to(device)
                    labels = batch["label"].to(device)
                    user_ids = batch["ann_id"].to(device)
                    group_fts = batch["group_features"].to(device)
                    outputs = model(response_fts, user_ids, group_fts).squeeze()
                    s_correct += ((outputs > 0.5).float() == labels).sum().item()
                    s_total += labels.size(0)
                split_accs[name] = s_correct / max(s_total, 1)

        history.epoch.append(epoch + 1)
        history.train_loss.append(epoch_train_loss)
        history.train_accuracy.append(epoch_train_acc)
        history.val_loss.append(epoch_val_loss)
        history.val_accuracy.append(epoch_val_acc)
        history.new_users_accuracy.append(split_accs["new_users"])
        history.new_scenarios_accuracy.append(split_accs["new_scenarios"])
        history.new_groups_accuracy.append(split_accs["new_groups"])
        history.combined_accuracy.append(split_accs["combined"])
        history.learning_rate.append(lr)

        if cfg.verbose:
            wandb_msg = f" | wandb={wandb_run.name}" if wandb_run is not None else ""
            print(
                f"Epoch {epoch + 1}/{cfg.epochs} | "
                f"train_loss={epoch_train_loss:.4f} train_acc={epoch_train_acc:.4f} | "
                f"val={epoch_val_acc:.4f} "
                f"new_users={split_accs['new_users']:.4f} "
                f"new_scenarios={split_accs['new_scenarios']:.4f} "
                f"new_groups={split_accs['new_groups']:.4f} "
                f"combined={split_accs['combined']:.4f}{wandb_msg}"
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
                    "new_users_accuracy": split_accs["new_users"],
                    "new_scenarios_accuracy": split_accs["new_scenarios"],
                    "new_groups_accuracy": split_accs["new_groups"],
                    "combined_accuracy": split_accs["combined"],
                    "learning_rate": lr,
                }
            )

    if cfg.verbose:
        print("Training complete.")
    if cfg.use_wandb:
        import wandb

        wandb.finish()

    out_path = Path(cfg.model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    if cfg.verbose:
        print(f"Saved weights to {out_path.resolve()}")

    return model, history
