# Auto-generated from Interpretative_jury_learning.ipynb
# File: 03_training.py

# ===== Cell 0 (code) =====
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import sqlite3
import pandas as pd

# ===== Cell 6 (code) =====
class MoralJuryDataset(Dataset):
    def __init__(self, df, feature_dict):
        self.response_features = torch.tensor(df[feature_dict['response_fts']].values, dtype=torch.float32)
        self.group_features = torch.tensor(df[feature_dict['group_fts']].values, dtype=torch.float32)
        self.labels = torch.tensor(df[feature_dict['target']].values, dtype=torch.float32).squeeze()
        self.user_ids = torch.tensor(df['UserID'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'response_features': self.response_features[idx],
            'group_features': self.group_features[idx],
            'label': self.labels[idx],
            'ann_id': self.user_ids[idx]
        }

# ===== Cell 7 (code) =====
class MoralJuryDCN(nn.Module):
    def __init__(self, num_users, num_response_features, num_group_features, embed_dim=32, hidden_dim=128):
        super().__init__()

        # 1. Feature Encoder (Replaces BERT)
        # Maps the 9 ethical dimensions to a dense representation
        self.response_encoder = nn.Sequential(
            nn.Linear(num_response_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 2. Identity Embeddings (User & Group)
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.group_encoder = nn.Sequential(
            nn.Linear(num_group_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Total Input Dim: 64 (Features) + 32 (User) + 32 (Cluster) = 128
        self.input_dim = 64 + embed_dim + embed_dim

        # 3. Cross Network (Parallel Branch)
        self.cross_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.input_dim) for _ in range(3)
        ])

        # 4. Deep Network (Parallel Branch)
        self.deep_layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 5. Combined Output Head
        # Concatenates Cross (input_dim) and Deep (hidden_dim)
        self.output_head = nn.Linear(self.input_dim + hidden_dim, 1)

    def forward(self, response_fts, user_ids, group_fts):
        # Step 1: Encode Scenario Attributes
        x_feat = self.response_encoder(response_fts)

        # Step 2: Create x0 (Base Interaction Vector)
        x0 = torch.cat([
            x_feat,
            self.user_embed(user_ids),
            self.group_encoder(group_fts)
        ], dim=-1)

        # Step 3: Cross Branch (Modeling explicit feature interactions)
        xl = x0
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl

        # Step 4: Deep Branch (Modeling non-linearities)
        xd = self.deep_layers(x0)

        # Step 5: Parallel Combination
        # We concatenate them side-by-side as per the DCN-V2 paper
        combined = torch.cat([xl, xd], dim=-1)

        return torch.sigmoid(self.output_head(combined))

# ===== Cell 8 (code) =====
import wandb # Import wandb for experiment tracking
import torch.nn as nn
import torch.optim as optim

def train_moral_model(model, train_loader, val_loader, device, epochs=10):
    # Initialize wandb run for tracking metrics
    # `reinit=True` allows multiple wandb.init calls in the same script/notebook
    wandb.init(project="moral-jury-model-training", reinit=True)
    wandb.watch(model, log="gradients", log_freq=10) # Log model gradients and parameters

    # 1. Loss and Optimizer
    # Binary Cross Entropy because the label 'Saved' is 0 or 1
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    for epoch in range(epochs):
        # --- PHASE CONTROL ---
        # After 3 epochs, we freeze the feature_encoder to prevent
        # the model from "over-adjusting" the basic scenario rules
        if epoch == int(epochs*(2/3)):
            print("--- Entering Phase 2: Freezing Response Feature Encoder ---")
            for param in model.response_encoder.parameters():
                param.requires_grad = False
            # Lower the learning rate for fine-tuning embeddings
            for g in optimizer.param_groups:
                g['lr'] = 0.0001

        # --- TRAINING ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        val_correct = 0
        val_total = 0

        for batch in train_loader:
            # Extract from our MoralJuryDataset structure
            response_fts = batch['response_features'].to(device)
            labels = batch['label'].to(device)
            user_ids = batch['ann_id'].to(device)
            group_fts = batch['group_features'].to(device)

            # Standard PyTorch Training Step
            optimizer.zero_grad() # Crucial: Clear old gradients

            outputs = model(response_fts, user_ids, group_fts).squeeze()
            loss = criterion(outputs, labels)

            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            # Metrics
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad(): # No gradients needed for evaluation
            for batch in val_loader:
              response_fts = batch['response_features'].to(device)
              labels = batch['label'].to(device)
              user_ids = batch['ann_id'].to(device)
              group_fts = batch['group_features'].to(device)

              outputs = model(response_fts, user_ids, group_fts).squeeze()
              predictions = (outputs > 0.5).float()
              val_correct += (predictions == labels).sum().item()
              val_total += labels.size(0)
              val_loss += criterion(outputs, labels).item()

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        # Print epoch stats
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {epoch_train_loss:.4f} | "
              f"Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2f}% | "
              f"Wandb Run: {wandb.run.name}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    print("Training Complete.")
    wandb.finish() # End the wandb run

# ===== Cell 10 (code) =====
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Cell 15 (code) =====
import torch
import pandas as pd
from torch.utils.data import DataLoader

# --- 1. DATA LOADING & PREP ---
# Load a subset to keep the notebook responsive (e.g., 500k rows)
df_processed, feature_dict = merge_and_process_moral_data_sql("moral_machine.db", 10000000)

df_train, df_val, df_new_users, df_new_scenarios, df_new_groups, df_combined = create_isolated_test_sets(df_processed, feature_dict)

# --- Assign UNSEEN_USER_ID to users not in the training set ---
train_user_ids = set(df_train['UserID'].unique())
UNSEEN_USER_ID = 0 # Now ID 0 is explicitly reserved for unseen users

# Helper function to replace UserIDs not in the training set
def assign_unseen_ids_to_df(df, train_ids, unseen_id):
    # Ensure a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    unseen_mask = ~df_copy['UserID'].isin(train_ids)
    df_copy.loc[unseen_mask, 'UserID'] = unseen_id
    return df_copy

df_val = assign_unseen_ids_to_df(df_val, train_user_ids, UNSEEN_USER_ID)
df_new_users = assign_unseen_ids_to_df(df_new_users, train_user_ids, UNSEEN_USER_ID)
df_new_scenarios = assign_unseen_ids_to_df(df_new_scenarios, train_user_ids, UNSEEN_USER_ID)
df_new_groups = assign_unseen_ids_to_df(df_new_groups, train_user_ids, UNSEEN_USER_ID)
df_combined = assign_unseen_ids_to_df(df_combined, train_user_ids, UNSEEN_USER_ID)

# Now define the number of users for the embedding layer, including the unseen user ID
# If UNSEEN_USER_ID is 0, then the number of unique IDs goes from 0 to max_actual_user_id.
global NUM_USERS_FOR_EMBEDDING
NUM_USERS_FOR_EMBEDDING = df_processed['UserID'].max() + 1

train_ds = MoralJuryDataset(df_train, feature_dict)
val_ds = MoralJuryDataset(df_val, feature_dict)
new_users_ds = MoralJuryDataset(df_new_users, feature_dict)
new_scenarios_ds = MoralJuryDataset(df_new_scenarios, feature_dict)
new_groups_ds = MoralJuryDataset(df_new_groups, feature_dict)
combined_ds = MoralJuryDataset(df_combined, feature_dict)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===== Cell 17 (code) =====
########## SINGLE USER ID FOR UNKNOWN ##############
# --- 2. INITIALIZE & TRAIN ---
# num_users = df_processed['UserID'].nunique() # Old way of calculating num_users
num_users = NUM_USERS_FOR_EMBEDDING # Use the globally defined number of users for embedding, including the 'unseen' user ID
num_response_features = len(feature_dict['response_fts'])
num_group_features = len(feature_dict['group_fts'])
print(f"Users (for embedding layer, including unseen): {num_users}")
EMBED_DIM = 128
HIDDEN_DIM = 512

# Initialize the model
model = MoralJuryDCN(
    num_users=num_users,
    num_response_features=num_response_features,
    num_group_features=num_group_features,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
)
# Run the training loop
train_moral_model(model, train_loader, val_loader, device=DEVICE, epochs=50)
