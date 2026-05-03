# Auto-generated from Interpretative_jury_learning.ipynb
# File: 02_model_architecture.py

# ===== Cell 0 (code) =====
import torch
import torch.nn as nn
import torch.optim as optim
from transfoxrmers import AutoModel, AutoTokenizer
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
