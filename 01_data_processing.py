# Auto-generated from Interpretative_jury_learning.ipynb
# File: 01_data_processing.py

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


# ===== Cell 1 (code) =====
def take_scenario_data_sql(db_path, output_file_path):
    conn = sqlite3.connect(db_path)

    character_cols = [
        'PedPed', 'Barrier', 'NumberOfCharacters', 'CrossingSignal', 'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman', 'Boy',
        'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal', 'MaleExecutive',
        'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete', 'FemaleDoctor',
        'MaleDoctor', 'Dog', 'Cat'
    ]

    sql_query = f"""
    SELECT
        {', '.join([f'r0.{c} AS Stay_{c}' for c in character_cols])},
        {', '.join([f'r1.{c} AS Swerve_{c}' for c in character_cols])}
    FROM survey s
    JOIN responses r0 ON s.ResponseID = r0.ResponseID AND r0.Intervention = 0
    JOIN responses r1 ON s.ResponseID = r1.ResponseID AND r1.Intervention = 1
    """

    df_final = pd.read_sql_query(sql_query, conn)

    # Identify columns that describe the scenario (all character columns)
    scenario_description_cols = [col for col in df_final.columns if col.startswith(('Stay_', 'Swerve_'))]

    # Drop duplicates based on the scenario description columns
    # We keep the first occurrence. ResponseID is not included in the deduplication criteria.
    df_unique_scenarios = df_final.drop_duplicates(subset=scenario_description_cols).reset_index(drop=True)

    # Save the DataFrame with unique scenarios to a CSV file
    df_unique_scenarios.to_csv(output_file_path, index=False)
    print(f"Unique character scenarios saved to {output_file_path}")

    print(f"Successfully processed and saved {len(df_unique_scenarios)} unique paired scenarios with only character features.")
    return df_unique_scenarios

# ===== Cell 2 (code) =====

# ===== Cell 3 (code) =====
unique_scenarios=take_scenario_data_sql("moral_machine.db", "unique_scenarios.csv")
print(len(unique_scenarios))
print(unique_scenarios)

# ===== Cell 4 (code) =====
def merge_and_process_moral_data_sql(db_path, subset_size):
    conn = sqlite3.connect(db_path)

    character_cols = [
        'NumberOfCharacters', 'Pedped', 'Barrier', 'CrossingSignal', 'Man', 'Woman', 'Pregnant', 'Stroller', 'OldMan', 'OldWoman', 'Boy',
        'Girl', 'Homeless', 'LargeWoman', 'LargeMan', 'Criminal', 'MaleExecutive',
        'FemaleExecutive', 'FemaleAthlete', 'MaleAthlete', 'FemaleDoctor',
        'MaleDoctor', 'Dog', 'Cat'
    ]

    # 1. SQL Query to get paired rows (Stay & Swerve) for the first X ResponseIDs
    # This query joins the table to itself to create one wide row immediately
    print(f"Querying and pairing {subset_size} scenarios...")

    # We select from survey first to get our subset, then join the two halves of the response
    sql_query = f"""
    SELECT
        s.ResponseID,
        s.UserID,
        s.Review_age, s.Review_education, s.Review_gender,
        s.Review_income, s.Review_political, s.Review_religious, s.UserCountry3,
        {', '.join([f'r0.{c} AS Stay_{c}' for c in character_cols])},
        r1.Saved AS Swerve_Saved,
        {', '.join([f'r1.{c} AS Swerve_{c}' for c in character_cols])}
    FROM (SELECT * FROM survey LIMIT {subset_size}) s
    JOIN responses r0 ON s.ResponseID = r0.ResponseID AND r0.Intervention = 0
    JOIN responses r1 ON s.ResponseID = r1.ResponseID AND r1.Intervention = 1
    """

    df_final = pd.read_sql_query(sql_query, conn)

    # Global UserID Encoding
    user_encoder = LabelEncoder()
    # 0 reserved for unknown users
    df_final['UserID'] = user_encoder.fit_transform(df_final['UserID']) + 1

    # 2. Create Target Label
    # If the characters in the Swerve intervention (r1) were saved, Decision_Swerve = 1
    df_final['Decision_Swerve'] = df_final['Swerve_Saved'].astype(int)

    # --- Feature Engineering & Scaling ---

    # A. Scale Age
    df_final['Review_age'] = pd.to_numeric(df_final['Review_age'], errors='coerce')
    df_final['Review_age'] = df_final['Review_age'].fillna(df_final['Review_age'].median())
    df_final['Review_age'] = df_final['Review_age'].clip(18, 75)
    df_final['Review_age'] = (df_final['Review_age'] - 18) / (75 - 18)

    # B. Scale Education
    edu_map = {'underHigh': 0.1, 'high': 0.3, 'vocational': 0.4, 'college': 0.6,
               'bachelor': 0.8, 'graduate': 1.0, 'other': 0.5, 'default': 0.5}
    df_final['Review_education'] = df_final['Review_education'].map(edu_map).fillna(0.5)

    # C. Scale Income
    income_map = {'under5000': 0.1, '5000': 0.2, '10000': 0.3, '15000': 0.4, '25000': 0.5,
                  '35000': 0.6, '50000': 0.7, '80000': 0.8, 'above100000': 1.0, 'default': 0.5}
    df_final['Review_income'] = df_final['Review_income'].map(income_map).fillna(0.5)
    df_final['Review_political'] = pd.to_numeric(df_final['Review_political'], errors='coerce').fillna(0.5)
    df_final['Review_religious'] = pd.to_numeric(df_final['Review_religious'], errors='coerce').fillna(0.5)

    # D. Get Dummies
    categorical_cols = ['Review_gender', 'UserCountry3']
    df_final = pd.get_dummies(df_final, columns=categorical_cols, prefix=['Gen', 'Cnt'])

    # 3. Define Feature Groups
    dummy_prefixes = ('Gen_', 'Cnt_')
    group_fts = [
        'Review_age', 'Review_education', 'Review_income',
        'Review_political', 'Review_religious'
    ] + [col for col in df_final.columns if col.startswith(dummy_prefixes)]

    response_fts = [f'Stay_{c}' for c in character_cols] + [f'Swerve_{c}' for c in character_cols]

    feature_dict = {
        'user_fts': ['UserID'],
        'group_fts': group_fts,
        'response_fts': response_fts,
        'target': ['Decision_Swerve']
    }

    # Final cleanup: ensure all group features are numeric for PyTorch
    df_final[group_fts] = df_final[group_fts].astype(float)

    print(f"Successfully processed {len(df_final)} paired scenarios.")
    return df_final, feature_dict

# ===== Cell 5 (code) =====
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def create_isolated_test_sets(df, feature_dict):
    # 1. NEW USERS: Hold out 10% of UserIDs entirely
    # These users will have their embeddings mapped to 0 (Unknown) later
    unique_users = df['UserID'].unique()
    train_u, test_u = train_test_split(unique_users, test_size=0.1, random_state=42)

    df_new_users = df[df['UserID'].isin(test_u)].copy()
    df_train_pool = df[df['UserID'].isin(train_u)].copy()

    # 2. NEW SCENARIOS: Hold out rows with specific characters
    # This tests if the model learned the "value" of these specific lives
    rare_chars = ['Stay_Homeless', 'Swerve_Homeless', 'Stay_Stroller', 'Swerve_Stroller']
    scenario_mask = (df_train_pool[rare_chars] > 0).any(axis=1)

    df_new_scenarios = df_train_pool[scenario_mask].copy()
    df_train_pool = df_train_pool[~scenario_mask]

    # 3. NEW GROUPS: Hold out a demographic slice
    country_cols = [col for col in df_train_pool.columns if col.startswith('Cnt_')]
    demographic_cols = [
        'Review_age', 'Review_education', 'Review_income',
        'Review_political', 'Review_religious'
    ] + country_cols
    group_key = df_train_pool[demographic_cols].astype(str).agg('-'.join, axis=1)

    # Generate the 30% group-based split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, holdout_idx = next(gss.split(df_train_pool, groups=group_key))

    group_mask = np.zeros(len(df_train_pool), dtype=bool)
    group_mask[holdout_idx] = True

    df_new_groups = df_train_pool[group_mask].copy()
    df_train_pool = df_train_pool[~group_mask]

    # 4. VALIDATION SPLIT: Create the 'Val' set from the remaining clean pool
    # This is for tuning during training
    df_train_final, df_val = train_test_split(df_train_pool, test_size=0.15, random_state=42)

    # 5. COMBINED (Optional "Boss Mode"): New User AND New Scenario
    # This is the hardest data for the model to predict
    combined_mask = (df_new_users[rare_chars] > 0).any(axis=1)
    df_combined = df_new_users[combined_mask].copy()

    print(f"--- Data Split Summary ---")
    print(f"Train:         {len(df_train_final)}")
    print(f"Val:           {len(df_val)}")
    print(f"New Users:     {len(df_new_users)}")
    print(f"New Scenarios: {len(df_new_scenarios)}")
    print(f"New Groups:    {len(df_new_groups)}")
    print(f"Combined Test: {len(df_combined)}")

    return df_train_final, df_val, df_new_users, df_new_scenarios, df_new_groups, df_combined

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

# ===== Cell 10 (code) =====
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Cell 11 (code) =====
import zipfile
import os

zip_path = "drive/My Drive/Interpretative Jury Learning/moral_machine.db.zip"
extract_path = "/content/" # Extracting to local Colab runtime for speed

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ===== Cell 12 (code) =====
def compare_user_ids(df1, df2, id_column='UserID'):
    """
    Compares unique UserIDs between two DataFrames and prints the overlap.
    """
    df1_ids = set(df1[id_column].unique())
    df2_ids = set(df2[id_column].unique())

    common_ids = df1_ids.intersection(df2_ids)

    print(f"\n--- User ID Comparison ---")
    print(f"Common IDs: {len(common_ids)}")
    if len(df1_ids) > 0:
        print(f"Percentage of DataFrame 1 IDs found in DataFrame 2: {len(common_ids) / len(df1_ids) * 100:.2f}%")
    if len(df2_ids) > 0:
        print(f"Percentage of DataFrame 2 IDs found in DataFrame 1: {len(common_ids) / len(df2_ids) * 100:.2f}%")

# ===== Cell 13 (markdown) =====
# ### Important Note on `MoralJuryDataset` `user_ids` Comparison
# 
# **Caution:** The `user_ids` attribute within each `MoralJuryDataset` instance (`train_ds.user_ids`, `new_users_ds.user_ids`, etc.) contains *factorized* (integer-encoded) IDs that are local to that specific dataset's `UserID` column. This means if the `UserID` column from `df_train` is factorized, and the `UserID` column from `df_new_users` is factorized, the integer `0` in `train_ds.user_ids` likely corresponds to a *different original user* than the integer `0` in `new_users_ds.user_ids`.
# 
# Therefore, comparing `train_ds.user_ids` and `new_users_ds.user_ids` directly will only tell you about overlaps in their *internal integer representations*, not whether the *same original user* is present in both sets. For a true comparison of original users, you should use the `compare_user_ids` function with the original DataFrames (e.g., `df_new_users`, `df_train`).
# 
# The following function is provided to compare the internal `user_ids` of `MoralJuryDataset` objects as requested, but its interpretation requires understanding this limitation.

# ===== Cell 14 (code) =====
def compare_dataset_user_ids(dataset1, dataset2):
    """
    Compares the internal factorized user IDs (ann_id) between two MoralJuryDataset objects.

    WARNING: These are factorized IDs specific to each dataset. An overlap here
    does NOT necessarily mean an overlap of original UserIDs unless the
    factorization was performed globally and consistently across all data.
    """
    # Convert PyTorch tensors to numpy arrays, then to Python sets
    ids1 = set(dataset1.user_ids.numpy())
    ids2 = set(dataset2.user_ids.numpy())

    common_ids = ids1.intersection(ids2)

    print(f"\n--- Dataset Internal User ID Comparison ---")
    print(f"Unique internal IDs in Dataset 1: {len(ids1)}")
    print(f"Unique internal IDs in Dataset 2: {len(ids2)}")
    print(f"Common internal IDs: {len(common_ids)}")
    if len(ids1) > 0:
        print(f"Percentage of Dataset 1 internal IDs found in Dataset 2: {len(common_ids) / len(ids1) * 100:.2f}%")
    if len(ids2) > 0:
        print(f"Percentage of Dataset 2 internal IDs found in Dataset 1: {len(common_ids) / len(ids2) * 100:.2f}%")

# Example usage (uncomment to run after defining the function):
# compare_dataset_user_ids(new_users_ds, train_ds)

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
