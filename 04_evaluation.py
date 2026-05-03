# Auto-generated from Interpretative_jury_learning.ipynb
# File: 04_evaluation.py

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

# ===== Cell 9 (code) =====
def evaluate_isolated_performance(model, splits, feature_dict, device):
    model.eval()
    results = {}

    for name, df_split in splits.items():
        # 1. Create Dataset and Loader for this split
        ds = MoralJuryDataset(df_split, feature_dict)
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                res_fts = batch['response_features'].to(device)
                group_fts = batch['group_features'].to(device)
                user_ids = batch['ann_id'].to(device) # Or batch['user_ids']
                labels = batch['label'].to(device)

                # 2. Forward pass
                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        results[name] = accuracy
        print(f"Accuracy for {name:15}: {100*correct/total:.2f}%")

    return results

# ===== Cell 10 (code) =====
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Cell 16 (code) =====
model_path = "moral_jury_dcn_model.pth"

# Initialize the model first with the correct architecture parameters
# These parameters should match those used during training
num_users = NUM_USERS_FOR_EMBEDDING
num_response_features = len(feature_dict['response_fts'])
num_group_features = len(feature_dict['group_fts'])
EMBED_DIM = 128  # Assuming these were the embedding dimensions used during training
HIDDEN_DIM = 512 # Assuming these were the hidden dimensions used during training

loaded_model = MoralJuryDCN(
    num_users=num_users,
    num_response_features=num_response_features,
    num_group_features=num_group_features,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
)

# Load the state dictionary
loaded_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
loaded_model.to(DEVICE)
loaded_model.eval() # Set the model to evaluation mode

print(f"Model loaded successfully from {model_path}")

# ===== Cell 18 (code) =====
# Wrap your processed test dataframes into a dictionary
test_splits = {
    "Validation": df_val,
    "New Users": df_new_users,
    "New Scenarios": df_new_scenarios,
    "New Groups": df_new_groups,
    "Combined": df_combined
}

# Call the evaluation function
performance_results = evaluate_isolated_performance(
    model=loaded_model,
    splits=test_splits,
    feature_dict=feature_dict,
    device=DEVICE
)

# ===== Cell 19 (code) =====
import matplotlib.pyplot as plt

def plot_performance(results):
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0'])

    plt.ylabel('Accuracy')
    plt.title('Model Generalization Performance')
    plt.ylim(0, 1.0)

    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.1%}', ha='center')

    plt.show()

# Run the plot
plot_performance(performance_results)

# ===== Cell 20 (code) =====
def evaluate_performance_by_country(model, df, feature_dict, device):
    model.eval()
    results = {}

    # Identify country columns based on feature_dict
    country_cols = [col for col in feature_dict['group_fts'] if col.startswith('Cnt_')]

    if not country_cols:
        print("No country-specific features found in feature_dict['group_fts']. Cannot evaluate by country.")
        return {}

    # Get all unique countries from the dataframe.
    # A row might belong to multiple 'countries' if a user belongs to multiple groups due to dummy encoding issues,
    # but typically only one Cnt_X column will be 1 for a given row.
    # We extract the country names from the column names.
    available_countries = [col.replace('Cnt_', '') for col in country_cols]

    print("Evaluating performance for each country...")

    for country_name in available_countries:
        # Create a mask for the current country
        country_filter_col = f'Cnt_{country_name}'
        if country_filter_col not in df.columns:
            continue # Skip if this specific country column doesn't exist in the df (e.g., if it was a dummy column not present in this split)

        df_country = df[df[country_filter_col] == 1].copy()

        if df_country.empty:
            # print(f"No data for country: {country_name}")
            continue

        # --- Workaround for MoralJuryDataset single-row issue ---
        # The MoralJuryDataset currently creates a 0-d tensor for labels
        # if initialized with a DataFrame containing only one row, which causes
        # a TypeError: len() of a 0-d tensor when DataLoader calls __len__.
        # This workaround skips evaluation for countries with only one data point.
        if len(df_country) == 1:
            print(f"Skipping evaluation for {country_name}: only 1 data point found. (Workaround for MoralJuryDataset issue).")
            results[country_name] = np.nan # Assign NaN or a placeholder
            continue
        # --- End Workaround ---

        # Create Dataset and Loader for this country
        ds_country = MoralJuryDataset(df_country, feature_dict)
        loader_country = DataLoader(ds_country, batch_size=64, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader_country:
                res_fts = batch['response_features'].to(device)
                group_fts = batch['group_features'].to(device)
                user_ids = batch['ann_id'].to(device)
                labels = batch['label'].to(device)

                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        results[country_name] = accuracy
        print(f"Accuracy for {country_name:15}: {100*accuracy:.2f}%")

    return results

# ===== Cell 21 (code) =====
# Evaluate performance by country on the validation dataset
country_performance_val = evaluate_performance_by_country(
    model=loaded_model,
    df=df_val, # Using the validation DataFrame
    feature_dict=feature_dict,
    device=DEVICE
)

print("\nCountry-wise performance on Validation Set:")
for country, accuracy in country_performance_val.items():
    print(f"{country:15}: {accuracy*100:.2f}%")

# ===== Cell 22 (code) =====
import subprocess
subprocess.run("pip install pycountry_convert", shell=True, check=False)

# ===== Cell 23 (code) =====
import pycountry_convert as pc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Install pycountry-convert if not already installed
import subprocess
subprocess.run("pip install pycountry-convert -qq", shell=True, check=False)

# 1. Get the country codes from the evaluation results
evaluated_countries = country_performance_val.keys()

# 2. Create a mapping from country code (alpha-3) to continent name
country_to_continent = {}
for country_code in evaluated_countries:
    if country_code == 'AUS': # Explicitly assign Australia as a continent
        country_to_continent[country_code] = 'Australia'
    else:
        try:
            # Convert 3-letter ISO country code to 2-letter ISO country code
            iso2_country_code = pc.country_alpha3_to_country_alpha2(country_code)
            # Convert 2-letter ISO country code to continent code
            continent_code = pc.country_alpha2_to_continent_code(iso2_country_code)
            # Convert continent code to continent name
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            country_to_continent[country_code] = continent_name
        except KeyError:
            # Handle cases where a country code might not be found in pycountry-convert
            country_to_continent[country_code] = 'Unknown'
            # print(f"Could not map country code: {country_code}") # Uncomment for debugging unknown countries

# 3. Group accuracies by continent
continent_accuracies = {}
for country_code, accuracy in country_performance_val.items():
    continent = country_to_continent.get(country_code, 'Unknown')
    if continent not in continent_accuracies:
        continent_accuracies[continent] = []
    continent_accuracies[continent].append(accuracy)

# 4. Calculate average accuracy per continent, handling NaNs from skipped countries
average_continent_accuracies = {}
for continent, accs in continent_accuracies.items():
    # Filter out None/NaN values before calculating mean
    valid_accs = [x for x in accs if pd.notna(x)]
    if valid_accs:
        average_continent_accuracies[continent] = np.mean(valid_accs)
    else:
        # If all accuracies for a continent are NaN (e.g., all countries skipped), assign NaN
        average_continent_accuracies[continent] = np.nan

# Filter out continents with no valid data and the 'Unknown' continent
average_continent_accuracies = {k: v for k, v in average_continent_accuracies.items() if pd.notna(v) and k != 'Unknown'}

# 5. Plot the results
continents = list(average_continent_accuracies.keys())
accuracies = [average_continent_accuracies[c] for c in continents]

# Sort for better visualization
sorted_indices = np.argsort(accuracies)[::-1] # Sort in descending order
continents = [continents[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]

plt.figure(figsize=(12, 7))
bars = plt.bar(continents, accuracies, color='teal')
plt.ylabel('Average Accuracy')
plt.title('Average Model Accuracy by Continent (Validation Set)')
plt.ylim(0.5, 1.0) # Assuming accuracies are mostly above 50%

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2%}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ===== Cell 24 (code) =====
# Evaluate performance by country on the NEW USERS dataset
country_performance_new_users = evaluate_performance_by_country(
    model=loaded_model,
    df=df_new_users, # Using the new users DataFrame
    feature_dict=feature_dict,
    device=DEVICE
)

print("\nCountry-wise performance on New Users Set:")
for country, accuracy in country_performance_new_users.items():
    print(f"{country:15}: {accuracy*100:.2f}%")


# --- Continent-wise analysis for New Users ---

# 1. Get the country codes from the evaluation results for new users
evaluated_countries_nu = country_performance_new_users.keys()

# 2. Create a mapping from country code (alpha-3) to continent name
country_to_continent_nu = {}
for country_code in evaluated_countries_nu:
    if country_code == 'AUS': # Explicitly assign Australia as a continent
        country_to_continent_nu[country_code] = 'Australia'
    else:
        try:
            iso2_country_code = pc.country_alpha3_to_country_alpha2(country_code)
            continent_code = pc.country_alpha2_to_continent_code(iso2_country_code)
            continent_name = pc.convert_continent_code_to_continent_name(continent_code)
            country_to_continent_nu[country_code] = continent_name
        except KeyError:
            country_to_continent_nu[country_code] = 'Unknown'

# 3. Group accuracies by continent
continent_accuracies_nu = {}
for country_code, accuracy in country_performance_new_users.items():
    continent = country_to_continent_nu.get(country_code, 'Unknown')
    if continent not in continent_accuracies_nu:
        continent_accuracies_nu[continent] = []
    continent_accuracies_nu[continent].append(accuracy)

# 4. Calculate average accuracy per continent, handling NaNs
average_continent_accuracies_nu = {}
for continent, accs in continent_accuracies_nu.items():
    valid_accs = [x for x in accs if pd.notna(x)]
    if valid_accs:
        average_continent_accuracies_nu[continent] = np.mean(valid_accs)
    else:
        average_continent_accuracies_nu[continent] = np.nan

# Filter out continents with no valid data and the 'Unknown' continent
average_continent_accuracies_nu = {k: v for k, v in average_continent_accuracies_nu.items() if pd.notna(v) and k != 'Unknown'}

# ===== Cell 25 (code) =====
# 5. Plot the results for new users
continents_nu = list(average_continent_accuracies_nu.keys())
accuracies_nu = [average_continent_accuracies_nu[c] for c in continents_nu]

# Sort for better visualization
sorted_indices_nu = np.argsort(accuracies_nu)[::-1] # Sort in descending order
continents_nu = [continents_nu[i] for i in sorted_indices_nu]
accuracies_nu = [accuracies_nu[i] for i in sorted_indices_nu]

plt.figure(figsize=(12, 7))
bars_nu = plt.bar(continents_nu, accuracies_nu, color='purple')
plt.ylabel('Average Accuracy')
plt.title('Average Model Accuracy by Continent (New Users Set)')
plt.ylim(0.5, 1.0) # Assuming accuracies are mostly above 50%

# Add percentage labels on top of bars
for bar in bars_nu:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2%}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ===== Cell 26 (code) =====
import matplotlib.pyplot as plt
import numpy as np

# Define the list of requested countries (alpha-3 codes)
requested_countries = ['CHN', 'USA', 'CAN', 'DEU', 'FRA', 'GBR', 'BRA']

# Filter the new user performance results for these countries
filtered_accuracies = {
    country: country_performance_new_users.get(country, np.nan)
    for country in requested_countries
}

# Remove any NaN values (countries not found or skipped in previous evaluation)
filtered_accuracies = {k: v for k, v in filtered_accuracies.items() if not np.isnan(v)}

# Prepare data for plotting
countries_to_plot = list(filtered_accuracies.keys())
accuracies_to_plot = list(filtered_accuracies.values())

# Create the bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(countries_to_plot, accuracies_to_plot, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet', 'orange', 'lightgray'])

plt.ylabel('Accuracy')
plt.title('New User Accuracies by Specific Country')
plt.ylim(0.0, 1.0) # Accuracies are between 0 and 1

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
