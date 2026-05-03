from __future__ import annotations

import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from jury_learning.config import RunConfig

_REQUIRED_MM_TABLES = ("survey", "responses")


def _ensure_db_file_exists(db_path: str) -> None:
    """SQLite opens a new empty database when the path is missing; avoid that confusing failure mode."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(
            f"SQLite database not found at {db_path!s}. "
            "If the path is wrong, SQLite creates an empty file with no tables, which then fails "
            "with errors like 'no such table: survey'. "
            "Download the Moral Machine SQLite export (with tables survey and responses), "
            "point RunConfig.db_path at it, or set extract_db_zip / extract_db_zip_dest."
        )


def _require_sqlite_tables(conn: sqlite3.Connection, required: tuple[str, ...]) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    present = {str(r[0]).lower() for r in rows}
    missing = [t for t in required if t.lower() not in present]
    if missing:
        raise ValueError(
            f"Database is missing required table(s): {missing}. "
            f"Found: {sorted(present)}. "
            "This code expects the MIT Moral Machine SQLite schema (tables `survey` and `responses`)."
        )


def extract_db_if_needed(cfg: RunConfig) -> None:
    if not cfg.extract_db_zip:
        return
    zip_path = Path(cfg.extract_db_zip)
    dest = Path(cfg.extract_db_zip_dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def take_scenario_data_sql(db_path: str, output_file_path: str, *, verbose: bool = True) -> pd.DataFrame:
    _ensure_db_file_exists(db_path)
    conn = sqlite3.connect(db_path)
    _require_sqlite_tables(conn, _REQUIRED_MM_TABLES)

    character_cols = [
        "PedPed",
        "Barrier",
        "NumberOfCharacters",
        "CrossingSignal",
        "Man",
        "Woman",
        "Pregnant",
        "Stroller",
        "OldMan",
        "OldWoman",
        "Boy",
        "Girl",
        "Homeless",
        "LargeWoman",
        "LargeMan",
        "Criminal",
        "MaleExecutive",
        "FemaleExecutive",
        "FemaleAthlete",
        "MaleAthlete",
        "FemaleDoctor",
        "MaleDoctor",
        "Dog",
        "Cat",
    ]

    sql_query = f"""
    SELECT
        {", ".join([f"r0.{c} AS Stay_{c}" for c in character_cols])},
        {", ".join([f"r1.{c} AS Swerve_{c}" for c in character_cols])}
    FROM survey s
    JOIN responses r0 ON s.ResponseID = r0.ResponseID AND r0.Intervention = 0
    JOIN responses r1 ON s.ResponseID = r1.ResponseID AND r1.Intervention = 1
    """

    df_final = pd.read_sql_query(sql_query, conn)
    conn.close()

    scenario_description_cols = [col for col in df_final.columns if col.startswith(("Stay_", "Swerve_"))]
    df_unique_scenarios = df_final.drop_duplicates(subset=scenario_description_cols).reset_index(drop=True)
    df_unique_scenarios.to_csv(output_file_path, index=False)
    if verbose:
        print(f"Unique character scenarios saved to {output_file_path}")
        print(
            f"Successfully processed and saved {len(df_unique_scenarios)} unique paired scenarios "
            "with only character features."
        )
    return df_unique_scenarios


def merge_and_process_moral_data_sql(
    db_path: str, subset_size: int, *, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    _ensure_db_file_exists(db_path)
    conn = sqlite3.connect(db_path)
    _require_sqlite_tables(conn, _REQUIRED_MM_TABLES)

    character_cols = [
        "NumberOfCharacters",
        "Pedped",
        "Barrier",
        "CrossingSignal",
        "Man",
        "Woman",
        "Pregnant",
        "Stroller",
        "OldMan",
        "OldWoman",
        "Boy",
        "Girl",
        "Homeless",
        "LargeWoman",
        "LargeMan",
        "Criminal",
        "MaleExecutive",
        "FemaleExecutive",
        "FemaleAthlete",
        "MaleAthlete",
        "FemaleDoctor",
        "MaleDoctor",
        "Dog",
        "Cat",
    ]

    if verbose:
        print(f"Querying and pairing up to {subset_size} survey rows...")
    sql_query = f"""
    SELECT
        s.ResponseID,
        s.UserID,
        s.Review_age, s.Review_education, s.Review_gender,
        s.Review_income, s.Review_political, s.Review_religious, s.UserCountry3,
        {", ".join([f"r0.{c} AS Stay_{c}" for c in character_cols])},
        r1.Saved AS Swerve_Saved,
        {", ".join([f"r1.{c} AS Swerve_{c}" for c in character_cols])}
    FROM (SELECT * FROM survey LIMIT {subset_size}) s
    JOIN responses r0 ON s.ResponseID = r0.ResponseID AND r0.Intervention = 0
    JOIN responses r1 ON s.ResponseID = r1.ResponseID AND r1.Intervention = 1
    """

    df_final = pd.read_sql_query(sql_query, conn)
    conn.close()

    user_encoder = LabelEncoder()
    df_final["UserID"] = user_encoder.fit_transform(df_final["UserID"]) + 1

    df_final["Decision_Swerve"] = df_final["Swerve_Saved"].astype(int)

    df_final["Review_age"] = pd.to_numeric(df_final["Review_age"], errors="coerce")
    df_final["Review_age"] = df_final["Review_age"].fillna(df_final["Review_age"].median())
    df_final["Review_age"] = df_final["Review_age"].clip(18, 75)
    df_final["Review_age"] = (df_final["Review_age"] - 18) / (75 - 18)

    edu_map = {
        "underHigh": 0.1,
        "high": 0.3,
        "vocational": 0.4,
        "college": 0.6,
        "bachelor": 0.8,
        "graduate": 1.0,
        "other": 0.5,
        "default": 0.5,
    }
    df_final["Review_education"] = df_final["Review_education"].map(edu_map).fillna(0.5)

    income_map = {
        "under5000": 0.1,
        "5000": 0.2,
        "10000": 0.3,
        "15000": 0.4,
        "25000": 0.5,
        "35000": 0.6,
        "50000": 0.7,
        "80000": 0.8,
        "above100000": 1.0,
        "default": 0.5,
    }
    df_final["Review_income"] = df_final["Review_income"].map(income_map).fillna(0.5)
    df_final["Review_political"] = pd.to_numeric(df_final["Review_political"], errors="coerce").fillna(0.5)
    df_final["Review_religious"] = pd.to_numeric(df_final["Review_religious"], errors="coerce").fillna(0.5)

    categorical_cols = ["Review_gender", "UserCountry3"]
    df_final = pd.get_dummies(df_final, columns=categorical_cols, prefix=["Gen", "Cnt"])

    dummy_prefixes = ("Gen_", "Cnt_")
    group_fts = (
        [
            "Review_age",
            "Review_education",
            "Review_income",
            "Review_political",
            "Review_religious",
        ]
        + [col for col in df_final.columns if col.startswith(dummy_prefixes)]
    )

    response_fts = [f"Stay_{c}" for c in character_cols] + [f"Swerve_{c}" for c in character_cols]

    feature_dict = {
        "user_fts": ["UserID"],
        "group_fts": group_fts,
        "response_fts": response_fts,
        "target": ["Decision_Swerve"],
    }

    df_final[group_fts] = df_final[group_fts].astype(float)

    if verbose:
        print(f"Successfully processed {len(df_final)} paired scenarios.")
    return df_final, feature_dict


def create_isolated_test_sets(df: pd.DataFrame, feature_dict: dict, cfg: RunConfig):
    rs = cfg.random_seed

    unique_users = df["UserID"].unique()
    train_u, test_u = train_test_split(
        unique_users,
        test_size=cfg.new_users_holdout_fraction,
        random_state=rs,
    )

    df_new_users = df[df["UserID"].isin(test_u)].copy()
    df_train_pool = df[df["UserID"].isin(train_u)].copy()

    rare_chars = list(cfg.rare_scenario_columns)
    missing = [c for c in rare_chars if c not in df_train_pool.columns]
    if missing:
        raise ValueError(f"Rare scenario columns not in dataframe: {missing}")

    scenario_mask = (df_train_pool[rare_chars] > 0).any(axis=1)
    df_new_scenarios = df_train_pool[scenario_mask].copy()
    df_train_pool = df_train_pool[~scenario_mask]

    country_cols = [col for col in df_train_pool.columns if col.startswith("Cnt_")]
    demographic_cols = [
        "Review_age",
        "Review_education",
        "Review_income",
        "Review_political",
        "Review_religious",
    ] + country_cols
    group_key = df_train_pool[demographic_cols].astype(str).agg("-".join, axis=1)

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.new_groups_holdout_fraction, random_state=rs)
    train_idx, holdout_idx = next(gss.split(df_train_pool, groups=group_key))

    group_mask = np.zeros(len(df_train_pool), dtype=bool)
    group_mask[holdout_idx] = True

    df_new_groups = df_train_pool[group_mask].copy()
    df_train_pool = df_train_pool[~group_mask]

    df_train_final, df_val = train_test_split(
        df_train_pool,
        test_size=cfg.val_fraction,
        random_state=rs,
    )

    combined_mask = (df_new_users[rare_chars] > 0).any(axis=1)
    df_combined = df_new_users[combined_mask].copy()

    if cfg.verbose:
        print("--- Data split sizes ---")
        print(f"Train:         {len(df_train_final)}")
        print(f"Val:           {len(df_val)}")
        print(f"New Users:     {len(df_new_users)}")
        print(f"New Scenarios: {len(df_new_scenarios)}")
        print(f"New Groups:    {len(df_new_groups)}")
        print(f"Combined Test: {len(df_combined)}")

    return df_train_final, df_val, df_new_users, df_new_scenarios, df_new_groups, df_combined


class MoralJuryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_dict: dict):
        self.response_features = torch.tensor(df[feature_dict["response_fts"]].values, dtype=torch.float32)
        self.group_features = torch.tensor(df[feature_dict["group_fts"]].values, dtype=torch.float32)
        labels = torch.tensor(df[feature_dict["target"]].values, dtype=torch.float32).reshape(-1)
        self.labels = labels
        self.user_ids = torch.tensor(df["UserID"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "response_features": self.response_features[idx],
            "group_features": self.group_features[idx],
            "label": self.labels[idx],
            "ann_id": self.user_ids[idx],
        }


def assign_unseen_user_id(df: pd.DataFrame, train_user_ids: set, unseen_id: int = 0) -> pd.DataFrame:
    df_copy = df.copy()
    unseen_mask = ~df_copy["UserID"].isin(train_user_ids)
    df_copy.loc[unseen_mask, "UserID"] = unseen_id
    return df_copy


@dataclass
class DataBundle:
    df_processed: pd.DataFrame
    feature_dict: dict
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_new_users: pd.DataFrame
    df_new_scenarios: pd.DataFrame
    df_new_groups: pd.DataFrame
    df_combined: pd.DataFrame
    num_users_for_embedding: int
    train_loader: DataLoader
    val_loader: DataLoader


def build_data_bundle(cfg: RunConfig) -> DataBundle:
    extract_db_if_needed(cfg)

    if cfg.export_unique_scenarios:
        take_scenario_data_sql(cfg.db_path, cfg.scenarios_csv, verbose=cfg.verbose)

    df_processed, feature_dict = merge_and_process_moral_data_sql(
        cfg.db_path, cfg.sql_subset_size, verbose=cfg.verbose
    )

    splits = create_isolated_test_sets(df_processed, feature_dict, cfg)
    df_train, df_val, df_new_users, df_new_scenarios, df_new_groups, df_combined = splits

    train_user_ids = set(df_train["UserID"].unique())
    unseen_id = 0

    df_val = assign_unseen_user_id(df_val, train_user_ids, unseen_id)
    df_new_users = assign_unseen_user_id(df_new_users, train_user_ids, unseen_id)
    df_new_scenarios = assign_unseen_user_id(df_new_scenarios, train_user_ids, unseen_id)
    df_new_groups = assign_unseen_user_id(df_new_groups, train_user_ids, unseen_id)
    df_combined = assign_unseen_user_id(df_combined, train_user_ids, unseen_id)

    num_users_for_embedding = int(df_processed["UserID"].max()) + 1

    train_ds = MoralJuryDataset(df_train, feature_dict)
    val_ds = MoralJuryDataset(df_val, feature_dict)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    return DataBundle(
        df_processed=df_processed,
        feature_dict=feature_dict,
        df_train=df_train,
        df_val=df_val,
        df_new_users=df_new_users,
        df_new_scenarios=df_new_scenarios,
        df_new_groups=df_new_groups,
        df_combined=df_combined,
        num_users_for_embedding=num_users_for_embedding,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def compare_user_ids(df1: pd.DataFrame, df2: pd.DataFrame, id_column: str = "UserID") -> None:
    df1_ids = set(df1[id_column].unique())
    df2_ids = set(df2[id_column].unique())
    common_ids = df1_ids.intersection(df2_ids)

    print("\n--- User ID Comparison ---")
    print(f"Common IDs: {len(common_ids)}")
    if len(df1_ids) > 0:
        print(f"Percentage of DataFrame 1 IDs found in DataFrame 2: {len(common_ids) / len(df1_ids) * 100:.2f}%")
    if len(df2_ids) > 0:
        print(f"Percentage of DataFrame 2 IDs found in DataFrame 1: {len(common_ids) / len(df2_ids) * 100:.2f}%")


def compare_dataset_user_ids(dataset1: MoralJuryDataset, dataset2: MoralJuryDataset) -> None:
    ids1 = set(dataset1.user_ids.numpy())
    ids2 = set(dataset2.user_ids.numpy())
    common_ids = ids1.intersection(ids2)

    print("\n--- Dataset Internal User ID Comparison ---")
    print(f"Unique internal IDs in Dataset 1: {len(ids1)}")
    print(f"Unique internal IDs in Dataset 2: {len(ids2)}")
    print(f"Common internal IDs: {len(common_ids)}")
    if len(ids1) > 0:
        print(f"Percentage of Dataset 1 internal IDs found in Dataset 2: {len(common_ids) / len(ids1) * 100:.2f}%")
    if len(ids2) > 0:
        print(f"Percentage of Dataset 2 internal IDs found in Dataset 1: {len(common_ids) / len(ids2) * 100:.2f}%")
