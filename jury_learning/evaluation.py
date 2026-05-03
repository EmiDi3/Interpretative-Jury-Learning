from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jury_learning.config import RunConfig
from jury_learning.data import DataBundle, MoralJuryDataset
from jury_learning.model import MoralJuryDCN
from jury_learning.training import build_model, resolve_device


def evaluate_isolated_performance(
    model: MoralJuryDCN,
    splits: dict[str, pd.DataFrame],
    feature_dict: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}

    for name, df_split in splits.items():
        ds = MoralJuryDataset(df_split, feature_dict)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                res_fts = batch["response_features"].to(device)
                group_fts = batch["group_features"].to(device)
                user_ids = batch["ann_id"].to(device)
                labels = batch["label"].to(device)

                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[name] = accuracy
        print(f"Accuracy for {name:15}: {100 * accuracy:.2f}%")

    return results


def plot_performance(results: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"])

    plt.ylabel("Accuracy")
    plt.title("Model Generalization Performance")
    plt.ylim(0, 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.1%}", ha="center")

    plt.show()


def evaluate_performance_by_country(
    model: MoralJuryDCN,
    df: pd.DataFrame,
    feature_dict: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}

    country_cols = [col for col in feature_dict["group_fts"] if col.startswith("Cnt_")]
    if not country_cols:
        print("No country dummy columns in group features; skipping country breakdown.")
        return results

    available_countries = [col.replace("Cnt_", "") for col in country_cols]

    for country_name in available_countries:
        country_filter_col = f"Cnt_{country_name}"
        if country_filter_col not in df.columns:
            continue

        df_country = df[df[country_filter_col] == 1].copy()
        if df_country.empty:
            continue

        if len(df_country) == 1:
            print(f"Skipping {country_name}: only one row (dataset edge case).")
            results[country_name] = float("nan")
            continue

        ds_country = MoralJuryDataset(df_country, feature_dict)
        loader_country = DataLoader(ds_country, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader_country:
                res_fts = batch["response_features"].to(device)
                group_fts = batch["group_features"].to(device)
                user_ids = batch["ann_id"].to(device)
                labels = batch["label"].to(device)

                outputs = model(res_fts, user_ids, group_fts).view(-1)
                predictions = (outputs > 0.5).float()

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[country_name] = accuracy
        print(f"Accuracy for {country_name:15}: {100 * accuracy:.2f}%")

    return results


def _continent_mapping(country_codes: list[str]) -> dict[str, str]:
    try:
        import pycountry_convert as pc
    except ImportError:
        print("Install pycountry_convert for continent plots: pip install pycountry_convert")
        return {}

    mapping: dict[str, str] = {}
    for country_code in country_codes:
        if country_code == "AUS":
            mapping[country_code] = "Australia"
            continue
        try:
            iso2 = pc.country_alpha3_to_country_alpha2(country_code)
            continent_code = pc.country_alpha2_to_continent_code(iso2)
            mapping[country_code] = pc.convert_continent_code_to_continent_name(continent_code)
        except KeyError:
            mapping[country_code] = "Unknown"
    return mapping


def average_accuracy_by_continent(country_accuracy: dict[str, float]) -> dict[str, float]:
    c2c = _continent_mapping(list(country_accuracy.keys()))
    if not c2c:
        return {}

    continent_groups: dict[str, list[float]] = {}
    for code, acc in country_accuracy.items():
        continent = c2c.get(code, "Unknown")
        continent_groups.setdefault(continent, []).append(acc)

    averages: dict[str, float] = {}
    for continent, accs in continent_groups.items():
        valid = [x for x in accs if pd.notna(x)]
        if valid:
            averages[continent] = float(np.mean(valid))
    return {k: v for k, v in averages.items() if pd.notna(v) and k != "Unknown"}


def plot_continent_accuracy(average_by_continent: dict[str, float], title: str) -> None:
    if not average_by_continent:
        return

    import matplotlib.pyplot as plt

    continents = list(average_by_continent.keys())
    accuracies = [average_by_continent[c] for c in continents]

    sorted_indices = np.argsort(accuracies)[::-1]
    continents = [continents[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(continents, accuracies, color="teal")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.ylim(0.5, 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.2%}", ha="center", va="bottom")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_country_subset(country_performance: dict[str, float], countries: tuple[str, ...], title: str) -> None:
    import matplotlib.pyplot as plt

    filtered = {c: country_performance.get(c, np.nan) for c in countries}
    filtered = {k: v for k, v in filtered.items() if pd.notna(v)}

    if not filtered:
        print("No overlapping countries to plot for requested subset.")
        return

    keys = list(filtered.keys())
    vals = list(filtered.values())

    plt.figure(figsize=(10, 6))
    plt.bar(keys, vals)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0.0, 1.0)

    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2%}", ha="center")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def load_trained_model(cfg: RunConfig, bundle: DataBundle, device: torch.device) -> MoralJuryDCN:
    model = build_model(cfg, bundle)
    path = cfg.model_path
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded model from {path}")
    return model


def run_evaluation_report(cfg: RunConfig, bundle: DataBundle, model: Optional[MoralJuryDCN] = None) -> MoralJuryDCN:
    device = resolve_device(cfg)
    model = model or load_trained_model(cfg, bundle, device)

    test_splits = {
        "Validation": bundle.df_val,
        "New Users": bundle.df_new_users,
        "New Scenarios": bundle.df_new_scenarios,
        "New Groups": bundle.df_new_groups,
        "Combined": bundle.df_combined,
    }

    performance = evaluate_isolated_performance(
        model=model,
        splits=test_splits,
        feature_dict=bundle.feature_dict,
        device=device,
        batch_size=cfg.eval_batch_size,
    )

    if cfg.plot_generalization:
        plot_performance(performance)

    if not cfg.plot_country_continent:
        return model

    country_val = evaluate_performance_by_country(
        model=model,
        df=bundle.df_val,
        feature_dict=bundle.feature_dict,
        device=device,
        batch_size=cfg.eval_batch_size,
    )
    print("\nCountry-wise performance (validation):")
    for country, accuracy in country_val.items():
        print(f"{country:15}: {accuracy * 100:.2f}%")

    avg_cont = average_accuracy_by_continent(country_val)
    plot_continent_accuracy(avg_cont, "Average accuracy by continent (validation)")

    country_nu = evaluate_performance_by_country(
        model=model,
        df=bundle.df_new_users,
        feature_dict=bundle.feature_dict,
        device=device,
        batch_size=cfg.eval_batch_size,
    )
    print("\nCountry-wise performance (new users):")
    for country, accuracy in country_nu.items():
        print(f"{country:15}: {accuracy * 100:.2f}%")

    avg_cont_nu = average_accuracy_by_continent(country_nu)
    plot_continent_accuracy(avg_cont_nu, "Average accuracy by continent (new users)")

    plot_country_subset(
        country_nu,
        cfg.country_plot_focus,
        "New-user accuracy for selected countries",
    )

    return model
