from __future__ import annotations

import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    get_shared_split_plan,
    load_saved_splits,
    prepare_splits,
    save_prepared_splits,
)
from .models import Chronos2MultimodalRegressor, TTMMultimodalRegressor
from .evaluation import export_evaluation, regression_metrics, write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(setting: str) -> torch.device:
    if setting != "auto":
        return torch.device(setting)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_write(path: Path, value: Any) -> None:
    write_json(path, value)


def _make_model(
    model_name: str,
    feature_groups: dict[str, list[int]],
    model_config: dict[str, Any],
    context_length: int,
    dropout: float,
):
    model_config = dict(model_config)
    model_config["context_length"] = context_length
    if model_name == "ttm":
        return TTMMultimodalRegressor(feature_groups, model_config, dropout=dropout)
    if model_name == "chronos2":
        return Chronos2MultimodalRegressor(feature_groups, model_config, dropout=dropout)
    raise ValueError(f"Unknown model name: {model_name}")


def _loader(arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays["X"]),
        torch.from_numpy(arrays["y"]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, y_pred)


def _predict_scaled(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(arrays["y"]) == 0:
        return np.asarray([], dtype=np.float32)
    model.eval()
    predictions = []
    loader = _loader(arrays, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            output = model(x.to(device))
            predictions.extend(output.detach().cpu().numpy().reshape(-1).tolist())
    return np.asarray(predictions, dtype=np.float32)


def _save_weights(model: nn.Module, destination: Path) -> None:
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(destination))


def prepare_shared_dataset(config: dict[str, Any], project_root: Path) -> Path:
    output_dir = (project_root / config["paths"]["output"]).resolve()
    encoder_cfg = config.get("encoder", {})
    if encoder_cfg.get("refit_preprocessing_for_shared_split", False):
        _, split_summary, _ = get_shared_split_plan(config, project_root)
        encoder_root = (project_root / encoder_cfg["root"]).resolve()
        pipeline_config = (encoder_root / encoder_cfg["pipeline_config"]).resolve()
        reducer_script = encoder_root / "07_fit_feature_reducers.py"
        vector_script = encoder_root / "08_build_weekly_vectors.py"
        if not reducer_script.is_file() or not vector_script.is_file():
            raise FileNotFoundError(
                "Could not locate the encoder's PCA/reduction stage under "
                f"{encoder_root}."
            )
        subprocess.run(
            [
                sys.executable,
                str(reducer_script),
                "--config",
                str(pipeline_config),
                "--train-end",
                split_summary["train_end"],
            ],
            check=True,
            cwd=encoder_root,
        )
        subprocess.run(
            [sys.executable, str(vector_script), "--config", str(pipeline_config)],
            check=True,
            cwd=encoder_root,
        )
    prepared = prepare_splits(config, project_root)
    save_prepared_splits(prepared, output_dir)
    return output_dir


def train_one_model(
    model_name: str,
    config: dict[str, Any],
    project_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    training_cfg = config["training"]
    model_cfg = dict(config["models"][model_name])
    output_dir = output_dir or (project_root / config["paths"]["output"]).resolve()
    seed_everything(int(training_cfg["seed"]))
    arrays, split_manifest, preprocessor, target_scaler = load_saved_splits(output_dir)
    feature_groups = split_manifest["feature_groups_indices"]
    context_length = int(split_manifest["context_length"])
    device = resolve_device(str(training_cfg.get("device", "auto")))
    dropout = float(config.get("model_defaults", {}).get("dropout", 0.2))

    model = _make_model(
        model_name=model_name,
        feature_groups=feature_groups,
        model_config=model_cfg,
        context_length=context_length,
        dropout=dropout,
    ).to(device)

    if model_name == "chronos2":
        batch_size = int(training_cfg.get("chronos2_batch_size", 2))
        pretrained_lr = float(training_cfg["chronos2_learning_rate"])
    else:
        batch_size = int(training_cfg["batch_size"])
        pretrained_lr = float(training_cfg["ttm_learning_rate"])
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.pretrained_parameters()),
                "lr": pretrained_lr,
            },
            {
                "params": list(model.head_parameters()),
                "lr": float(training_cfg["head_learning_rate"]),
            },
        ],
        weight_decay=float(training_cfg["weight_decay"]),
    )
    loss_fn = nn.HuberLoss()
    train_loader = _loader(arrays["train"], batch_size=batch_size, shuffle=True)
    validation_loader = _loader(
        arrays["validation"], batch_size=batch_size, shuffle=False
    )
    patience = int(training_cfg["patience"])
    min_delta = float(training_cfg["min_delta"])
    freeze_epochs = int(training_cfg["freeze_pretrained_epochs"])
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    model_dir = output_dir / "models" / model_name
    checkpoint_dir = model_dir / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(training_cfg["epochs"])):
        model.set_pretrained_trainable(epoch >= freeze_epochs)
        model.train()
        if epoch < freeze_epochs:
            model.pretrained.eval()
        train_loss_total = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction.reshape_as(y), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg["gradient_clip_norm"])
            )
            optimizer.step()
            train_loss_total += float(loss.detach().cpu()) * len(y)
            train_count += len(y)

        model.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.no_grad():
            for x, y in validation_loader:
                prediction = model(x.to(device))
                val_loss_total += float(
                    loss_fn(prediction.reshape_as(y.to(device)), y.to(device)).cpu()
                ) * len(y)
                val_count += len(y)
        train_loss = train_loss_total / train_count if train_count else float("nan")
        val_loss = val_loss_total / val_count if val_count else float("nan")
        history.append(
            {"epoch": epoch + 1, "train_huber": train_loss, "validation_huber": val_loss}
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            _save_weights(model, checkpoint_dir / "model.safetensors")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a valid best validation checkpoint.")

    from safetensors.torch import load_file

    best_state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(best_state)
    model.eval()
    target_mean = float(target_scaler["mean"])
    target_scale = float(target_scaler["scale"])

    model_meta = {
        "model_name": model_name,
        "model_config": model_cfg,
        "backbone_config": model.pretrained.config.to_dict(),
        "context_length": context_length,
        "feature_names": split_manifest["feature_names"],
        "feature_groups_indices": feature_groups,
        "target_column": split_manifest["target_column"],
        "target_scaler": target_scaler,
        "split_manifest": split_manifest,
        "dropout": dropout,
        "package_version": "0.1.0",
    }
    _json_write(checkpoint_dir / "model_config.json", model_meta)
    _json_write(checkpoint_dir / "preprocessing.json", preprocessor)
    encoder_artifacts = Path(
        split_manifest["encoder_preprocessing"]["reducer_manifest"]
    ).parent
    shutil.copytree(
        encoder_artifacts,
        checkpoint_dir / "encoder_artifacts",
        dirs_exist_ok=True,
    )
    _json_write(
        checkpoint_dir / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_validation_huber_scaled": best_val,
            "history": history,
            "seed": int(training_cfg["seed"]),
            "pretrained_frozen_epochs": freeze_epochs,
            "device": str(device),
            "training_config": training_cfg,
        },
    )

    split_metrics: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        prediction_scaled = _predict_scaled(
            model, arrays[split_name], batch_size=batch_size, device=device
        )
        y_true = arrays[split_name]["y_raw"].astype(np.float64)
        y_pred = prediction_scaled.astype(np.float64) * target_scale + target_mean
        table = pd.DataFrame(
            {
                "brand_id": arrays[split_name]["brand_id"],
                "input_week_start": arrays[split_name]["input_week_start"],
                "split_date": arrays[split_name]["split_date"],
                "target_period": arrays[split_name]["target_period"],
                "bmi_actual": y_true,
                "bmi_predicted": y_pred,
                "residual": y_pred - y_true,
                "model": model_name,
            }
        )
        table.to_csv(model_dir / f"{split_name}_predictions.csv", index=False)
        split_metrics[split_name] = export_evaluation(
            table, model_dir, split_name,
            float(config.get("reporting", {}).get("change_threshold", 0.0)),
        )

    result = {
        "model": model_name,
        "base_model_id": model_cfg["model_id"],
        "base_revision": model_cfg.get("revision"),
        "best_epoch": best_epoch,
        "validation": split_metrics["validation"],
        "test": split_metrics["test"],
        "split_counts": split_manifest["sample_counts"],
        "split_basis": split_manifest["split_basis"],
    }
    _json_write(model_dir / "metrics.json", result)
    return result


def train_comparison(config: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    output_dir = (project_root / config["paths"]["output"]).resolve()
    # Prepare once per comparison invocation, then reuse these exact artifacts
    # for both model runs.
    prepare_shared_dataset(config, project_root)
    results = []
    for model_name in ("ttm", "chronos2"):
        results.append(train_one_model(model_name, config, project_root, output_dir))
    if config.get("baselines", {}).get("enabled", True):
        from .baselines import run_baselines
        results.extend(run_baselines(output_dir, config))
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {
            "model": result["model"],
            "base_model_id": result.get("base_model_id"),
            "base_revision": result.get("base_revision"),
            "best_epoch": result.get("best_epoch"),
            "train_samples": result["split_counts"]["train"],
            "validation_samples": result["split_counts"]["validation"],
            "test_samples": result["split_counts"]["test"],
            "split_basis": result["split_basis"],
        }
        for split in ("validation", "test"):
            for metric, value in result[split].items():
                row[f"{split}_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(comparison_dir / "metrics.csv", index=False)
    return results
