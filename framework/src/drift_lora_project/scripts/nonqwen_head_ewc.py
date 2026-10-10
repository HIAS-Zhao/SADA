#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScalerState:
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class HeadState:
    labels: list[str]
    weight: np.ndarray
    bias: np.ndarray
    scaler: ScalerState


@dataclass
class FisherState:
    weight: np.ndarray
    bias: np.ndarray


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def fit_scaler(x: np.ndarray) -> ScalerState:
    values = np.asarray(x, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return ScalerState(mean=mean, scale=scale)


def transform(x: np.ndarray, scaler: ScalerState) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - scaler.mean) / scaler.scale


def initialize_head(labels: list[str], feature_dim: int, scaler: ScalerState) -> HeadState:
    return HeadState(
        labels=list(labels),
        weight=np.zeros((len(labels), feature_dim), dtype=np.float64),
        bias=np.zeros(len(labels), dtype=np.float64),
        scaler=scaler,
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _ovr_gradient(scores: np.ndarray, target_index: int) -> np.ndarray:
    signs = -np.ones(scores.shape[0], dtype=np.float64)
    signs[target_index] = 1.0
    margins = signs * scores
    return -signs * _sigmoid(-margins)


def _optimal_init(alpha: float) -> float:
    typw = math.sqrt(1.0 / math.sqrt(alpha))
    initial_eta0 = typw
    return 1.0 / (initial_eta0 * alpha)


def normalized_fisher(fisher: FisherState) -> FisherState:
    weight_mean = max(float(np.mean(fisher.weight)), 1e-12)
    bias_mean = max(float(np.mean(fisher.bias)), 1e-12)
    return FisherState(
        weight=fisher.weight / weight_mean,
        bias=fisher.bias / bias_mean,
    )


def ewc_penalty(
    state: HeadState,
    anchor: HeadState,
    fisher: FisherState,
    *,
    reduction: str = "mean",
) -> float:
    weight_term = float(np.sum(fisher.weight * np.square(state.weight - anchor.weight)))
    bias_term = float(np.sum(fisher.bias * np.square(state.bias - anchor.bias)))
    total = weight_term + bias_term
    if reduction == "mean":
        total /= max(1, state.weight.size + state.bias.size)
    elif reduction != "sum":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return 0.5 * total


def train_head(
    x: np.ndarray,
    y: np.ndarray,
    initial: HeadState,
    *,
    epochs: int,
    seed: str,
    alpha: float = 1e-4,
    anchor: HeadState | None = None,
    fisher: FisherState | None = None,
    ewc_lambda: float = 0.0,
) -> tuple[HeadState, dict[str, Any]]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.int64)
    state = HeadState(
        labels=list(initial.labels),
        weight=np.array(initial.weight, copy=True),
        bias=np.array(initial.bias, copy=True),
        scaler=initial.scaler,
    )
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.shape}")
    if targets.shape != (features.shape[0],):
        raise ValueError(f"Target shape {targets.shape} does not match features {features.shape}")
    if ewc_lambda and (anchor is None or fisher is None):
        raise ValueError("EWC training requires anchor and fisher")

    normalized = normalized_fisher(fisher) if fisher is not None else None
    count = max(1, state.weight.size + state.bias.size)
    rng = np.random.default_rng(stable_seed(seed))
    optimal_init = _optimal_init(alpha)
    t = 1.0
    epoch_losses: list[float] = []
    epoch_penalties: list[float] = []

    for _epoch in range(epochs):
        order = np.arange(features.shape[0])
        rng.shuffle(order)
        total_loss = 0.0
        for index in order:
            row = features[index]
            target = int(targets[index])
            scores = state.weight @ row + state.bias
            gradient = _ovr_gradient(scores, target)
            eta = 1.0 / (alpha * (optimal_init + t - 1.0))

            state.weight *= max(0.0, 1.0 - eta * alpha)
            update = -eta * gradient
            state.weight += update[:, None] * row[None, :]
            state.bias += update

            if ewc_lambda and anchor is not None and normalized is not None:
                weight_strength = eta * ewc_lambda * normalized.weight / count
                bias_strength = eta * ewc_lambda * normalized.bias / count
                state.weight = (
                    state.weight + weight_strength * anchor.weight
                ) / (1.0 + weight_strength)
                state.bias = (
                    state.bias + bias_strength * anchor.bias
                ) / (1.0 + bias_strength)

            signs = -np.ones(scores.shape[0], dtype=np.float64)
            signs[target] = 1.0
            total_loss += float(np.logaddexp(0.0, -signs * scores).mean())
            t += 1.0
        epoch_losses.append(total_loss / max(1, features.shape[0]))
        epoch_penalties.append(
            ewc_penalty(state, anchor, normalized)
            if anchor is not None and normalized is not None
            else 0.0
        )

    metrics = {
        "epochs": epochs,
        "samples": int(features.shape[0]),
        "classes": len(state.labels),
        "alpha": alpha,
        "ewc_lambda": ewc_lambda,
        "epoch_log_loss": epoch_losses,
        "epoch_ewc_penalty": epoch_penalties,
    }
    return state, metrics


def compute_empirical_fisher(
    x: np.ndarray,
    y: np.ndarray,
    state: HeadState,
) -> FisherState:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.int64)
    fisher_weight = np.zeros_like(state.weight, dtype=np.float64)
    fisher_bias = np.zeros_like(state.bias, dtype=np.float64)
    for row, target in zip(features, targets):
        scores = state.weight @ row + state.bias
        gradient = _ovr_gradient(scores, int(target))
        fisher_weight += np.square(gradient[:, None] * row[None, :])
        fisher_bias += np.square(gradient)
    divisor = max(1, features.shape[0])
    return FisherState(
        weight=fisher_weight / divisor,
        bias=fisher_bias / divisor,
    )


def predict_scores(state: HeadState, x: np.ndarray) -> np.ndarray:
    features = transform(np.asarray(x, dtype=np.float64), state.scaler)
    return features @ state.weight.T + state.bias[None, :]


def save_head(path: Path, state: HeadState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(state.labels, dtype=str),
        weight=state.weight,
        bias=state.bias,
        scaler_mean=state.scaler.mean,
        scaler_scale=state.scaler.scale,
    )


def load_head(path: Path) -> HeadState:
    with np.load(path, allow_pickle=False) as payload:
        return HeadState(
            labels=[str(value) for value in payload["labels"].tolist()],
            weight=np.asarray(payload["weight"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
            scaler=ScalerState(
                mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
                scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
            ),
        )


def save_fisher(path: Path, fisher: FisherState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weight=fisher.weight, bias=fisher.bias)


def load_fisher(path: Path) -> FisherState:
    with np.load(path, allow_pickle=False) as payload:
        return FisherState(
            weight=np.asarray(payload["weight"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
        )
