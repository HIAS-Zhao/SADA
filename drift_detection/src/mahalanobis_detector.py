from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import PCA

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


class MahalanobisDetector:
    """Window-level Mahalanobis detector with PCA baseline modeling and Monte Carlo calibration."""

    def __init__(
        self,
        pca_dim: int,
        window_size: int,
        calibration_windows: int,
        calibration_quantile: float,
        cov_eps: float = 1e-6,
        random_seed: int = 42,
    ) -> None:
        self.requested_pca_dim = pca_dim
        self.window_size = window_size
        self.calibration_windows = calibration_windows
        self.calibration_quantile = calibration_quantile
        self.cov_eps = cov_eps
        self.random_seed = random_seed

        self.pca: PCA | None = None
        self.actual_pca_dim: int | None = None
        self.mu_base: np.ndarray | None = None
        self.sigma_base: np.ndarray | None = None
        self.sigma_inv: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibration_scores: np.ndarray | None = None

    def fit_reference(self, features: np.ndarray) -> None:
        features = self._validate_features(features, name="reference")
        actual_dim = min(self.requested_pca_dim, features.shape[0], features.shape[1])
        if actual_dim < 1:
            raise ValueError("PCA dimension must be at least 1 after clipping.")

        self.pca = PCA(n_components=actual_dim, svd_solver="auto", random_state=self.random_seed)
        projected = self.pca.fit_transform(features)
        self.actual_pca_dim = actual_dim
        self.mu_base = projected.mean(axis=0)
        self.sigma_base = np.cov(projected, rowvar=False) + np.eye(actual_dim, dtype=np.float64) * self.cov_eps
        self.sigma_inv = np.linalg.pinv(self.sigma_base)

    def calibrate_threshold(self, features: np.ndarray, progress_label: str | None = None) -> float:
        transformed = self.transform(features)
        rng = np.random.default_rng(self.random_seed)
        scores = np.empty(self.calibration_windows, dtype=np.float64)

        calibration_indices = tqdm(
            range(self.calibration_windows),
            total=self.calibration_windows,
            desc=progress_label or "Threshold calibration",
            leave=False,
            mininterval=1.0,
        )
        for index in calibration_indices:
            window = self._sample_window(transformed, self.window_size, rng)
            scores[index] = self._mahalanobis_from_projected(window)

        self.calibration_scores = scores
        self.threshold = float(np.quantile(scores, self.calibration_quantile))
        return self.threshold

    def score(self, features: np.ndarray) -> float:
        transformed = self.transform(features)
        if transformed.shape[0] != self.window_size:
            raise ValueError(
                f"Expected a window with {self.window_size} samples, got {transformed.shape[0]} samples."
            )
        return self._mahalanobis_from_projected(transformed)

    def score_projected_window(self, projected_window: np.ndarray) -> float:
        projected_window = self._validate_features(projected_window, name="projected_window")
        if projected_window.shape[0] != self.window_size:
            raise ValueError(
                f"Expected a projected window with {self.window_size} samples, got {projected_window.shape[0]} samples."
            )
        return self._mahalanobis_from_projected(projected_window)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = self._validate_features(features, name="features")
        if self.pca is None:
            raise RuntimeError("PCA model has not been fitted.")
        return self.pca.transform(features)

    def evaluate_mixed_windows(
        self,
        no_drift_features: np.ndarray,
        drift_features: np.ndarray,
        drift_ratios: Sequence[float],
        windows_per_ratio: int,
        progress_label: str | None = None,
    ) -> list[dict[str, float]]:
        no_drift_projected = self.transform(no_drift_features)
        drift_projected = self.transform(drift_features)
        rng = np.random.default_rng(self.random_seed)
        results: list[dict[str, float]] = []

        ratio_iterator = tqdm(
            drift_ratios,
            total=len(drift_ratios),
            desc=progress_label or "Mixed-window evaluation",
            leave=False,
            mininterval=1.0,
        )
        for ratio in ratio_iterator:
            detections = 0
            n_drift = int(round(self.window_size * ratio))
            n_clean = self.window_size - n_drift
            scores = np.empty(windows_per_ratio, dtype=np.float64)

            window_iterator = tqdm(
                range(windows_per_ratio),
                total=windows_per_ratio,
                desc=f"ratio={float(ratio):.2f}",
                leave=False,
                mininterval=1.0,
            )
            for index in window_iterator:
                clean_window = self._sample_window(no_drift_projected, n_clean, rng) if n_clean > 0 else None
                drift_window = self._sample_window(drift_projected, n_drift, rng) if n_drift > 0 else None

                parts = [part for part in (clean_window, drift_window) if part is not None]
                window = np.concatenate(parts, axis=0)
                if window.shape[0] != self.window_size:
                    raise RuntimeError("Constructed window has an unexpected size.")
                window = window[rng.permutation(self.window_size)]
                score = self._mahalanobis_from_projected(window)
                scores[index] = score
                detections += int(score > self.threshold)

            results.append(
                {
                    "drift_ratio": float(ratio),
                    "windows": int(windows_per_ratio),
                    "n_clean": int(n_clean),
                    "n_drift": int(n_drift),
                    "detection_rate": float(detections / windows_per_ratio),
                    "metric_name": "fpr" if ratio == 0.0 else "recall",
                    "mean_mahalanobis": float(scores.mean()),
                    "std_mahalanobis": float(scores.std(ddof=0)),
                }
            )

        return results

    def save_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.pca is None or self.mu_base is None or self.sigma_base is None or self.sigma_inv is None:
            raise RuntimeError("Detector must be fitted before saving artifacts.")

        joblib.dump(self.pca, output_dir / "pca_model.joblib")
        np.savez(
            output_dir / "baseline_stats.npz",
            mu_base=self.mu_base,
            sigma_base=self.sigma_base,
            sigma_inv=self.sigma_inv,
            threshold=np.array(self.threshold if self.threshold is not None else np.nan, dtype=np.float64),
            actual_pca_dim=np.array(self.actual_pca_dim if self.actual_pca_dim is not None else -1, dtype=np.int64),
        )
        if self.calibration_scores is not None:
            np.savez(
                output_dir / "threshold_calibration.npz",
                scores=self.calibration_scores,
                quantile=np.array(self.calibration_quantile, dtype=np.float64),
                threshold=np.array(self.threshold if self.threshold is not None else np.nan, dtype=np.float64),
            )

    def _mahalanobis_from_projected(self, projected_window: np.ndarray) -> float:
        if self.mu_base is None or self.sigma_inv is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        mu_window = projected_window.mean(axis=0)
        diff = mu_window - self.mu_base
        sq = float(diff @ self.sigma_inv @ diff)
        return float(np.sqrt(max(sq, 0.0)))

    def _sample_window(self, features: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
        if size == 0:
            return np.empty((0, features.shape[1]), dtype=np.float64)
        replace = features.shape[0] < size
        indices = rng.choice(features.shape[0], size=size, replace=replace)
        return features[indices]

    @staticmethod
    def _validate_features(features: np.ndarray, name: str) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"Expected {name} to be a 2D array, got shape {features.shape}.")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError(f"Expected {name} to be non-empty, got shape {features.shape}.")
        return features
