from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class WindowStats:
    mean: np.ndarray
    covariance: np.ndarray


class FDDDetector:
    """Window-level FDD detector with PCA baseline modeling and Monte Carlo calibration."""

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
        self.sqrt_sigma_base: np.ndarray | None = None
        self.trace_sigma_base: float | None = None
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

        stats = self._compute_window_stats(projected)
        self.mu_base = stats.mean
        self.sigma_base = stats.covariance
        self.sqrt_sigma_base = self._matrix_sqrt_psd(self.sigma_base)
        self.trace_sigma_base = float(np.trace(self.sigma_base))

    def calibrate_threshold(self, features: np.ndarray) -> float:
        transformed = self.transform(features)
        rng = np.random.default_rng(self.random_seed)
        scores = np.empty(self.calibration_windows, dtype=np.float64)

        for index in range(self.calibration_windows):
            window = self._sample_window(transformed, self.window_size, rng)
            scores[index] = self._fdd_from_projected(window)

        self.calibration_scores = scores
        self.threshold = float(np.quantile(scores, self.calibration_quantile))
        return self.threshold

    def score(self, features: np.ndarray) -> float:
        transformed = self.transform(features)
        if transformed.shape[0] != self.window_size:
            raise ValueError(
                f"Expected a window with {self.window_size} samples, got {transformed.shape[0]} samples."
            )
        return self._fdd_from_projected(transformed)

    def score_projected_window(self, projected_window: np.ndarray) -> float:
        projected_window = self._validate_features(projected_window, name="projected_window")
        if projected_window.shape[0] != self.window_size:
            raise ValueError(
                f"Expected a projected window with {self.window_size} samples, got {projected_window.shape[0]} samples."
            )
        return self._fdd_from_projected(projected_window)

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
    ) -> list[dict[str, float]]:
        no_drift_projected = self.transform(no_drift_features)
        drift_projected = self.transform(drift_features)
        rng = np.random.default_rng(self.random_seed)
        results: list[dict[str, float]] = []

        for ratio in drift_ratios:
            detections = 0
            n_drift = int(round(self.window_size * ratio))
            n_clean = self.window_size - n_drift
            scores = np.empty(windows_per_ratio, dtype=np.float64)

            for index in range(windows_per_ratio):
                clean_window = self._sample_window(no_drift_projected, n_clean, rng) if n_clean > 0 else None
                drift_window = self._sample_window(drift_projected, n_drift, rng) if n_drift > 0 else None

                parts = [part for part in (clean_window, drift_window) if part is not None]
                window = np.concatenate(parts, axis=0)
                window = window[rng.permutation(window.shape[0])]

                score = self._fdd_from_projected(window)
                scores[index] = score
                if self.threshold is not None and score > self.threshold:
                    detections += 1

            detection_rate = detections / windows_per_ratio
            results.append(
                {
                    "drift_ratio": float(ratio),
                    "windows": int(windows_per_ratio),
                    "n_clean": int(n_clean),
                    "n_drift": int(n_drift),
                    "detection_rate": float(detection_rate),
                    "metric_name": "fpr" if ratio == 0 else "recall",
                    "mean_fdd": float(scores.mean()),
                    "std_fdd": float(scores.std(ddof=0)),
                }
            )

        return results

    def save_artifacts(self, output_dir: str | Path, pca_filename: str, baseline_filename: str, calibration_filename: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.pca is None or self.mu_base is None or self.sigma_base is None or self.sqrt_sigma_base is None:
            raise RuntimeError("Cannot save artifacts before fitting the reference model.")

        joblib.dump(self.pca, output_path / pca_filename)
        np.savez(
            output_path / baseline_filename,
            mu_base=self.mu_base,
            sigma_base=self.sigma_base,
            sqrt_sigma_base=self.sqrt_sigma_base,
            trace_sigma_base=np.array(self.trace_sigma_base, dtype=np.float64),
            actual_pca_dim=np.array(self.actual_pca_dim, dtype=np.int64),
            requested_pca_dim=np.array(self.requested_pca_dim, dtype=np.int64),
            window_size=np.array(self.window_size, dtype=np.int64),
            cov_eps=np.array(self.cov_eps, dtype=np.float64),
        )
        np.savez(
            output_path / calibration_filename,
            threshold=np.array(self.threshold if self.threshold is not None else np.nan, dtype=np.float64),
            calibration_scores=self.calibration_scores if self.calibration_scores is not None else np.array([], dtype=np.float64),
            calibration_quantile=np.array(self.calibration_quantile, dtype=np.float64),
            calibration_windows=np.array(self.calibration_windows, dtype=np.int64),
        )

    def _fdd_from_projected(self, projected_window: np.ndarray) -> float:
        if self.mu_base is None or self.sigma_base is None or self.sqrt_sigma_base is None or self.trace_sigma_base is None:
            raise RuntimeError("Baseline statistics are not initialized.")

        window_stats = self._compute_window_stats(projected_window)
        mean_diff = self.mu_base - window_stats.mean
        mean_term = float(mean_diff @ mean_diff)

        middle = self.sqrt_sigma_base @ window_stats.covariance @ self.sqrt_sigma_base
        middle = self._symmetrize(middle)
        trace_middle_sqrt = float(np.trace(self._matrix_sqrt_psd(middle)))
        cov_term = self.trace_sigma_base + float(np.trace(window_stats.covariance)) - 2.0 * trace_middle_sqrt
        return float(mean_term + max(cov_term, 0.0))

    def _compute_window_stats(self, projected_window: np.ndarray) -> WindowStats:
        projected_window = self._validate_features(projected_window, name="window")
        mean = projected_window.mean(axis=0)
        covariance = np.cov(projected_window, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]], dtype=np.float64)
        covariance = self._symmetrize(np.asarray(covariance, dtype=np.float64))
        covariance += np.eye(covariance.shape[0], dtype=np.float64) * self.cov_eps
        return WindowStats(mean=mean.astype(np.float64), covariance=covariance)

    def _sample_window(self, features: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if count == 0:
            return np.empty((0, features.shape[1]), dtype=features.dtype)
        replace = features.shape[0] < count
        indices = rng.choice(features.shape[0], size=count, replace=replace)
        return features[indices]

    @staticmethod
    def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
        matrix = FDDDetector._symmetrize(np.asarray(matrix, dtype=np.float64))
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        clipped = np.clip(eigenvalues, a_min=0.0, a_max=None)
        sqrt_diag = np.sqrt(clipped)
        return (eigenvectors * sqrt_diag) @ eigenvectors.T

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return (matrix + matrix.T) * 0.5

    @staticmethod
    def _validate_features(features: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{name} cannot be empty.")
        return array
