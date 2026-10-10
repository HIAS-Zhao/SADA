from __future__ import annotations

import copy
import math
import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


class Detector:
    drift_detected: bool

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, value: float) -> bool:
        raise NotImplementedError

    def clone(self) -> "Detector":
        return copy.deepcopy(self)


class DDM(Detector):
    def __init__(self, warm_start: int = 30, warning_threshold: float = 2.0, drift_threshold: float = 3.0):
        self.warm_start = warm_start
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.p_min: float | None = None
        self.s_min: float | None = None
        self.ps_min = float("inf")
        self.warning_detected = False
        self.drift_detected = False

    def update(self, value: float) -> bool:
        x = float(value)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        std = math.sqrt(max(self.mean * (1.0 - self.mean) / self.n, 0.0))
        self.warning_detected = False
        self.drift_detected = False

        if self.n > self.warm_start:
            if self.mean + std <= self.ps_min:
                self.p_min = self.mean
                self.s_min = std
                self.ps_min = self.mean + std
            if self.p_min is not None and self.s_min is not None:
                self.warning_detected = (
                    self.mean + std > self.p_min + self.warning_threshold * self.s_min
                )
                self.drift_detected = (
                    self.mean + std > self.p_min + self.drift_threshold * self.s_min
                )
                if self.drift_detected:
                    self.warning_detected = False
        return self.drift_detected


class RunningVariance:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


class EDDM(Detector):
    def __init__(self, warm_start: int = 30, alpha: float = 0.95, beta: float = 0.9):
        if alpha < beta:
            raise ValueError("alpha must be greater than or equal to beta")
        self.warm_start = warm_start
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.last_error = 0
        self.n_errors = 0
        self.distances = RunningVariance()
        self.max_score = -1.0
        self.warning_detected = False
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.n += 1
        self.warning_detected = False
        self.drift_detected = False
        if int(value) != 1:
            return False

        self.n_errors += 1
        self.distances.update(float(self.n - self.last_error))
        self.last_error = self.n
        if self.n <= self.warm_start or self.distances.n < 2:
            return False

        score = self.distances.mean + 2.0 * math.sqrt(max(self.distances.variance, 0.0))
        if score > self.max_score:
            self.max_score = score
        elif self.n_errors > self.warm_start and self.max_score > 0:
            level = score / self.max_score
            if level < self.beta:
                self.drift_detected = True
            elif level < self.alpha:
                self.warning_detected = True
        return self.drift_detected


@dataclass
class _EWMASample:
    lambda_val: float
    ewma: float = 0.0
    ibc: float = 1.0
    initialized: bool = False

    def update(self, value: float) -> None:
        if not self.initialized:
            self.ewma = float(value)
        else:
            self.ewma = self.lambda_val * float(value) + (1.0 - self.lambda_val) * self.ewma
        self.initialized = True
        self.ibc = self.lambda_val**2 + (1.0 - self.lambda_val) ** 2 * self.ibc


class HDDMW(Detector):
    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        lambda_val: float = 0.05,
    ):
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_val = lambda_val
        self.reset()

    def reset(self) -> None:
        self.total = _EWMASample(self.lambda_val)
        self.s1 = _EWMASample(self.lambda_val)
        self.s2 = _EWMASample(self.lambda_val)
        self.cutpoint = float("inf")
        self.warning_detected = False
        self.drift_detected = False

    @staticmethod
    def _bound(ibc: float, confidence: float) -> float:
        return math.sqrt(max(ibc, 0.0) * math.log(1.0 / confidence) / 2.0)

    def _changed(self, confidence: float) -> bool:
        if not (self.s1.initialized and self.s2.initialized):
            return False
        bound = self._bound(self.s1.ibc + self.s2.ibc, confidence)
        return self.s2.ewma - self.s1.ewma > bound

    def update(self, value: float) -> bool:
        x = float(value)
        self.total.update(x)
        epsilon = self._bound(self.total.ibc, self.drift_confidence)
        if self.total.ewma + epsilon < self.cutpoint:
            self.cutpoint = self.total.ewma + epsilon
            self.s1 = copy.deepcopy(self.total)
            self.s2 = _EWMASample(self.lambda_val)
        else:
            self.s2.update(x)

        self.drift_detected = self._changed(self.drift_confidence)
        self.warning_detected = not self.drift_detected and self._changed(self.warning_confidence)
        return self.drift_detected


class PageHinkley(Detector):
    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
    ):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        if mode not in {"up", "down", "both"}:
            raise ValueError("mode must be one of: up, down, both")
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.sum_increase = 0.0
        self.sum_decrease = 0.0
        self.minimum_increase = float("inf")
        self.maximum_decrease = -float("inf")
        self.drift_detected = False

    def update(self, value: float) -> bool:
        x = float(value)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        deviation = x - self.mean
        self.sum_increase = self.alpha * self.sum_increase + deviation - self.delta
        self.sum_decrease = self.alpha * self.sum_decrease + deviation + self.delta
        self.minimum_increase = min(self.minimum_increase, self.sum_increase)
        self.maximum_decrease = max(self.maximum_decrease, self.sum_decrease)
        increase_test = self.sum_increase - self.minimum_increase
        decrease_test = self.maximum_decrease - self.sum_decrease
        if self.mode == "up":
            changed = increase_test > self.threshold
        elif self.mode == "down":
            changed = decrease_test > self.threshold
        else:
            changed = increase_test > self.threshold or decrease_test > self.threshold
        self.drift_detected = self.n >= self.min_instances and changed
        return self.drift_detected


class KSWIN(Detector):
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 42,
        clock: int = 5,
    ):
        if stat_size >= window_size:
            raise ValueError("stat_size must be smaller than window_size")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.window: deque[float] = deque(maxlen=self.window_size)
        self.n = 0
        self.p_value = 1.0
        self.rng = random.Random(self.seed)
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.n += 1
        self.window.append(float(value))
        self.drift_detected = False
        if len(self.window) < self.window_size or self.n % self.clock:
            return False

        values = list(self.window)
        reference_end = self.window_size - self.stat_size
        reference_indices = self.rng.sample(range(reference_end), self.stat_size)
        reference = [values[index] for index in reference_indices]
        recent = values[-self.stat_size :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            statistic, self.p_value = stats.ks_2samp(reference, recent, method="auto")
        if self.p_value <= self.alpha and statistic > 0.1:
            self.drift_detected = True
            self.window = deque(recent, maxlen=self.window_size)
        return self.drift_detected


class ADWIN(Detector):
    """Compact ADWIN-style detector using the paper's adaptive-window bound."""

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 16,
        min_window_length: int = 10,
        grace_period: int = 30,
        max_window: int = 400,
        cut_step: int = 5,
        two_sided: bool = True,
    ):
        self.delta = delta
        self.clock = clock
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self.max_window = max_window
        self.cut_step = cut_step
        self.two_sided = two_sided
        self.reset()

    def reset(self) -> None:
        self.window: deque[float] = deque(maxlen=self.max_window)
        self.n = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.n += 1
        self.window.append(float(value))
        self.drift_detected = False
        n = len(self.window)
        if n < self.grace_period or self.n % self.clock:
            return False

        values = np.asarray(self.window, dtype=np.float64)
        variance = float(np.var(values))
        log_term = math.log(max(2.0 * math.log(max(n, 2)) / self.delta, 1.0000001))
        for cut in range(self.min_window_length, n - self.min_window_length + 1, self.cut_step):
            left = values[:cut]
            right = values[cut:]
            mean_diff = float(np.mean(right) - np.mean(left))
            test_difference = abs(mean_diff) if self.two_sided else mean_diff
            harmonic = 1.0 / len(left) + 1.0 / len(right)
            epsilon = math.sqrt(2.0 * variance * harmonic * log_term) + (
                2.0 * harmonic * log_term / 3.0
            )
            if test_difference > epsilon:
                self.drift_detected = True
                self.window = deque(right.tolist(), maxlen=self.max_window)
                break
        return self.drift_detected


class OPTWIN(Detector):
    """Paper-faithful OPTWIN reproduction using one-sided t- and F-tests.

    The implementation searches admissible sub-window cuts directly. The earliest
    statistically significant cut is retained as the new reference window.
    """

    def __init__(
        self,
        delta: float = 0.001,
        rigor: float = 0.5,
        max_window: int = 200,
        min_subwindow: int = 20,
        clock: int = 5,
        cut_step: int = 5,
        use_variance_test: bool = True,
        two_sided: bool = False,
    ):
        self.delta = delta
        self.rigor = rigor
        self.max_window = max_window
        self.min_subwindow = min_subwindow
        self.clock = clock
        self.cut_step = cut_step
        self.use_variance_test = use_variance_test
        self.two_sided = two_sided
        self.reset()

    def reset(self) -> None:
        self.window: deque[float] = deque(maxlen=self.max_window)
        self.n = 0
        self.drift_detected = False
        self.cut_index: int | None = None

    def update(self, value: float) -> bool:
        self.n += 1
        self.window.append(float(value))
        self.drift_detected = False
        self.cut_index = None
        n = len(self.window)
        if n < 2 * self.min_subwindow or self.n % self.clock:
            return False

        values = np.asarray(self.window, dtype=np.float64)
        cuts = np.arange(
            self.min_subwindow,
            n - self.min_subwindow + 1,
            self.cut_step,
            dtype=np.int64,
        )
        if cuts.size == 0:
            return False

        prefix_sum = np.concatenate(([0.0], np.cumsum(values)))
        prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))
        old_n = cuts.astype(np.float64)
        recent_n = (n - cuts).astype(np.float64)
        old_sum = prefix_sum[cuts]
        recent_sum = prefix_sum[n] - old_sum
        old_mean = old_sum / old_n
        recent_mean = recent_sum / recent_n
        increase = recent_mean - old_mean

        old_sq = prefix_sq[cuts]
        recent_sq = prefix_sq[n] - old_sq
        old_var = np.maximum((old_sq - old_sum * old_sum / old_n) / np.maximum(old_n - 1.0, 1.0), 0.0)
        recent_var = np.maximum(
            (recent_sq - recent_sum * recent_sum / recent_n) / np.maximum(recent_n - 1.0, 1.0),
            0.0,
        )
        standard_error_sq = old_var / old_n + recent_var / recent_n
        valid_t = standard_error_sq > 1e-12
        t_stat = np.zeros_like(increase)
        t_stat[valid_t] = increase[valid_t] / np.sqrt(standard_error_sq[valid_t])
        numerator = standard_error_sq * standard_error_sq
        denominator = (
            (old_var / old_n) ** 2 / np.maximum(old_n - 1.0, 1.0)
            + (recent_var / recent_n) ** 2 / np.maximum(recent_n - 1.0, 1.0)
        )
        degrees = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 1.0),
            where=denominator > 1e-18,
        )
        t_pvalues = np.ones_like(increase)
        if self.two_sided:
            t_pvalues[valid_t] = 2.0 * stats.t.sf(np.abs(t_stat[valid_t]), degrees[valid_t])
        else:
            t_pvalues[valid_t] = stats.t.sf(t_stat[valid_t], degrees[valid_t])

        valid_f = (old_var > 1e-12) & (recent_var > 1e-12)
        f_pvalues = np.ones_like(increase)
        variance_ratio = np.ones_like(increase)
        variance_ratio[valid_f] = recent_var[valid_f] / old_var[valid_f]
        upper_tail = np.ones_like(increase)
        upper_tail[valid_f] = stats.f.sf(
            variance_ratio[valid_f],
            recent_n[valid_f] - 1.0,
            old_n[valid_f] - 1.0,
        )
        if self.two_sided:
            lower_tail = np.ones_like(increase)
            lower_tail[valid_f] = stats.f.cdf(
                variance_ratio[valid_f],
                recent_n[valid_f] - 1.0,
                old_n[valid_f] - 1.0,
            )
            f_pvalues[valid_f] = np.minimum(
                1.0,
                2.0 * np.minimum(upper_tail[valid_f], lower_tail[valid_f]),
            )
        else:
            f_pvalues[valid_f] = upper_tail[valid_f]

        checks_per_window = max(math.ceil(self.max_window / self.clock), 1)
        alpha = self.delta / (cuts.size * checks_per_window)
        tested_effect = np.abs(increase) if self.two_sided else increase
        effect_floor = self.rigor * np.sqrt(np.maximum(old_var, 1e-12)) / np.sqrt(recent_n)
        test_passed = t_pvalues <= alpha
        if self.use_variance_test:
            test_passed |= f_pvalues <= alpha
        significant = (
            (tested_effect > 0.0)
            & (tested_effect >= effect_floor)
            & test_passed
        )
        if np.any(significant):
            first = int(np.flatnonzero(significant)[0])
            cut = int(cuts[first])
            self.drift_detected = True
            self.cut_index = cut
            self.window = deque(values[cut:].tolist(), maxlen=self.max_window)
        return self.drift_detected


def warm_detector(detector: Detector, values: Iterable[float]) -> Detector:
    for value in values:
        if detector.update(float(value)):
            detector.reset()
    detector.drift_detected = False
    return detector


def detects_in_window(warmed_detector: Detector, values: Iterable[float]) -> tuple[bool, int | None]:
    detector = warmed_detector.clone()
    for index, value in enumerate(values):
        if detector.update(float(value)):
            return True, index
    return False, None


def detects_after_onset(
    warmed_detector: Detector,
    clean_prefix: Iterable[float],
    drift_suffix: Iterable[float],
) -> tuple[bool, int | None, int]:
    detector = warmed_detector.clone()
    prechange_alarms = 0
    for value in clean_prefix:
        if detector.update(float(value)):
            prechange_alarms += 1
            detector.reset()

    for index, value in enumerate(drift_suffix):
        if detector.update(float(value)):
            return True, index, prechange_alarms
    return False, None, prechange_alarms
