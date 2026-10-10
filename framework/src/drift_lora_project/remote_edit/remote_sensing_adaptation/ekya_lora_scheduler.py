from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Any, Iterable


@dataclass(frozen=True)
class LambdaProfile:
    lambda_id: str
    image_resolution: int
    max_new_tokens: int
    model_precision: str
    batch_size: int
    accuracy: float
    resource_cost: float
    token_retention_ratio: float = 1.0
    roi_strategy: str = "none"
    peak_memory_gib: float | None = None
    power_watts: float | None = None
    thermal_score: float | None = None


@dataclass(frozen=True)
class GammaProfile:
    gamma_id: str
    method: str
    lora_rank: int
    lora_alpha: int
    target_modules: str
    learning_rate: float
    batch_size: int
    grad_accum: int
    train_steps: int
    train_data_fraction: float
    profile_gain: float
    predicted_final_gain: float
    resource_cost: float
    train_time: float
    peak_memory_gib: float | None = None
    power_watts: float | None = None
    thermal_score: float | None = None


@dataclass(frozen=True)
class SchedulerSettings:
    retraining_window_sec: float
    data_arrival_rate: float = 1.0
    future_windows: int = 3
    total_resource: float = 1.0
    steal_increment: float = 0.01
    initial_inference_resource: float = 0.5
    min_accuracy: float = 0.0
    validation_epsilon: float = 0.0
    max_accuracy: float = 1.0
    max_iterations: int = 200
    discount_factor: float = 1.0
    max_peak_memory_gib: float | None = None
    max_power_watts: float | None = None
    max_thermal_score: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1")


@dataclass(frozen=True)
class ConfigDecision:
    lambda_profile: LambdaProfile
    gamma_profile: GammaProfile | None
    inference_resource: float
    training_resource: float
    current_window_accuracy: float
    future_window_accuracy: float
    long_avg_accuracy: float
    finish_time: float | None
    adapter_replaced: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class ThiefSchedulerResult:
    lambda_profile: LambdaProfile
    gamma_profile: GammaProfile | None
    inference_resource: float
    training_resource: float
    current_window_accuracy: float
    future_window_accuracy: float
    long_avg_accuracy: float
    initial_long_avg_accuracy: float
    finish_time: float | None
    adapter_replaced: bool
    settings: SchedulerSettings
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def optimus_accuracy(x: float, b0: float, b1: float, b2: float) -> float:
    return 1.0 - (1.0 / (b0 * x + b1) + b2)


def fit_optimus_curve(
    points: Iterable[tuple[float, float]],
    max_accuracy: float = 1.0,
) -> Any:
    """Fit an Ekya/Optimus-style saturating curve from early profile points.

    Ekya uses an Optimus-style curve, A(x)=1-(1/(b0*x+b1)+b2), to extrapolate
    short micro-profile measurements. This lightweight fit keeps that curve
    shape while avoiding a SciPy dependency: it anchors the initial point and
    searches a small saturation family that minimizes squared error.
    """

    ordered = sorted((float(x), float(y)) for x, y in points)
    if len(ordered) < 2:
        raise ValueError("At least two profile points are required")
    x0, y0 = ordered[0]
    if x0 != 0:
        ordered.insert(0, (0.0, y0))
        x0 = 0.0
    y0 = ordered[0][1]
    cap = max(y0, min(max_accuracy, 1.0))
    if cap <= y0:
        return lambda x: y0

    best_scale = None
    best_error = float("inf")
    last_x = max(x for x, _ in ordered)
    scales = [max(last_x / div, 1e-6) for div in (8, 4, 2, 1, 0.5, 0.25, 0.125)]
    scales += [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for scale in sorted(set(scales)):
        error = 0.0
        for x, y in ordered:
            pred = y0 + (cap - y0) * (x / (x + scale)) if x > 0 else y0
            error += (pred - y) ** 2
        if error < best_error:
            best_error = error
            best_scale = scale

    assert best_scale is not None

    def curve(x: float) -> float:
        x = max(0.0, float(x))
        if x == 0:
            return y0
        pred = y0 + (cap - y0) * (x / (x + best_scale))
        return max(0.0, min(cap, pred))

    return curve


def choose_inference_config(
    lambdas: list[LambdaProfile],
    inference_resource: float,
    min_accuracy: float,
    settings: SchedulerSettings | None = None,
) -> LambdaProfile | None:
    candidates = [
        profile
        for profile in lambdas
        if (
            profile.resource_cost <= inference_resource
            and profile.accuracy >= min_accuracy
            and profile_pair_within_limits(profile, None, settings)
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda profile: (profile.accuracy, -profile.resource_cost))


def _sum_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return sum(values)


def _under_limit(value: float | None, limit: float | None) -> bool:
    return limit is None or value is None or value <= limit


def profile_pair_within_limits(
    lambda_profile: LambdaProfile,
    gamma_profile: GammaProfile | None,
    settings: SchedulerSettings | None,
) -> bool:
    if settings is None:
        return True
    peak_memory_gib = _sum_optional(
        lambda_profile.peak_memory_gib,
        gamma_profile.peak_memory_gib if gamma_profile else None,
    )
    power_watts = _sum_optional(
        lambda_profile.power_watts,
        gamma_profile.power_watts if gamma_profile else None,
    )
    thermal_score = _sum_optional(
        lambda_profile.thermal_score,
        gamma_profile.thermal_score if gamma_profile else None,
    )
    return (
        _under_limit(peak_memory_gib, settings.max_peak_memory_gib)
        and _under_limit(power_watts, settings.max_power_watts)
        and _under_limit(thermal_score, settings.max_thermal_score)
    )


def discounted_long_average(
    current_accuracy: float,
    future_accuracy: float,
    future_windows: int,
    discount_factor: float,
) -> float:
    weights = [1.0]
    weights.extend(discount_factor**window for window in range(1, max(0, future_windows) + 1))
    numerator = weights[0] * current_accuracy + sum(weight * future_accuracy for weight in weights[1:])
    return numerator / sum(weights)


def estimate_accuracy(
    lambda_profile: LambdaProfile,
    gamma_profile: GammaProfile | None,
    inference_resource: float,
    training_resource: float,
    settings: SchedulerSettings,
) -> ConfigDecision:
    coverage = min(1.0, inference_resource / max(lambda_profile.resource_cost, 1e-9))
    base_accuracy = lambda_profile.accuracy * coverage

    if (
        gamma_profile is None
        or training_resource < gamma_profile.resource_cost
        or not profile_pair_within_limits(lambda_profile, gamma_profile, settings)
    ):
        return ConfigDecision(
            lambda_profile=lambda_profile,
            gamma_profile=None,
            inference_resource=round(inference_resource, 6),
            training_resource=round(training_resource, 6),
            current_window_accuracy=round(base_accuracy, 6),
            future_window_accuracy=round(base_accuracy, 6),
            long_avg_accuracy=round(base_accuracy, 6),
            finish_time=None,
            adapter_replaced=False,
        )

    finish_time = gamma_profile.train_time / max(training_resource, 1e-9)
    if finish_time > settings.retraining_window_sec:
        return ConfigDecision(
            lambda_profile=lambda_profile,
            gamma_profile=gamma_profile,
            inference_resource=round(inference_resource, 6),
            training_resource=round(training_resource, 6),
            current_window_accuracy=round(base_accuracy, 6),
            future_window_accuracy=round(base_accuracy, 6),
            long_avg_accuracy=round(base_accuracy, 6),
            finish_time=round(finish_time, 6) if isfinite(finish_time) else None,
            adapter_replaced=False,
        )

    updated_accuracy = min(
        settings.max_accuracy,
        lambda_profile.accuracy + gamma_profile.predicted_final_gain,
    ) * coverage
    adapter_replaced = updated_accuracy >= base_accuracy + settings.validation_epsilon
    if not adapter_replaced:
        updated_accuracy = base_accuracy

    current_ratio_old = max(0.0, min(1.0, finish_time / settings.retraining_window_sec))
    current_accuracy = current_ratio_old * base_accuracy + (1.0 - current_ratio_old) * updated_accuracy
    future_accuracy = updated_accuracy
    long_avg = discounted_long_average(
        current_accuracy=current_accuracy,
        future_accuracy=future_accuracy,
        future_windows=settings.future_windows,
        discount_factor=settings.discount_factor,
    )
    return ConfigDecision(
        lambda_profile=lambda_profile,
        gamma_profile=gamma_profile,
        inference_resource=round(inference_resource, 6),
        training_resource=round(training_resource, 6),
        current_window_accuracy=round(current_accuracy, 6),
        future_window_accuracy=round(future_accuracy, 6),
        long_avg_accuracy=round(long_avg, 6),
        finish_time=round(finish_time, 6),
        adapter_replaced=adapter_replaced,
    )


def pick_configs(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    inference_resource: float,
    training_resource: float,
    settings: SchedulerSettings,
) -> ConfigDecision:
    lambda_candidates = [
        profile
        for profile in lambdas
        if (
            profile.resource_cost <= inference_resource
            and profile.accuracy >= settings.min_accuracy
            and profile_pair_within_limits(profile, None, settings)
        )
    ]
    if not lambda_candidates:
        raise ValueError(
            f"No inference profile can keep up with resource I={inference_resource:.4f}"
        )

    candidates: list[ConfigDecision] = []
    for lambda_profile in lambda_candidates:
        candidates.append(
            estimate_accuracy(
                lambda_profile=lambda_profile,
                gamma_profile=None,
                inference_resource=inference_resource,
                training_resource=training_resource,
                settings=settings,
            )
        )
        for gamma_profile in gammas:
            candidates.append(
                estimate_accuracy(
                    lambda_profile=lambda_profile,
                    gamma_profile=gamma_profile,
                    inference_resource=inference_resource,
                    training_resource=training_resource,
                    settings=settings,
                )
            )

    return max(
        candidates,
        key=lambda decision: (
            decision.long_avg_accuracy,
            decision.future_window_accuracy,
            decision.current_window_accuracy,
            decision.lambda_profile.accuracy,
            -decision.lambda_profile.resource_cost,
            decision.training_resource,
        ),
    )


def run_thief_scheduler(
    lambdas: list[LambdaProfile],
    gammas: list[GammaProfile],
    settings: SchedulerSettings,
) -> ThiefSchedulerResult:
    if settings.steal_increment <= 0:
        raise ValueError("steal_increment must be positive")
    if settings.total_resource <= 0:
        raise ValueError("total_resource must be positive")

    inference_resource = settings.initial_inference_resource
    training_resource = settings.total_resource - inference_resource
    initial = pick_configs(lambdas, gammas, inference_resource, training_resource, settings)
    initial_score = initial.long_avg_accuracy
    trace: list[dict[str, Any]] = [
        {"iteration": 0, "action": "initial", "decision": initial.to_dict()}
    ]

    steps = int(ceil(settings.total_resource / settings.steal_increment))
    grid = {
        0.0,
        settings.total_resource,
        settings.initial_inference_resource,
    }
    for lambda_profile in lambdas:
        if 0.0 <= lambda_profile.resource_cost <= settings.total_resource:
            grid.add(lambda_profile.resource_cost)
    for gamma_profile in gammas:
        candidate_i = settings.total_resource - gamma_profile.resource_cost
        if 0.0 <= candidate_i <= settings.total_resource:
            grid.add(candidate_i)
    for step in range(steps + 1):
        grid.add(round(min(settings.total_resource, step * settings.steal_increment), 6))

    candidates: list[ConfigDecision] = [initial]
    for iteration, candidate_i in enumerate(sorted(grid), start=1):
        candidate_i = min(settings.total_resource, max(0.0, candidate_i))
        candidate_r = settings.total_resource - candidate_i
        if candidate_i < -1e-9 or candidate_r < -1e-9:
            continue
        try:
            decision = pick_configs(
                lambdas,
                gammas,
                candidate_i,
                candidate_r,
                settings,
            )
        except ValueError:
            trace.append(
                {
                    "iteration": iteration,
                    "action": "skip_infeasible",
                    "inference_resource": candidate_i,
                    "training_resource": candidate_r,
                }
            )
            continue
        candidates.append(decision)
        trace.append(
            {
                "iteration": iteration,
                "action": "evaluate_grid",
                "decision": decision.to_dict(),
            }
        )

    current = max(
        candidates,
        key=lambda decision: (
            decision.long_avg_accuracy,
            decision.future_window_accuracy,
            decision.current_window_accuracy,
            decision.training_resource,
        ),
    )
    trace.append({"iteration": len(trace), "action": "select_best", "decision": current.to_dict()})

    return ThiefSchedulerResult(
        lambda_profile=current.lambda_profile,
        gamma_profile=current.gamma_profile,
        inference_resource=current.inference_resource,
        training_resource=current.training_resource,
        current_window_accuracy=current.current_window_accuracy,
        future_window_accuracy=current.future_window_accuracy,
        long_avg_accuracy=current.long_avg_accuracy,
        initial_long_avg_accuracy=initial_score,
        finish_time=current.finish_time,
        adapter_replaced=current.adapter_replaced,
        settings=settings,
        trace=trace,
    )


def infer_retraining_window(train_time_ref: float, initial_training_resource: float = 0.5, margin: float = 1.2) -> int:
    if train_time_ref <= 0:
        raise ValueError("train_time_ref must be positive")
    if initial_training_resource <= 0:
        raise ValueError("initial_training_resource must be positive")
    return int(ceil((train_time_ref / initial_training_resource) * margin))
