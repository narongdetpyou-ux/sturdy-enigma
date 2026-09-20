#!/usr/bin/env python3
"""Black Swan Logic v7 prototype.

The engine is domain-neutral.  A caller supplies metric histories, a recent
window, data-quality states, lifecycle facts, and independently sourced context.
Ground-truth labels are deliberately absent from the prediction interface.

v7 adds four safeguards to the v6-style point detector:
1. holdout-calibrated family-wise evidence threshold;
2. temporal accumulation for low-and-slow deviations;
3. cross-metric evidence fusion for correlated weak signals;
4. a critic pass that validates quality, lifecycle, and context scope before the
   action policy is allowed to emit a security alert.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


COMPLETE = "complete"
MISSING_STATES = {"missing", "failed", "suppressed", "not_published"}


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    xs = sorted(float(value) for value in values)
    probability = min(1.0, max(0.0, probability))
    position = (len(xs) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return xs[lower]
    fraction = position - lower
    return xs[lower] + fraction * (xs[upper] - xs[lower])


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _robust_scale(values: Sequence[float], center: Optional[float] = None) -> float:
    """Return a stable MAD scale with defensible fallbacks for discrete metrics."""
    if not values:
        return 1.0
    center = _median(values) if center is None else float(center)
    mad = _median([abs(value - center) for value in values])
    scale = 1.4826 * mad
    if scale <= 1e-9 and len(values) > 1:
        scale = statistics.pstdev(values)
    magnitude_floor = max(1e-6, abs(center) * 0.01)
    return max(scale, magnitude_floor)


@dataclass(frozen=True)
class ContextEvent:
    event_key: str
    covered_metrics: Tuple[str, ...]
    approved: bool = True
    active: bool = True
    confidence: float = 0.99
    evidence_ref: str = "synthetic_change_record"


@dataclass
class ScenarioInput:
    scenario_id: str
    history: Dict[str, List[Optional[float]]]
    recent: Dict[str, List[Optional[float]]]
    history_status: Dict[str, List[str]] = field(default_factory=dict)
    recent_status: Dict[str, List[str]] = field(default_factory=dict)
    lifecycle: Dict[str, str] = field(default_factory=dict)
    context_events: List[ContextEvent] = field(default_factory=list)


@dataclass(frozen=True)
class Calibration:
    composite_threshold: float
    quantile: float
    calibration_size: int
    target_false_alert_rate: float
    base_point: float = 3.0
    base_temporal: float = 8.0
    base_collective: float = 5.0
    review_fraction: float = 0.80
    context_min_confidence: float = 0.95
    min_history: int = 72
    max_fused_metrics: int = 4


@dataclass
class MetricEvidence:
    metric: str
    history_n: int
    recent_n: int
    expected_last: Optional[float]
    observed_last: Optional[float]
    point_score: float
    last_point_score: float
    temporal_score: float
    max_relative_deviation: float
    last_relative_deviation: float
    quality_ok: bool
    statuses: Tuple[str, ...]


@dataclass
class EvidenceFeatures:
    metric_evidence: List[MetricEvidence]
    point_score: float
    temporal_score: float
    collective_score: float
    composite_score: float
    driver_metrics: Tuple[str, ...]
    quality_issues: Dict[str, Tuple[str, ...]]
    insufficient_history_metrics: Tuple[str, ...]


@dataclass
class Decision:
    scenario_id: str
    engine_version: str
    initial_route: str
    final_route: str
    security_alert: bool
    operational_escalation: bool
    composite_score: float
    calibrated_threshold: float
    evidence_margin: float
    uncertainty: float
    point_score: float
    temporal_score: float
    collective_score: float
    dominant_channel: str
    driver_metrics: Tuple[str, ...]
    correction_applied: bool
    correction_reason: str
    context_keys_considered: Tuple[str, ...]
    quality_issues: Dict[str, Tuple[str, ...]]
    lifecycle: Dict[str, str]
    ledger: List[dict]

    def as_record(self) -> dict:
        record = asdict(self)
        record["driver_metrics"] = "|".join(self.driver_metrics)
        record["context_keys_considered"] = "|".join(self.context_keys_considered)
        record["quality_issues"] = "|".join(
            f"{metric}:{','.join(states)}" for metric, states in self.quality_issues.items()
        )
        record["lifecycle"] = "|".join(
            f"{metric}:{state}" for metric, state in self.lifecycle.items()
        )
        return record


def _status_vector(
    supplied: Mapping[str, List[str]], metric: str, count: int
) -> List[str]:
    statuses = list(supplied.get(metric, []))
    if not statuses:
        return [COMPLETE] * count
    if len(statuses) != count:
        raise ValueError(f"status length mismatch for {metric}: {len(statuses)} != {count}")
    return statuses


def _seasonal_profile(values: Sequence[float], period: int = 24) -> Dict[int, float]:
    global_median = _median(values)
    profile: Dict[int, float] = {}
    for phase in range(period):
        bucket = [value for index, value in enumerate(values) if index % period == phase]
        profile[phase] = _median(bucket) if bucket else global_median
    return profile


def _metric_features(
    metric: str,
    history: Sequence[Optional[float]],
    recent: Sequence[Optional[float]],
    history_status: Sequence[str],
    recent_status: Sequence[str],
    *,
    min_history: int,
) -> Tuple[MetricEvidence, List[float]]:
    quality_ok = all(status == COMPLETE for status in recent_status)
    complete_history = [
        float(value)
        for value, status in zip(history, history_status)
        if value is not None and status == COMPLETE
    ]
    complete_recent = [
        float(value)
        for value, status in zip(recent, recent_status)
        if value is not None and status == COMPLETE
    ]
    statuses = tuple(sorted(set(recent_status)))

    if len(complete_history) < min_history or not complete_recent or not quality_ok:
        evidence = MetricEvidence(
            metric=metric,
            history_n=len(complete_history),
            recent_n=len(complete_recent),
            expected_last=None,
            observed_last=complete_recent[-1] if complete_recent else None,
            point_score=0.0,
            last_point_score=0.0,
            temporal_score=0.0,
            max_relative_deviation=0.0,
            last_relative_deviation=0.0,
            quality_ok=quality_ok,
            statuses=statuses,
        )
        return evidence, []

    # Histories are contiguous hourly observations.  Use hour-of-cycle medians,
    # then one pooled robust residual scale.  This prevents daily seasonality from
    # masquerading as an anomaly while retaining a distribution-free center.
    profile = _seasonal_profile(complete_history)
    residuals = [
        value - profile[index % 24] for index, value in enumerate(complete_history)
    ]
    scale = _robust_scale(residuals, 0.0)

    z_scores: List[float] = []
    relative_deviations: List[float] = []
    expected_values: List[float] = []
    start = len(complete_history)
    for offset, value in enumerate(complete_recent):
        expected = profile[(start + offset) % 24]
        expected_values.append(expected)
        z_scores.append((value - expected) / scale)
        relative_deviations.append(abs(value - expected) / max(abs(expected), scale, 1e-9))

    positive = 0.0
    negative = 0.0
    temporal = 0.0
    allowance = 0.50
    for z_score in z_scores:
        positive = max(0.0, positive + z_score - allowance)
        negative = max(0.0, negative - z_score - allowance)
        temporal = max(temporal, positive, negative)

    evidence = MetricEvidence(
        metric=metric,
        history_n=len(complete_history),
        recent_n=len(complete_recent),
        expected_last=expected_values[-1],
        observed_last=complete_recent[-1],
        point_score=max(abs(value) for value in z_scores),
        last_point_score=abs(z_scores[-1]),
        temporal_score=temporal,
        max_relative_deviation=max(relative_deviations),
        last_relative_deviation=relative_deviations[-1],
        quality_ok=True,
        statuses=statuses,
    )
    return evidence, z_scores


def extract_features(scenario: ScenarioInput, calibration: Calibration) -> EvidenceFeatures:
    metrics = sorted(set(scenario.history) | set(scenario.recent))
    evidence_rows: List[MetricEvidence] = []
    z_by_metric: Dict[str, List[float]] = {}
    quality_issues: Dict[str, Tuple[str, ...]] = {}
    insufficient: List[str] = []

    for metric in metrics:
        history = list(scenario.history.get(metric, []))
        recent = list(scenario.recent.get(metric, []))
        history_status = _status_vector(scenario.history_status, metric, len(history))
        recent_status = _status_vector(scenario.recent_status, metric, len(recent))
        evidence, z_scores = _metric_features(
            metric,
            history,
            recent,
            history_status,
            recent_status,
            min_history=calibration.min_history,
        )
        evidence_rows.append(evidence)
        if z_scores:
            z_by_metric[metric] = z_scores
        if not evidence.quality_ok:
            quality_issues[metric] = evidence.statuses
        if evidence.history_n < calibration.min_history:
            insufficient.append(metric)

    point_score = max((row.point_score for row in evidence_rows), default=0.0)
    temporal_score = max((row.temporal_score for row in evidence_rows), default=0.0)

    # Cross-metric channel: persistent top-k energy above an ordinary multivariate
    # noise floor.  Calibration converts this dimensionless statistic into a
    # family-wise decision threshold on benign holdout simulations.
    collective_score = 0.0
    if z_by_metric:
        window = min(len(scores) for scores in z_by_metric.values())
        cumulative = 0.0
        for offset in range(window):
            magnitudes = sorted(
                (abs(scores[offset]) for scores in z_by_metric.values()), reverse=True
            )[: calibration.max_fused_metrics]
            energy = math.sqrt(sum(value * value for value in magnitudes))
            cumulative += max(0.0, energy - 2.80)
        collective_score = cumulative / math.sqrt(max(1, window))

    composite_score = max(
        point_score / calibration.base_point,
        temporal_score / calibration.base_temporal,
        collective_score / calibration.base_collective,
    )

    max_point = max(point_score, 1e-9)
    max_temporal = max(temporal_score, 1e-9)
    drivers = [
        row.metric
        for row in evidence_rows
        if row.quality_ok
        and (
            row.point_score >= max(2.0, 0.45 * max_point)
            or row.temporal_score >= max(3.5, 0.45 * max_temporal)
        )
    ]

    return EvidenceFeatures(
        metric_evidence=evidence_rows,
        point_score=point_score,
        temporal_score=temporal_score,
        collective_score=collective_score,
        composite_score=composite_score,
        driver_metrics=tuple(sorted(set(drivers))),
        quality_issues=quality_issues,
        insufficient_history_metrics=tuple(sorted(insufficient)),
    )


def calibrate(
    benign_scenarios: Iterable[ScenarioInput],
    *,
    quantile: float = 0.99,
    target_false_alert_rate: float = 0.01,
    base_point: float = 3.0,
    base_temporal: float = 8.0,
    base_collective: float = 5.0,
    min_history: int = 72,
) -> Calibration:
    # The channel normalizers are learned at the 95th percentile of the null.
    # A second family-wise gate is then learned on the maximum normalized score.
    # This gives temporal and collective evidence an equal chance to contribute
    # while the final quantile, not three independent thresholds, controls alerting.
    provisional = Calibration(
        composite_threshold=1.0,
        quantile=quantile,
        calibration_size=0,
        target_false_alert_rate=target_false_alert_rate,
        base_point=base_point,
        base_temporal=base_temporal,
        base_collective=base_collective,
        min_history=min_history,
    )
    channel_rows: List[Tuple[float, float, float]] = []
    for scenario in benign_scenarios:
        if scenario.lifecycle:
            continue
        features = extract_features(scenario, provisional)
        if features.quality_issues or features.insufficient_history_metrics:
            continue
        channel_rows.append(
            (features.point_score, features.temporal_score, features.collective_score)
        )
    if len(channel_rows) < 100:
        raise ValueError("at least 100 valid benign calibration scenarios are required")
    channel_quantile = 0.95
    point_normalizer = max(1e-9, _percentile([row[0] for row in channel_rows], channel_quantile))
    temporal_normalizer = max(1e-9, _percentile([row[1] for row in channel_rows], channel_quantile))
    collective_normalizer = max(1e-9, _percentile([row[2] for row in channel_rows], channel_quantile))
    normalized_scores = [
        max(
            point / point_normalizer,
            temporal / temporal_normalizer,
            collective / collective_normalizer,
        )
        for point, temporal, collective in channel_rows
    ]
    threshold = max(1.0, _percentile(normalized_scores, quantile))
    return Calibration(
        composite_threshold=threshold,
        quantile=quantile,
        calibration_size=len(channel_rows),
        target_false_alert_rate=target_false_alert_rate,
        base_point=point_normalizer,
        base_temporal=temporal_normalizer,
        base_collective=collective_normalizer,
        min_history=min_history,
    )


def _dominant_channel(features: EvidenceFeatures, calibration: Calibration) -> str:
    channels = {
        "point": features.point_score / calibration.base_point,
        "temporal": features.temporal_score / calibration.base_temporal,
        "collective": features.collective_score / calibration.base_collective,
    }
    return max(channels, key=channels.get)


def _valid_context_coverage(
    events: Sequence[ContextEvent], drivers: Sequence[str], minimum_confidence: float
) -> Tuple[bool, Tuple[str, ...], str]:
    considered = tuple(event.event_key for event in events)
    valid = [
        event
        for event in events
        if event.approved and event.active and event.confidence >= minimum_confidence
    ]
    covered = {metric for event in valid for metric in event.covered_metrics}
    driver_set = set(drivers)
    if not driver_set:
        return False, considered, "no evidence-driving metrics to explain"
    if driver_set.issubset(covered):
        return True, considered, "approved active context covers every evidence driver"
    uncovered = sorted(driver_set - covered)
    return False, considered, f"context scope does not cover drivers: {','.join(uncovered)}"


def _context_covers_evidence(
    events: Sequence[ContextEvent],
    features: EvidenceFeatures,
    calibration: Calibration,
    minimum_confidence: float,
) -> Tuple[bool, Tuple[str, ...], str]:
    """Require context to cover every independently strong metric.

    Random secondary peaks are not allowed to invalidate a well-scoped change
    record, but an uncovered metric that independently approaches the calibrated
    alert boundary prevents suppression.  This is the critic's anti-context-
    poisoning rule.
    """
    considered = tuple(event.event_key for event in events)
    valid = [
        event
        for event in events
        if event.approved and event.active and event.confidence >= minimum_confidence
    ]
    covered = {metric for event in valid for metric in event.covered_metrics}
    independent_boundary = calibration.composite_threshold * 0.85
    strong_metrics = {
        row.metric
        for row in features.metric_evidence
        if row.quality_ok
        and max(
            row.point_score / max(calibration.base_point, 1e-9),
            row.temporal_score / max(calibration.base_temporal, 1e-9),
        ) >= independent_boundary
    }
    evidence_metrics = strong_metrics or set(features.driver_metrics)
    if not evidence_metrics:
        return False, considered, "no evidence-driving metrics to explain"
    if evidence_metrics.issubset(covered):
        return True, considered, "context covers every independently strong metric"
    uncovered = sorted(evidence_metrics - covered)
    return False, considered, f"context does not cover strong metrics: {','.join(uncovered)}"


class BlackSwanV7:
    def __init__(self, calibration: Calibration):
        self.calibration = calibration

    def decide(self, scenario: ScenarioInput) -> Decision:
        features = extract_features(scenario, self.calibration)
        threshold = self.calibration.composite_threshold
        score = features.composite_score
        margin = score / max(threshold, 1e-9) - 1.0
        uncertainty = max(0.0, 1.0 - min(1.0, abs(margin)))
        dominant = _dominant_channel(features, self.calibration)
        ledger: List[dict] = [
            {
                "stage": "detector",
                "composite_score": score,
                "threshold": threshold,
                "point_score": features.point_score,
                "temporal_score": features.temporal_score,
                "collective_score": features.collective_score,
                "dominant_channel": dominant,
                "drivers": list(features.driver_metrics),
            }
        ]

        # Critic guard 1: lifecycle facts take precedence over numerical absence.
        if scenario.lifecycle:
            route = "SERIES_LIFECYCLE_CHANGE"
            ledger.append({"stage": "critic", "rule": "lifecycle_before_forecast"})
            return self._decision(
                scenario, features, route, route, False, False, margin, uncertainty,
                dominant, False, "lifecycle routed before numerical scoring", (), ledger,
            )

        # Critic guard 2: missing telemetry is never coerced to zero.  Deliberate
        # suppression has a distinct route because blindness itself is a threat.
        if features.quality_issues:
            states = {state for values in features.quality_issues.values() for state in values}
            suspected_suppression = "suppressed" in states
            route = (
                "TELEMETRY_HOLD_SECURITY_RISK"
                if suspected_suppression
                else "TELEMETRY_HOLD"
            )
            ledger.append(
                {
                    "stage": "critic",
                    "rule": "quality_before_forecast",
                    "suppression_risk": suspected_suppression,
                }
            )
            return self._decision(
                scenario, features, route, route, False, True, margin, uncertainty,
                dominant, False, "non-numeric telemetry routed without zero imputation", (), ledger,
            )

        if features.insufficient_history_metrics:
            route = "INSUFFICIENT_HISTORY_HOLD"
            ledger.append(
                {
                    "stage": "critic",
                    "rule": "minimum_history",
                    "metrics": list(features.insufficient_history_metrics),
                }
            )
            return self._decision(
                scenario, features, route, route, False, True, margin, uncertainty,
                dominant, False, "insufficient continuous history", (), ledger,
            )

        candidate = score >= threshold
        review_band = score >= threshold * self.calibration.review_fraction
        initial_route = "RAW_ANOMALY_CANDIDATE" if candidate else (
            "RAW_MODEL_REVIEW" if review_band else "RAW_NO_ALERT"
        )

        covered, context_keys, context_reason = _context_covers_evidence(
            scenario.context_events,
            features,
            self.calibration,
            self.calibration.context_min_confidence,
        )
        moderate_coverage, _, moderate_reason = _context_covers_evidence(
            scenario.context_events,
            features,
            self.calibration,
            0.80,
        )
        ledger.append(
            {
                "stage": "critic",
                "rule": "context_scope_and_provenance",
                "candidate": candidate,
                "context_keys": list(context_keys),
                "context_covers_drivers": covered,
                "reason": context_reason,
            }
        )

        if candidate and covered:
            final_route = "BENIGN_EXPLAINED_BREAK"
            return self._decision(
                scenario, features, initial_route, final_route, False, False,
                margin, uncertainty, dominant, True,
                "critic converted a raw anomaly after verified context coverage",
                context_keys, ledger,
            )
        if candidate and moderate_coverage:
            final_route = "MODEL_REVIEW_CONTEXT_UNCERTAIN"
            return self._decision(
                scenario, features, initial_route, final_route, False, True,
                margin, uncertainty, dominant, True,
                "critic abstained because context covers the signal but confidence is insufficient to explain it",
                context_keys, ledger,
            )
        if candidate:
            final_route = "SECURITY_ANOMALY_REVIEW"
            return self._decision(
                scenario, features, initial_route, final_route, True, True,
                margin, uncertainty, dominant, False, context_reason,
                context_keys, ledger,
            )
        if review_band:
            final_route = "MODEL_REVIEW"
            return self._decision(
                scenario, features, initial_route, final_route, False, True,
                margin, uncertainty, dominant, False,
                "evidence is inside calibrated abstention band",
                context_keys, ledger,
            )
        return self._decision(
            scenario, features, initial_route, "NO_SECURITY_ALERT", False, False,
            margin, uncertainty, dominant, False,
            "evidence remains below calibrated review band", context_keys, ledger,
        )

    def _decision(
        self,
        scenario: ScenarioInput,
        features: EvidenceFeatures,
        initial_route: str,
        final_route: str,
        security_alert: bool,
        operational_escalation: bool,
        margin: float,
        uncertainty: float,
        dominant: str,
        correction_applied: bool,
        correction_reason: str,
        context_keys: Tuple[str, ...],
        ledger: List[dict],
    ) -> Decision:
        ledger.append(
            {
                "stage": "policy",
                "initial_route": initial_route,
                "final_route": final_route,
                "security_alert": security_alert,
                "operational_escalation": operational_escalation,
            }
        )
        return Decision(
            scenario_id=scenario.scenario_id,
            engine_version="v7",
            initial_route=initial_route,
            final_route=final_route,
            security_alert=security_alert,
            operational_escalation=operational_escalation,
            composite_score=features.composite_score,
            calibrated_threshold=self.calibration.composite_threshold,
            evidence_margin=margin,
            uncertainty=uncertainty,
            point_score=features.point_score,
            temporal_score=features.temporal_score,
            collective_score=features.collective_score,
            dominant_channel=dominant,
            driver_metrics=features.driver_metrics,
            correction_applied=correction_applied,
            correction_reason=correction_reason,
            context_keys_considered=context_keys,
            quality_issues=features.quality_issues,
            lifecycle=dict(scenario.lifecycle),
            ledger=ledger,
        )


class BlackSwanV6Baseline:
    """A frozen v6-compatible comparator for the new sequential benchmark.

    It preserves the defining v6 gates: 3-sigma point evidence plus at least
    25% relative deviation, lifecycle/quality routing, and context-based
    explanation.  It intentionally has no temporal or cross-metric fusion.
    """

    def __init__(self, min_history: int = 72):
        self.calibration = Calibration(
            composite_threshold=1.0,
            quantile=0.0,
            calibration_size=0,
            target_false_alert_rate=0.0,
            min_history=min_history,
        )

    def decide(self, scenario: ScenarioInput) -> Decision:
        features = extract_features(scenario, self.calibration)
        anomalous: List[str] = []
        metric_consensus_risk: Dict[str, float] = {}
        for row in features.metric_evidence:
            metric = row.metric
            if not row.quality_ok:
                continue
            history = [
                float(value)
                for value, status in zip(
                    scenario.history.get(metric, []),
                    _status_vector(
                        scenario.history_status,
                        metric,
                        len(scenario.history.get(metric, [])),
                    ),
                )
                if value is not None and status == COMPLETE
            ]
            recent = [
                float(value)
                for value, status in zip(
                    scenario.recent.get(metric, []),
                    _status_vector(
                        scenario.recent_status,
                        metric,
                        len(scenario.recent.get(metric, [])),
                    ),
                )
                if value is not None and status == COMPLETE
            ]
            if len(history) < self.calibration.min_history or not recent:
                continue
            prior = history + recent[:-1]
            target = recent[-1]

            # Three deliberately different one-step models approximate v6's
            # naive/trend/growth consensus on the sequential benchmark.
            naive_expected = prior[-1]
            naive_errors = [prior[index] - prior[index - 1] for index in range(1, len(prior))]
            naive_scale = _robust_scale(naive_errors, _median(naive_errors))

            trend_values = prior[-48:]
            count = len(trend_values)
            x_mean = (count - 1) / 2.0
            y_mean = sum(trend_values) / count
            denominator = sum((index - x_mean) ** 2 for index in range(count))
            slope = (
                sum((index - x_mean) * (value - y_mean) for index, value in enumerate(trend_values))
                / denominator
                if denominator
                else 0.0
            )
            intercept = y_mean - slope * x_mean
            trend_expected = intercept + slope * count
            trend_errors = [
                value - (intercept + slope * index)
                for index, value in enumerate(trend_values)
            ]
            trend_scale = _robust_scale(trend_errors, 0.0)

            if len(prior) >= 48:
                daily_changes = [
                    prior[index] - prior[index - 24]
                    for index in range(24, len(prior))
                ]
                daily_drift = _median(daily_changes)
                seasonal_expected = prior[-24] + daily_drift
                seasonal_scale = _robust_scale(daily_changes, daily_drift)
            else:
                seasonal_expected = naive_expected
                seasonal_scale = naive_scale

            strengths: List[float] = []
            for expected, scale in (
                (naive_expected, naive_scale),
                (trend_expected, trend_scale),
                (seasonal_expected, seasonal_scale),
            ):
                gap = abs(target - expected)
                z_strength = gap / max(3.0 * scale, 1e-9)
                relative_strength = gap / max(0.25 * abs(expected), 0.25 * scale, 1e-9)
                strengths.append(min(z_strength, relative_strength))
            strengths.sort(reverse=True)
            consensus_risk = strengths[1]  # second-strongest == 2-of-3 consensus
            metric_consensus_risk[metric] = consensus_risk
            if consensus_risk >= 1.0:
                anomalous.append(metric)
        risk = max(metric_consensus_risk.values(), default=0.0)
        quality = features.quality_issues
        ledger = [{
            "stage": "detector",
            "v6_anomalous_metrics": anomalous,
            "consensus_risk": metric_consensus_risk,
            "risk": risk,
        }]
        if scenario.lifecycle:
            route, alert, escalation = "SERIES_LIFECYCLE_CHANGE", False, False
        elif quality:
            route, alert, escalation = "TELEMETRY_HOLD", False, True
        elif features.insufficient_history_metrics:
            route, alert, escalation = "INSUFFICIENT_HISTORY_HOLD", False, True
        elif anomalous:
            covered, _, _ = _valid_context_coverage(
                scenario.context_events, anomalous, 0.95
            )
            if covered:
                route, alert, escalation = "BENIGN_EXPLAINED_BREAK", False, False
            else:
                route, alert, escalation = "SECURITY_ANOMALY_REVIEW", True, True
        elif risk >= 0.80:
            route, alert, escalation = "MODEL_REVIEW", False, True
        else:
            route, alert, escalation = "NO_SECURITY_ALERT", False, False
        ledger.append({"stage": "policy", "final_route": route})
        margin = risk - 1.0
        return Decision(
            scenario_id=scenario.scenario_id,
            engine_version="v6_baseline",
            initial_route=route,
            final_route=route,
            security_alert=alert,
            operational_escalation=escalation,
            composite_score=risk,
            calibrated_threshold=1.0,
            evidence_margin=margin,
            uncertainty=max(0.0, 1.0 - min(1.0, abs(margin))),
            point_score=features.point_score,
            temporal_score=0.0,
            collective_score=0.0,
            dominant_channel="point",
            driver_metrics=tuple(anomalous),
            correction_applied=False,
            correction_reason="frozen v6 comparator",
            context_keys_considered=tuple(event.event_key for event in scenario.context_events),
            quality_issues=quality,
            lifecycle=dict(scenario.lifecycle),
            ledger=ledger,
        )
