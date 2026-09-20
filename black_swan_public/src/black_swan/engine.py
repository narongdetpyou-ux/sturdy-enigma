"""Black Swan v8: bounded inputs, explicit review, trusted context, cached v7 math.

This in-process API is for primitive, trusted Python containers. Use the isolated
pool for hard cancellation of compute; decide() alone has no hard deadline.
No trained model, external trust lookup, or production deployment is implied.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

from black_swan_v7_engine import (
    BlackSwanV7, Calibration, ContextEvent, Decision, EvidenceFeatures,
    MetricEvidence, ScenarioInput, _context_covers_evidence, _dominant_channel,
    _robust_scale,
)


@dataclass(frozen=True)
class Limits:
    max_metrics: int = 256
    max_history: int = 4096
    max_recent: int = 256
    max_scalars: int = 262144
    max_contexts: int = 32
    max_text: int = 256
    max_abs_value: float = 1e100
    cache_entries: int = 256


@dataclass(frozen=True)
class TrustedContext:
    """Installed by operator, never auto-created from request metadata.

    Local registry is an explicit trust boundary, NOT cryptographic proof or a
    causality oracle. Production must populate/revoke it from verified records.
    """
    event_key: str
    evidence_ref: str
    covered_metrics: tuple[str, ...]
    confidence: float
    valid_from: float
    valid_until: float


@dataclass
class V8Decision(Decision):
    reason_code: str = ''
    excluded_metrics: dict = field(default_factory=dict)


@dataclass
class WindowUpdate:
    """Recent window against an operator-registered immutable history revision."""
    baseline_id: str
    scenario_id: str
    recent: dict
    recent_status: dict = field(default_factory=dict)
    lifecycle: dict = field(default_factory=dict)
    context_events: list = field(default_factory=list)


def fallback(route, reason, scenario_id='unavailable', threshold=1.0):
    return V8Decision(
        scenario_id=scenario_id if type(scenario_id) is str else 'unavailable',
        engine_version='v8', initial_route=route, final_route=route,
        security_alert=False, operational_escalation=True, composite_score=0.0,
        calibrated_threshold=threshold, evidence_margin=0.0, uncertainty=1.0,
        point_score=0.0, temporal_score=0.0, collective_score=0.0,
        dominant_channel='unavailable', driver_metrics=(), correction_applied=False,
        correction_reason=reason, context_keys_considered=(), quality_issues={},
        lifecycle={}, ledger=[{'stage': 'terminal', 'reason_code': reason}],
        reason_code=reason,
    )


class InvalidInput(ValueError):
    pass


class BlackSwanV8(BlackSwanV7):
    def __init__(self, calibration, *, limits=Limits(), trusted_contexts=()):
        super().__init__(calibration)
        self.limits = limits
        self._cache = OrderedDict()
        self._cache_lock = threading.RLock()
        self.cache_hits = self.cache_misses = 0
        self._config_error = self._validate_config()
        self._trusted = {}
        self._baselines = {}
        self._baseline_names = {}
        for record in trusted_contexts:
            if (type(record) is not TrustedContext or not record.evidence_ref
                    or type(record.covered_metrics) is not tuple
                    or not all(type(k) is str for k in record.covered_metrics)
                    or not all(type(x) in (int, float) and math.isfinite(x)
                               for x in (record.confidence, record.valid_from, record.valid_until))
                    or not 0 <= record.confidence <= 1 or record.valid_until <= record.valid_from):
                self._config_error = 'INVALID_TRUST_REGISTRY'
                continue
            key = (record.event_key, record.evidence_ref)
            if key in self._trusted:
                self._config_error = 'DUPLICATE_TRUST_RECORD'
            self._trusted[key] = record

    def _validate_config(self):
        c = self.calibration
        if type(c) is not Calibration or type(self.limits) is not Limits:
            return 'INVALID_CONFIGURATION_TYPE'
        for key in ('composite_threshold', 'base_point', 'base_temporal', 'base_collective'):
            value = getattr(c, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                return 'INVALID_CALIBRATION_DIVISOR'
        for key in ('quantile', 'target_false_alert_rate', 'review_fraction', 'context_min_confidence'):
            value = getattr(c, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                return 'INVALID_CALIBRATION_PROBABILITY'
        if c.context_min_confidence < .8:
            return 'INVALID_CONTEXT_THRESHOLD'
        for key in ('min_history', 'max_fused_metrics', 'calibration_size'):
            if type(getattr(c, key)) is not int or getattr(c, key) < 1:
                return 'INVALID_CALIBRATION_COUNT'
        for key in ('max_metrics', 'max_history', 'max_recent', 'max_scalars', 'max_contexts', 'max_text'):
            if type(getattr(self.limits, key)) is not int or getattr(self.limits, key) < 1:
                return 'INVALID_LIMIT'
        if (type(self.limits.cache_entries) is not int or self.limits.cache_entries < 0
                or type(self.limits.max_abs_value) not in (float, int)
                or not 0 < self.limits.max_abs_value <= 1e100):
            return 'INVALID_CACHE_OR_NUMERIC_LIMIT'
        return None

    def _text(self, value, *, empty=False):
        if type(value) is not str or len(value) > self.limits.max_text or (not empty and not value):
            raise InvalidInput('INVALID_TEXT_FIELD')
        return value

    def _numeric(self, values):
        result = []
        maximum = self.limits.max_abs_value
        for value in values:
            if value is None:
                result.append(None)
            elif (type(value) not in (int, float) or abs(value) > maximum
                  or not math.isfinite(value)):
                raise InvalidInput('NONFINITE_OR_INVALID_NUMBER')
            else:
                result.append(float(value))
        return result

    def _validate(self, s, prepared=None):
        if type(s) is not ScenarioInput:
            raise InvalidInput('INVALID_ENVELOPE')
        self._text(s.scenario_id)
        fields = (s.history, s.recent, s.history_status, s.recent_status, s.lifecycle)
        if any(type(x) is not dict for x in fields):
            raise InvalidInput('INVALID_MAPPING')
        if any(len(x) > self.limits.max_metrics for x in fields):
            raise InvalidInput('TOO_MANY_METRICS')
        metrics = set(s.history) | set(s.recent)
        if not metrics or len(metrics) > self.limits.max_metrics:
            raise InvalidInput('EMPTY_OR_OVERSIZED_INPUT')
        for mapping in fields:
            for key in mapping:
                self._text(key)
        if (set(s.history_status) - set(s.history) or set(s.recent_status) - set(s.recent)):
            raise InvalidInput('ORPHAN_STATUS')
        total = 0
        normalized, excluded, issues = {}, {}, {}
        recent_lengths = set()
        for metric in sorted(metrics):
            h, r = s.history.get(metric, []), s.recent.get(metric, [])
            if (prepared is None and type(h) is not list) or type(r) is not list:
                raise InvalidInput('INVALID_SERIES_TYPE')
            if len(h) > self.limits.max_history or len(r) > self.limits.max_recent:
                raise InvalidInput('SERIES_TOO_LARGE')
            total += len(h) + len(r)
            if total > self.limits.max_scalars:
                raise InvalidInput('PAYLOAD_TOO_LARGE')
            hs = s.history_status.get(metric, ['complete'] * len(h)) if prepared is None else ()
            rs = s.recent_status.get(metric, ['complete'] * len(r))
            if (prepared is None and (type(hs) is not list or len(hs) != len(h))) or type(rs) is not list or len(rs) != len(r):
                raise InvalidInput('STATUS_LENGTH_OR_TYPE')
            for status in (hs + rs if prepared is None else rs):
                self._text(status)
            h, r = (self._numeric(h) if prepared is None else h), self._numeric(r)
            if r:
                recent_lengths.add(len(r))
            reasons = []
            if metric in s.lifecycle:
                self._text(s.lifecycle[metric])
                reasons.append('LIFECYCLE')
            if not r or any(x is None for x in r):
                reasons.append('MISSING_RECENT')
            if any(x != 'complete' for x in rs):
                reasons.extend(sorted(set(x for x in rs if x != 'complete')))
            if len(h) < self.calibration.min_history:
                reasons.append('INSUFFICIENT_HISTORY')
            if prepared is None and (any(x is None for x in h) or any(x != 'complete' for x in hs)):
                reasons.append('HISTORY_GAP')
            if reasons:
                excluded[metric] = tuple(reasons)
                issues[metric] = tuple(reasons)
            else:
                normalized[metric] = (tuple(h), r)
        if len(recent_lengths) > 1:
            raise InvalidInput('RECENT_WINDOW_MISALIGNMENT')
        for metric, state in s.lifecycle.items():
            self._text(state)
            if metric not in metrics:
                issues[metric] = ('UNRELATED_LIFECYCLE',)
        if type(s.context_events) is not list or len(s.context_events) > self.limits.max_contexts:
            raise InvalidInput('INVALID_CONTEXT_LIST')
        events = []
        now = time.time()
        for event in s.context_events:
            if type(event) is not ContextEvent:
                raise InvalidInput('INVALID_CONTEXT_TYPE')
            self._text(event.event_key)
            self._text(event.evidence_ref, empty=True)
            if (type(event.covered_metrics) is not tuple or len(event.covered_metrics) > self.limits.max_metrics
                    or type(event.approved) is not bool or type(event.active) is not bool
                    or type(event.confidence) not in (int, float) or not math.isfinite(event.confidence)
                    or not 0 <= event.confidence <= 1):
                raise InvalidInput('INVALID_CONTEXT_FIELDS')
            for metric in event.covered_metrics:
                self._text(metric)
            trusted = self._trusted.get((event.event_key, event.evidence_ref))
            if (trusted and event.approved and event.active
                    and trusted.valid_from <= now < trusted.valid_until
                    and frozenset(event.covered_metrics) == frozenset(trusted.covered_metrics)
                    and event.confidence <= trusted.confidence):
                events.append(event)
        return normalized, excluded, issues, events

    def _history_stats(self, values):
        # Entire validated history tuple is the key: changed history cannot use
        # stale profile/scale. Cache is per-engine, bounded, and lock-protected.
        with self._cache_lock:
            found = self._cache.get(values)
            if found is not None:
                self.cache_hits += 1
                self._cache.move_to_end(values)
                return found
        buckets = [[] for _ in range(24)]
        for i, value in enumerate(values):
            buckets[i % 24].append(value)
        center = statistics.median(values)
        profile = tuple(float(statistics.median(b)) if b else float(center) for b in buckets)
        residuals = [value - profile[i % 24] for i, value in enumerate(values)]
        result = (profile, _robust_scale(residuals, 0.0))
        with self._cache_lock:
            self.cache_misses += 1
            if self.limits.cache_entries:
                self._cache[values] = result
                self._cache.move_to_end(values)
                while len(self._cache) > self.limits.cache_entries:
                    self._cache.popitem(last=False)
        return result

    def _features(self, normalized, issues):
        rows, z_by_metric = [], {}
        for metric, (h, r) in normalized.items():
            profile, scale = self._history_stats(h)
            expected = [profile[(len(h) + i) % 24] for i in range(len(r))]
            zs = [(value - e) / scale for value, e in zip(r, expected)]
            relative = [abs(value - e) / max(abs(e), scale, 1e-9) for value, e in zip(r, expected)]
            positive = negative = temporal = 0.0
            for z in zs:
                positive = max(0.0, positive + z - .5)
                negative = max(0.0, negative - z - .5)
                temporal = max(temporal, positive, negative)
            rows.append(MetricEvidence(metric, len(h), len(r), expected[-1], r[-1],
                                       max(abs(z) for z in zs), abs(zs[-1]), temporal,
                                       max(relative), relative[-1], True, ('complete',)))
            z_by_metric[metric] = zs
        point = max((r.point_score for r in rows), default=0.0)
        temporal = max((r.temporal_score for r in rows), default=0.0)
        collective = 0.0
        if z_by_metric:
            window = len(next(iter(z_by_metric.values())))
            cumulative = 0.0
            for i in range(window):
                magnitudes = sorted((abs(z[i]) for z in z_by_metric.values()), reverse=True)[:self.calibration.max_fused_metrics]
                energy = math.sqrt(sum(x * x for x in magnitudes))
                cumulative += max(0.0, energy - 2.8)
            collective = cumulative / math.sqrt(window)
        c = self.calibration
        composite = max(point / c.base_point, temporal / c.base_temporal, collective / c.base_collective)
        if not all(math.isfinite(x) for x in (point, temporal, collective, composite)):
            raise InvalidInput('NONFINITE_DERIVED_EVIDENCE')
        drivers = tuple(sorted(r.metric for r in rows if
                               r.point_score >= max(2.0, .45 * max(point, 1e-9)) or
                               r.temporal_score >= max(3.5, .45 * max(temporal, 1e-9))))
        return EvidenceFeatures(rows, point, temporal, collective, composite, drivers, issues, ())

    def decide(self, scenario):
        if self._config_error:
            return fallback('CONFIGURATION_REVIEW', self._config_error)
        try:
            return self._decide(scenario)
        except InvalidInput as exc:
            return fallback('INVALID_INPUT_REVIEW', str(exc))
        except Exception as exc:
            # No exception details from payloads are echoed. This is a failure
            # signal, not an automatic benign verdict.
            return fallback('VERIFIER_ERROR_REVIEW', f'VERIFIER_EXCEPTION_{type(exc).__name__}')

    def register_baseline(self, name, scenario):
        """Administrative preload, outside decision timing. Raises on bad setup.

        Snapshots are copied, immutable and content-addressed. Re-registering a
        name invalidates its previous token. Caller must advance the revision
        when the baseline/history should change; no automatic time ingestion.
        """
        if self._config_error:
            raise InvalidInput(self._config_error)
        self._text(name)
        rows, excluded, issues, _ = self._validate(scenario)
        if excluded or issues:
            raise InvalidInput('BASELINE_MUST_HAVE_COMPLETE_HISTORY_AND_ALIGNED_WINDOW')
        prepared = {key: values[0] for key, values in rows.items()}
        digest = hashlib.sha256(json.dumps([name, prepared], sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()
        with self._cache_lock:
            if name not in self._baseline_names and len(self._baseline_names) >= 128:
                raise InvalidInput('BASELINE_REGISTRY_FULL')
            previous = self._baseline_names.get(name)
            if previous:
                self._baselines.pop(previous, None)
            self._baseline_names[name] = digest
            self._baselines[digest] = prepared
        for h in prepared.values():
            self._history_stats(h)
        return digest

    def decide_update(self, update):
        if self._config_error:
            return fallback('CONFIGURATION_REVIEW', self._config_error)
        try:
            if type(update) is not WindowUpdate:
                raise InvalidInput('INVALID_UPDATE_ENVELOPE')
            self._text(update.scenario_id)
            self._text(update.baseline_id)
            with self._cache_lock:
                prepared = self._baselines.get(update.baseline_id)
            if prepared is None:
                return fallback('BASELINE_REVIEW', 'UNKNOWN_OR_REVOKED_BASELINE', update.scenario_id)
            if type(update.recent) is not dict or set(update.recent) - set(prepared):
                raise InvalidInput('UNKNOWN_UPDATE_METRIC')
            s = ScenarioInput(update.scenario_id, prepared, update.recent,
                              recent_status=update.recent_status, lifecycle=update.lifecycle,
                              context_events=update.context_events)
            return self._decide(s, prepared=prepared)
        except InvalidInput as exc:
            return fallback('INVALID_INPUT_REVIEW', str(exc))
        except Exception as exc:
            return fallback('VERIFIER_ERROR_REVIEW', f'VERIFIER_EXCEPTION_{type(exc).__name__}')

    def _decide(self, s, prepared=None):
        normalized, excluded, issues, events = self._validate(s, prepared)
        features = self._features(normalized, issues)
        score, c = features.composite_score, self.calibration
        margin = score / c.composite_threshold - 1.0
        uncertainty = max(0.0, 1.0 - min(1.0, abs(margin)))
        candidate = score >= c.composite_threshold
        review = score >= c.composite_threshold * c.review_fraction
        initial = 'RAW_ANOMALY_CANDIDATE' if candidate else ('RAW_MODEL_REVIEW' if review else 'RAW_NO_ALERT')
        ledger = [{'stage': 'input_validation', 'valid_metrics': len(normalized), 'excluded': excluded},
                  {'stage': 'detector', 'composite_score': score, 'threshold': c.composite_threshold},
                  {'stage': 'context_registry', 'supplied': len(s.context_events),
                   'verified': len(events), 'trust_mode': 'operator_installed_registry'}]
        corrected = False
        if candidate and issues:
            route, alert, escalation, reason = 'SECURITY_ANOMALY_REVIEW', True, True, 'SECURITY_EVIDENCE_WITH_DATA_ISSUES'
        elif issues:
            states = {state for values in issues.values() for state in values}
            if 'suppressed' in states:
                route = 'TELEMETRY_HOLD_SECURITY_RISK'
            elif 'LIFECYCLE' in states or 'UNRELATED_LIFECYCLE' in states:
                route = 'SERIES_LIFECYCLE_CHANGE'
            elif 'INSUFFICIENT_HISTORY' in states:
                route = 'INSUFFICIENT_HISTORY_HOLD'
            else:
                route = 'TELEMETRY_HOLD'
            alert, escalation, reason, uncertainty = False, True, 'INCOMPLETE_EVIDENCE_REQUIRES_REVIEW', 1.0
        else:
            covered, _, _ = _context_covers_evidence(events, features, c, c.context_min_confidence)
            moderate, _, _ = _context_covers_evidence(events, features, c, .8)
            if candidate and covered:
                route, alert, escalation, reason, corrected = 'BENIGN_EXPLAINED_BREAK', False, False, 'TRUSTED_CONTEXT_COVERS_EVIDENCE', True
            elif candidate and moderate:
                route, alert, escalation, reason, corrected = 'MODEL_REVIEW_CONTEXT_UNCERTAIN', False, True, 'TRUSTED_CONTEXT_CONFIDENCE_INSUFFICIENT', True
            elif candidate:
                route, alert, escalation, reason = 'SECURITY_ANOMALY_REVIEW', True, True, 'UNEXPLAINED_ANOMALY'
            elif review:
                route, alert, escalation, reason = 'MODEL_REVIEW', False, True, 'SCORE_IN_REVIEW_BAND'
            else:
                route, alert, escalation, reason = 'NO_SECURITY_ALERT', False, False, 'NO_OBSERVABLE_ANOMALY'
        d = super()._decision(s, features, initial, route, alert, escalation, margin, uncertainty,
                             _dominant_channel(features, c), corrected, reason,
                             tuple(e.event_key for e in events), ledger)
        d.engine_version = 'v8'
        d.ledger.append({'stage': 'terminal', 'reason_code': reason})
        return V8Decision(**vars(d), reason_code=reason, excluded_metrics=excluded)
