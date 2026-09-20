"""Black Swan v8: deterministic anomaly triage; synthetic evaluation only."""
from .engine import BlackSwanV8, Limits, TrustedContext, WindowUpdate, V8Decision
from .runtime import IsolatedDecisionPool
from black_swan_v7_engine import Calibration, ContextEvent, ScenarioInput

__version__ = "8.0.0"
__all__ = ["BlackSwanV8", "Limits", "TrustedContext", "WindowUpdate", "V8Decision",
           "IsolatedDecisionPool", "Calibration", "ContextEvent", "ScenarioInput"]
