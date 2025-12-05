"""Minimal DSPy LM adapter exposing a cached predict helper."""

from __future__ import annotations

from typing import Any

import dspy


class DSPyAgentClient:
    """Adapter over a DSPy language model that caches predictors by signature."""

    def __init__(self, lm_config: dict[str, Any]):
        if not lm_config:
            raise ValueError("lm_config with model details is required to initialize DSPyAgentClient")

        self.lm = dspy.LM(**lm_config)
        self._predictors: dict[type[dspy.Signature], dspy.Predict] = {}

    def predict(self, signature: type[dspy.Signature] | dspy.Signature, **kwargs: Any):
        """Execute a DSPy prediction for the given signature with caching."""

        signature_type = signature if isinstance(signature, type) else signature.__class__
        predictor = self._predictors.get(signature_type)
        if predictor is None:
            predictor = dspy.Predict(signature, lm=self.lm)
            self._predictors[signature_type] = predictor

        return predictor(**kwargs)

