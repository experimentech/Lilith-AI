from __future__ import annotations

from typing import Any, Dict, Optional

from lilith.general_purpose_learner import GeneralPurposeLearner, OutcomeSignals

from .pattern_store_adapter import PatternStoreAdapter


class ConceptGPLearner(GeneralPurposeLearner):
    """Layer-specific learner for concepts using the GPL core loop."""

    def __init__(
        self,
        pattern_store: PatternStoreAdapter,
        learning_rate: float = 0.1,
        learning_mode: str = "moderate",
    ) -> None:
        super().__init__("concept", pattern_store, learning_rate, learning_mode)

    def _evaluate_outcome(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]],
    ) -> OutcomeSignals:
        ctx = context or {}
        success = float(ctx.get("success", 1.0))
        confidence = float(ctx.get("confidence", 0.5))
        success = max(-1.0, min(1.0, success))
        confidence = max(0.0, min(1.0, confidence))
        return OutcomeSignals(
            layer_name="concept",
            overall_success=success,
            confidence=confidence,
            layer_signals={"source": ctx.get("source", "runtime")},
        )

    def _extract_and_store_pattern(
        self,
        layer_input: Any,
        layer_output: Any,
        signals: OutcomeSignals,
        context: Optional[Dict[str, Any]],
    ) -> None:
        ctx = context or {}
        event = layer_input if isinstance(layer_input, dict) else {"text": str(layer_input)}
        trigger = str(event.get("text") or event.get("trigger") or layer_input)
        response = str(event.get("response") or event.get("value") or layer_output)
        intent = str(event.get("intent") or ctx.get("intent") or "concept")
        embedding = ctx.get("embedding")
        pm_latent = ctx.get("pm_latent")
        pm_raw = ctx.get("pm_raw")
        fragment_id = event.get("id") if isinstance(event.get("id"), str) else None
        pid = self.pattern_store.add_pattern(
            fragment_id=fragment_id,
            trigger_context=trigger,
            response_text=response,
            intent=intent,
            success_score=signals.overall_success,
            embedding=embedding,
            pm_latent=pm_latent,
            pm_raw=pm_raw,
        )
        self.patterns_learned += 1
        ctx["pattern_id"] = pid

    def _apply_reinforcement(self, layer_output: Any, signals: OutcomeSignals) -> None:
        feedback = signals.overall_success * self.learning_rate
        pid = getattr(layer_output, "id", None)
        if isinstance(layer_output, dict):
            pid = layer_output.get("id", pid)
        if pid:
            self.pattern_store.update_success(pid, feedback, plasticity_rate=1.0)
