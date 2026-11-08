"""Block language-model checkpoints that do not fit this laptop.

The image classifier does not call this module. A 27B model is never loaded.
MedGemma 1.5 4B is optional and is rejected because its published weights are
8.60 GB, above the 8 GB checkpoint cap. Nothing here downloads weights.
"""

from __future__ import annotations

from utils.hardware import assert_model_id_allowed, ram_bytes

MEDGEMMA_27B = "google/medgemma-27b-text-it"
MEDGEMMA_4B = "google/medgemma-1.5-4b-it"


def ram_gib() -> float | None:
    memory = ram_bytes()
    return None if memory is None else memory / 1024**3


def assert_requested_model_can_load(model_id: str, ram_gib_available: float | None = None) -> None:
    """Reject oversized ids. ``ram_gib_available`` is accepted and not used to waive the cap."""
    del ram_gib_available
    assert_model_id_allowed(model_id)
