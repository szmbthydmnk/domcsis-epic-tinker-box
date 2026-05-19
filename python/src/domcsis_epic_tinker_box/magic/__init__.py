from .stabilizer_renyi_entropy import (
    stabilizer_renyi_entropy,
    _stabilizer_renyi_entropy_unchecked,
    _validate_alpha,
    _validate_compatible,
    _characteristic_function,
    _compute_sre,
)

__all__ = [
    "stabilizer_renyi_entropy",
    # Fast path and standalone validators — not in public API
    # but exposed for hot-loop callers who validate once externally.
    "_stabilizer_renyi_entropy_unchecked",
    "_validate_alpha",
    "_validate_compatible",
    "_characteristic_function",
    "_compute_sre",
]
