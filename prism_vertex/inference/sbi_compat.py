from __future__ import annotations

# Compatible posterior_nn import across sbi versions.
try:  # older sbi (e.g., 0.19.x)
    from sbi.utils.get_nn_models import posterior_nn  # type: ignore
except Exception:  # newer sbi (e.g., 0.23.x)
    from sbi.neural_nets import posterior_nn  # type: ignore

__all__ = ["posterior_nn"]

