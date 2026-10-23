from __future__ import annotations

PAPER_METHODS: tuple[str, ...] = (
    "hf_smc_abc",
    "hf_npe",
    "hf_snpe",
    "mf_sbi",
)

PAPER_METHODS_STR = ",".join(PAPER_METHODS)

IMPLEMENTED_METHODS = frozenset(PAPER_METHODS)


def validate_method_supported(method: str) -> None:
    if method in IMPLEMENTED_METHODS:
        return
    raise ValueError(f"Unknown method: {method}. Supported: {', '.join(PAPER_METHODS)}")
