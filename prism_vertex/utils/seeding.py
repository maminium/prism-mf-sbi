from __future__ import annotations

import hashlib


def derive_seed(*parts: object, mod: int = 2**32) -> int:
    """
    Deterministically derive an integer seed from arbitrary inputs.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") % mod

