from __future__ import annotations

import os

import numpy as np

from prism_vertex.simulator.chaste_runner import (
    ChasteRunError,
    run_chaste,
)
from prism_vertex.simulator.fidelities import (
    HF_SINGLE,
    FidelitySpec,
    get_fidelity,
    lf_fidelity_from_env,
)
from prism_vertex.simulator.constants import K_HF_REPLICATES
from prism_vertex.utils.seeding import derive_seed


class ProliferatingVertexSimulator:
    """
    Unified entry point for Chaste runs (scoreboard + inference).

    HF label for training: mean of ``K`` runs with ``HF_SINGLE``.
    LF label: one run with ``lf_*`` spec (env ``PRISM_VERTEX_LF_FIDELITY``).
    """

    def run_single(
        self,
        theta: np.ndarray,
        sim_seed: int,
        spec: FidelitySpec,
    ) -> np.ndarray:
        return run_chaste(theta, sim_seed, spec).summary_6d

    def run_hf_single(self, theta: np.ndarray, sim_seed: int) -> np.ndarray:
        return self.run_single(theta, sim_seed, HF_SINGLE)

    def run_lf(self, theta: np.ndarray, sim_seed: int, fidelity_id: str | None = None) -> np.ndarray:
        spec = get_fidelity(fidelity_id) if fidelity_id else lf_fidelity_from_env()
        return self.run_single(theta, sim_seed, spec)

    def run_hf_k_mean(
        self,
        theta: np.ndarray,
        batch_seed: int,
        k: int = K_HF_REPLICATES,
        *,
        strict_any_fail: bool | None = None,
    ) -> np.ndarray:
        if strict_any_fail is None:
            strict_any_fail = os.environ.get("PRISM_VERTEX_HF_STRICT_ANY_FAIL", "1") not in {
                "0",
                "false",
                "False",
            }

        xs: list[np.ndarray] = []
        n_fail = 0
        last_err = ""
        for replicate_k in range(k):
            sim_seed = int(derive_seed("proliferating_vertex", "hf_train", batch_seed, replicate_k))
            try:
                xs.append(self.run_hf_single(theta, sim_seed))
            except ChasteRunError as exc:
                n_fail += 1
                last_err = str(exc)
                if strict_any_fail:
                    raise ChasteRunError(
                        f"HF strict any-fail: replicate {replicate_k + 1}/{k} failed at theta={theta}; {last_err}"
                    ) from exc
                continue
        if not xs:
            raise ChasteRunError(f"All {k} HF replicates failed at theta={theta}; {last_err}")
        return np.mean(np.stack(xs, axis=0), axis=0)
