from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SUMMARY_DIM = 8
SUMMARY_NAMES: tuple[str, ...] = (
    "mean_log_ell",
    "std_log_ell",
    "mean_c2",
    "mean_s2",
    "ell_c2_cov",
    "mean_log_A",
    "std_log_A",
    "frac_hex",
)


@dataclass(frozen=True)
class GeometrySummaryResult:
    s: np.ndarray
    n_cells: int
    n_internal_cells: int
    n_internal_edges: int
    median_area: float
    length_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "s": self.s.tolist(),
            "s_names": list(SUMMARY_NAMES),
            "n_cells": self.n_cells,
            "n_internal_cells": self.n_internal_cells,
            "n_internal_edges": self.n_internal_edges,
            "median_area": self.median_area,
            "length_scale": self.length_scale,
        }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n % 2 == 1:
        return float(v[n // 2])
    return float(0.5 * (v[n // 2 - 1] + v[n // 2]))


def _std_sample(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.sum((v - mean) ** 2) / (len(values) - 1)))


def _shoelace_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def summarise_polygons(
    nodes: np.ndarray,
    elements: Sequence[Sequence[int]],
    *,
    element_is_internal: Sequence[bool] | None = None,
    edge_is_boundary: Mapping[tuple[int, int], bool] | None = None,
) -> GeometrySummaryResult:
    """Compute 8D summary from node coords and element connectivity.

    If ``element_is_internal`` is None, infer like Chaste: an element is internal
    iff none of its edges is a boundary edge (shared by ≤1 element).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elems = [list(map(int, e)) for e in elements]
    n_cells = len(elems)

    # Edge -> list of element indices that contain it
    edge_elems: dict[tuple[int, int], list[int]] = {}
    for ei, elem in enumerate(elems):
        n = len(elem)
        for i in range(n):
            a, b = elem[i], elem[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            edge_elems.setdefault(key, []).append(ei)

    def is_boundary_edge(a: int, b: int) -> bool:
        key = (a, b) if a < b else (b, a)
        if edge_is_boundary is not None and key in edge_is_boundary:
            return bool(edge_is_boundary[key])
        return len(edge_elems.get(key, [])) <= 1

    if element_is_internal is None:
        internal_flags = []
        for elem in elems:
            n = len(elem)
            ok = True
            for i in range(n):
                if is_boundary_edge(elem[i], elem[(i + 1) % n]):
                    ok = False
                    break
            internal_flags.append(ok)
    else:
        internal_flags = [bool(x) for x in element_is_internal]

    areas: list[float] = []
    internal_indices: list[int] = []
    n_hex = 0
    for ei, elem in enumerate(elems):
        if not internal_flags[ei]:
            continue
        xy = nodes[np.asarray(elem, dtype=int)]
        area = abs(_shoelace_area(xy))
        if not (area > 0.0 and np.isfinite(area)):
            continue
        areas.append(area)
        internal_indices.append(ei)
        if len(elem) == 6:
            n_hex += 1

    n_internal = len(internal_indices)
    median_area = _median(areas)
    length_scale = float(np.sqrt(max(median_area, 1e-16)))

    edges: dict[tuple[int, int], tuple[float, float]] = {}
    for ei in internal_indices:
        elem = elems[ei]
        n = len(elem)
        for i in range(n):
            a, b = elem[i], elem[(i + 1) % n]
            if is_boundary_edge(a, b):
                continue
            key = (a, b) if a < b else (b, a)
            if key in edges:
                continue
            vec = nodes[b] - nodes[a]
            ell = float(np.linalg.norm(vec) / length_scale)
            theta = float(np.arctan2(vec[1], vec[0]))
            edges[key] = (ell, theta)

    log_ell: list[float] = []
    c2: list[float] = []
    s2: list[float] = []
    ell_nd: list[float] = []
    for ell, theta in edges.values():
        if not (ell > 0.0 and np.isfinite(ell)):
            continue
        ell_nd.append(ell)
        log_ell.append(float(np.log(ell)))
        c2.append(float(np.cos(2.0 * theta)))
        s2.append(float(np.sin(2.0 * theta)))

    log_A = [float(np.log(max(a / max(median_area, 1e-16), 1e-16))) for a in areas]

    def mean_of(v: Sequence[float]) -> float:
        return float(np.mean(v)) if v else 0.0

    mean_log_ell = mean_of(log_ell)
    mean_c2_v = mean_of(c2)
    mean_s2_v = mean_of(s2)
    mean_ell = mean_of(ell_nd)
    mean_log_A = mean_of(log_A)

    cov = 0.0
    if len(ell_nd) >= 2:
        acc = 0.0
        for i in range(len(ell_nd)):
            acc += (ell_nd[i] - mean_ell) * (c2[i] - mean_c2_v)
        cov = acc / (len(ell_nd) - 1)

    frac_hex = (float(n_hex) / float(n_internal)) if n_internal else 0.0
    s = np.asarray(
        [
            mean_log_ell,
            _std_sample(log_ell, mean_log_ell),
            mean_c2_v,
            mean_s2_v,
            cov,
            mean_log_A,
            _std_sample(log_A, mean_log_A),
            frac_hex,
        ],
        dtype=np.float64,
    )
    return GeometrySummaryResult(
        s=s,
        n_cells=n_cells,
        n_internal_cells=n_internal,
        n_internal_edges=len(edges),
        median_area=median_area,
        length_scale=length_scale,
    )


@dataclass
class VDatMesh:
    nodes: np.ndarray
    cells: list[list[int]]
    cell_ext: list[bool]
    edge_ext: dict[tuple[int, int], bool]
    meta: dict[str, Any]


_J_RE = re.compile(r"^J\[(\d+)\]\s+([^\s]+)\s+([^\s]+)")
_E_RE = re.compile(r"^E\[(\d+)\]\s+(\d+)\s+(\d+)")
_C_RE = re.compile(r"^C\[(\d+)\]\s+(\d+)\s*:\s*(.*)$")


def parse_vdat(path: str | Path) -> VDatMesh:
    """Parse GetVertex / Yan / SSBD ``VDat_*.dat`` files."""
    path = Path(path)
    nodes_dict: dict[int, tuple[float, float]] = {}
    cells: dict[int, list[int]] = {}
    cell_ext: dict[int, bool] = {}
    edge_ext: dict[tuple[int, int], bool] = {}
    meta: dict[str, Any] = {"path": str(path)}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "C_NUM" in line and "IN_" not in line and "EX_" not in line:
                    parts = line.split()
                    meta["C_NUM"] = int(parts[2])
                elif "IN_CNUM" in line:
                    meta["IN_CNUM"] = int(line.split()[2])
                elif "EX_CNUM" in line:
                    meta["EX_CNUM"] = int(line.split()[2])
                elif "E_NUM" in line and "IN_" not in line and "EX_" not in line:
                    meta["E_NUM"] = int(line.split()[2])
                elif "V_NUM" in line and "IN_" not in line and "EX_" not in line:
                    meta["V_NUM"] = int(line.split()[2])
                continue

            if line.startswith("J["):
                m = _J_RE.match(line)
                if not m:
                    continue
                vid = int(m.group(1))
                nodes_dict[vid] = (float(m.group(2)), float(m.group(3)))
                continue

            if line.startswith("E["):
                m = _E_RE.match(line)
                if not m:
                    continue
                a, b = int(m.group(2)), int(m.group(3))
                key = (a, b) if a < b else (b, a)
                edge_ext[key] = "Ext" in line
                continue

            if line.startswith("C["):
                m = _C_RE.match(line)
                if not m:
                    continue
                cid = int(m.group(1))
                rest = m.group(3).strip()
                is_ext = "Ext" in rest
                if is_ext:
                    rest = rest.replace("Ext", "").strip()
                junc = [int(tok) for tok in rest.split() if tok.isdigit()]
                cells[cid] = junc
                cell_ext[cid] = is_ext

    if not nodes_dict:
        raise ValueError(f"No vertices in VDat: {path}")
    max_id = max(nodes_dict)
    nodes = np.zeros((max_id + 1, 2), dtype=np.float64)
    for vid, xy in nodes_dict.items():
        nodes[vid] = xy

    n_cells = meta.get("C_NUM", (max(cells) + 1) if cells else 0)
    elem_list: list[list[int]] = []
    ext_list: list[bool] = []
    for cid in range(n_cells):
        if cid not in cells:
            # Some files may skip ids; keep placeholder empty (filtered later).
            elem_list.append([])
            ext_list.append(True)
            continue
        elem_list.append(cells[cid])
        ext_list.append(bool(cell_ext.get(cid, False)))

    return VDatMesh(
        nodes=nodes,
        cells=elem_list,
        cell_ext=ext_list,
        edge_ext=edge_ext,
        meta=meta,
    )


def summarise_vdat(path: str | Path) -> GeometrySummaryResult:
    """8D summary for a Yan/SSBD VDat file (internal = non-Ext cells)."""
    mesh = parse_vdat(path)
    # Drop empty placeholders
    keep = [(c, ext) for c, ext in zip(mesh.cells, mesh.cell_ext) if len(c) >= 3]
    if not keep:
        raise ValueError(f"No usable cells in {path}")
    cells = [c for c, _ in keep]
    # VDat Ext flag: internal cells are not Ext. Boundary edges from VDat Ext tags.
    element_is_internal = [not ext for _, ext in keep]
    return summarise_polygons(
        mesh.nodes,
        cells,
        element_is_internal=element_is_internal,
        edge_is_boundary=mesh.edge_ext,
    )


def summarise_mesh_json(path: str | Path) -> GeometrySummaryResult:
    """8D summary for Chaste ``mesh_*.json`` dumps (nodes + elements)."""
    data = json.loads(Path(path).read_text())
    nodes = np.asarray(data["nodes"], dtype=np.float64)
    elements = data["elements"]
    return summarise_polygons(nodes, elements)


def summarise_vdat_vector(path: str | Path) -> np.ndarray:
    return summarise_vdat(path).s.copy()
