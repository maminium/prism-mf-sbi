#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

from prism_vertex.benchmark.comparison_paths import results_dir  # noqa: E402
from prism_vertex.simulator.scoreboard_analysis import load_cells_frame  # noqa: E402
from prism_vertex.simulator.scoreboard_io import scoreboard_root  # noqa: E402
from scripts.figures._lib.panels_fig7 import (  # noqa: E402
    DPI,
    apply_rcparams,
)
from scripts.figures._lib.panels_fig2 import (  # noqa: E402
    FIG3_TITLE_COLOR,
    FIG3_TITLE_FONTSIZE,
    panel_title,
)
from scripts.figures._lib.output_formats import (  # noqa: E402
    FIG8_BUILD_PANELS,
    ensure_fig8_e_image,
    manuscript_build_dir,
    manuscript_fig_stem,
    save_manuscript_submission_formats,
    save_matplotlib_manuscript_png,
)
from scripts.figures._lib.fig7_panels_build import (  # noqa: E402
    plot_panel_c_merged,
    plot_panel_d,
)

FIG7_COMPOSITE_TITLES: dict[str, str] = {
    "a": "Failure map",
    "b": "Seed $\\times$ fidelity failures",
    "c": "Inference methods",
}


def _composite_panel_title(fig: plt.Figure, ax: plt.Axes, letter: str) -> None:
    bbox = ax.get_position()
    subtitle = FIG7_COMPOSITE_TITLES.get(letter, "")
    fig.text(
        0.5 * (bbox.x0 + bbox.x1),
        bbox.y1 + 0.008,
        panel_title(letter, subtitle),
        ha="center",
        va="bottom",
        fontsize=FIG3_TITLE_FONTSIZE,
        color=FIG3_TITLE_COLOR,
        transform=fig.transFigure,
    )


def _load_panel(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing panel image: {path}")
    return mpimg.imread(path)


def _ensure_fig8_build_panels(exp_dir: Path, *, scoreboard_root_path: Path | None = None) -> Path:
    build = manuscript_build_dir(exp_dir)
    build.mkdir(parents=True, exist_ok=True)
    sb_root = Path(scoreboard_root_path) if scoreboard_root_path is not None else (exp_dir / "lf_selection")
    df = load_cells_frame(sb_root)
    theta = np.load(sb_root / "theta_pilot_list.npy")
    plot_panel_c_merged(df, theta, build / FIG8_BUILD_PANELS["c"])
    plot_panel_d(df, build / FIG8_BUILD_PANELS["d"])
    return build


def plot_fig8_manuscript_composite(
    exp_dir: Path,
    *,
    out_stem: Path | None = None,
    dpi: int = DPI,
    e_image: Path | None = None,
    scoreboard_root_path: Path | None = None,
    rebuild_panels: bool = True,
) -> tuple[Path, Path]:
    apply_rcparams()
    build = _ensure_fig8_build_panels(exp_dir, scoreboard_root_path=scoreboard_root_path) if rebuild_panels else manuscript_build_dir(exp_dir)
    out_stem = out_stem or manuscript_fig_stem(exp_dir, 8)
    e_path = e_image or ensure_fig8_e_image(exp_dir)
    if not e_path.is_file():
        raise FileNotFoundError(
            f"Missing panel (e) image: {e_path}. Place your schematic at all/e.png."
        )

    panel_c = build / FIG8_BUILD_PANELS["c"]
    panel_d = build / FIG8_BUILD_PANELS["d"]

    img_c = _load_panel(panel_c)
    img_d = _load_panel(panel_d)
    img_e = _load_panel(e_path)

    # Height ratios from native panel aspects so panel (C) is not padded with whitespace.
    def _aspect(img: np.ndarray) -> float:
        h, w = img.shape[:2]
        return h / max(w, 1)

    top_h = max(_aspect(img_c), _aspect(img_d))
    bot_h = _aspect(img_e)
    fig = plt.figure(figsize=(10.8, 10.8 * (top_h + bot_h) / 1.15), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[top_h, bot_h],
        hspace=0.10,
        wspace=0.06,
        left=0.02,
        right=0.98,
        top=0.94,
        bottom=0.02,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax, img, letter in (
        (ax_a, img_c, "a"),
        (ax_b, img_d, "b"),
        (ax_c, img_e, "c"),
    ):
        ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.axis("off")
        _composite_panel_title(fig, ax, letter)

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    save_matplotlib_manuscript_png(
        fig,
        png,
        dpi=dpi,
        save_kwargs=dict(bbox_inches="tight", facecolor="white", pad_inches=0.02),
    )
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    save_manuscript_submission_formats(
        fig, out_stem, dpi=dpi, save_kwargs=dict(bbox_inches="tight", pad_inches=0.02)
    )
    plt.close(fig)
    return png, pdf


# Backward-compatible alias
plot_fig2_composite = plot_fig8_manuscript_composite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--out-stem", type=Path, default=None)
    ap.add_argument("--e-image", type=Path, default=None, help="Panel (e) image (default: all/e.png)")
    ap.add_argument("--scoreboard-root", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--no-rebuild-panels", action="store_true")
    args = ap.parse_args()

    exp_dir = results_dir(args.results_root)
    png, pdf = plot_fig8_manuscript_composite(
        exp_dir,
        out_stem=args.out_stem,
        dpi=args.dpi,
        e_image=args.e_image,
        scoreboard_root_path=args.scoreboard_root,
        rebuild_panels=not args.no_rebuild_panels,
    )
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
