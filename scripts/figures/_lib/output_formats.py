"""Paper figure output paths."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from prism_vertex.benchmark.comparison_paths import results_dir, fig4_comparison_dir

MANUSCRIPT_BUILD_DIR = ".build"
MANUSCRIPT_EPS_DIR = "EPS"
MANUSCRIPT_TIFF_DIR = "TIFF"

# PLOS Computational Biology submission limits
PLOS_TIFF_MAX_BYTES = 10 * 1024 * 1024
SUBMISSION_TIFF_DPI = 300
# Manuscript PNG previews: 300 ppi matches TIFF submission; 600 ppi PNGs are ~4× larger with little gain.
MANUSCRIPT_PNG_DPI = 300

FIG7_BUILD_PANELS: dict[str, str] = {
    "a": "fig7_a_comparison_table.png",
    "c": "fig7_c_hf_failure_map.png",
    "d": "fig7_d_seed_fidelity_failures.png",
}

# Paper figure number -> source content (legacy script names in comments).
PAPER_FIGURE_SOURCES: dict[int | str, str] = {
    1: "fig1",
    2: "fig3_abcdefg",  # PRISM screening
    3: "fig4_abcdefgh",  # scoreboard + post-training metrics
    4: "fig4_main_4panel",  # HF baselines + best MF-SBI LF
    5: "fig5_mf_sbi_calibration",  # TARP/SBC
    6: "fig6_nonproliferating_scoreboard",
    7: "fig7_nonproliferating_posteriors",
    8: "fig8_methods_composite",
}


def manuscript_figures_root(results_root: Path | None = None) -> Path:
    return Path(__file__).resolve().parents[3] / "figures"


def manuscript_figures_dir(exp_dir: Path) -> Path:
    return Path(__file__).resolve().parents[3] / "figures"


def manuscript_build_dir(exp_dir: Path) -> Path:
    return manuscript_figures_dir(exp_dir) / MANUSCRIPT_BUILD_DIR


def manuscript_fig_stem(exp_dir: Path, paper_number: int | str) -> Path:
    if paper_number not in PAPER_FIGURE_SOURCES:
        raise ValueError(f"Unknown paper figure number: {paper_number}")
    return manuscript_figures_dir(exp_dir) / f"fig{paper_number}"


def manuscript_eps_dir(all_dir: Path) -> Path:
    return all_dir / MANUSCRIPT_EPS_DIR


def manuscript_tiff_dir(all_dir: Path) -> Path:
    return all_dir / MANUSCRIPT_TIFF_DIR


def optimize_manuscript_png(png_path: Path) -> Path:
    """Shrink a manuscript PNG with max deflate (lossless)."""
    from PIL import Image  # noqa: WPS433

    png_path = Path(png_path)
    with Image.open(png_path) as img:
        has_alpha = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )
        out = img.convert("RGBA" if has_alpha else "RGB")
        tmp = png_path.with_suffix(".png.opt")
        out.save(
            tmp,
            format="PNG",
            optimize=True,
            compress_level=9,
        )
    tmp.replace(png_path)
    return png_path


def save_matplotlib_manuscript_png(
    fig,
    png_path: Path,
    *,
    dpi: int | None = None,
    save_kwargs: dict | None = None,
) -> Path:
    """Write a compact manuscript PNG at ``MANUSCRIPT_PNG_DPI`` (or lower ``dpi``)."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # Honour explicit dpi (Mo 8 used 600 for several composites); default 300 for previews.
    png_dpi = MANUSCRIPT_PNG_DPI if dpi is None else int(dpi)
    kw = dict(facecolor="white", bbox_inches="tight")
    if save_kwargs:
        kw.update(save_kwargs)
    fig.savefig(
        png_path,
        dpi=png_dpi,
        pil_kwargs={"optimize": True, "compress_level": 9},
        **kw,
    )
    return optimize_manuscript_png(png_path)


def save_manuscript_submission_formats(
    fig,
    out_stem: Path,
    *,
    dpi: int,
    save_kwargs: dict | None = None,
) -> tuple[Path | None, Path | None]:
    """Write PLOS-style EPS + TIFF next to the paper figures."""
    kw = dict(facecolor="white", bbox_inches="tight")
    if save_kwargs:
        kw.update(save_kwargs)

    all_dir = out_stem.parent
    eps_path = manuscript_eps_dir(all_dir) / f"{out_stem.name}.eps"
    tiff_path = manuscript_tiff_dir(all_dir) / f"{out_stem.name}.tiff"
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(eps_path, format="eps", **kw)

    tiff_dpi = min(int(dpi), SUBMISSION_TIFF_DPI)
    for attempt_dpi in (tiff_dpi, 300, 250, 200):
        fig.savefig(
            tiff_path,
            dpi=attempt_dpi,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            **kw,
        )
        if tiff_path.stat().st_size <= PLOS_TIFF_MAX_BYTES:
            break

    if tiff_path.stat().st_size > PLOS_TIFF_MAX_BYTES:
        print(
            f"Warning: {tiff_path} is {tiff_path.stat().st_size / (1024 * 1024):.1f} MB "
            f"(PLOS limit {PLOS_TIFF_MAX_BYTES // (1024 * 1024)} MB)",
            file=sys.stderr,
        )

    return eps_path, tiff_path


def export_external_png_submission_formats(
    png_path: Path,
    *,
    stem_name: str | None = None,
    target_dpi: int = SUBMISSION_TIFF_DPI,
) -> tuple[Path, Path]:
    """Convert an external PNG (e.g. fig1) to PLOS EPS + TIFF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    from PIL import Image  # noqa: WPS433

    png_path = Path(png_path)
    if not png_path.is_file():
        raise FileNotFoundError(f"Missing figure PNG: {png_path}")

    all_dir = png_path.parent
    stem = stem_name or png_path.stem
    eps_path = manuscript_eps_dir(all_dir) / f"{stem}.eps"
    tiff_path = manuscript_tiff_dir(all_dir) / f"{stem}.tiff"
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    rgba = Image.open(png_path).convert("RGBA")
    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba, mask=rgba.split()[3])
    img = np.asarray(rgb, dtype=np.float32) / 255.0

    width_px, height_px = rgb.size
    fig_w_in = width_px / target_dpi
    fig_h_in = height_px / target_dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=target_dpi, facecolor="white")
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    save_kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0)
    fig.savefig(eps_path, format="eps", **save_kw)

    for attempt_dpi in (target_dpi, 300, 250, 200):
        fig.set_dpi(attempt_dpi)
        fig.savefig(
            tiff_path,
            dpi=attempt_dpi,
            format="tiff",
            pil_kwargs={"compression": "tiff_lzw"},
            **save_kw,
        )
        if tiff_path.stat().st_size <= PLOS_TIFF_MAX_BYTES:
            break

    plt.close(fig)

    if tiff_path.stat().st_size > PLOS_TIFF_MAX_BYTES:
        print(
            f"Warning: {tiff_path} is {tiff_path.stat().st_size / (1024 * 1024):.1f} MB "
            f"(PLOS limit {PLOS_TIFF_MAX_BYTES // (1024 * 1024)} MB)",
            file=sys.stderr,
        )

    return eps_path, tiff_path


def _run_subprocess(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {detail}")


def crop_whitespace_rgba(
    img,
    *,
    threshold: int = 250,
    pad: int = 8,
):
    """Tight-crop near-white margins (Mo 8 fig1 / fig8c)."""
    import numpy as np  # noqa: WPS433
    from PIL import Image  # noqa: WPS433

    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL Image")
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    mask = (arr < threshold).any(axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return img
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
    return img.crop((x0, y0, x1, y1))


def rasterise_pdf_cropped_png(
    pdf_path: Path,
    png_path: Path,
    *,
    dpi: int = SUBMISSION_TIFF_DPI,
    threshold: int = 250,
    pad: int = 8,
) -> Path:
    """Rasterise a PDF and tight-crop whitespace into ``png_path``."""
    from PIL import Image  # noqa: WPS433

    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    tmp_stem = png_path.parent / f".{png_path.stem}_pdf_export"
    _run_subprocess(
        [
            "pdftocairo",
            "-png",
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            str(tmp_stem),
        ]
    )
    tmp_png = Path(f"{tmp_stem}.png")
    if not tmp_png.is_file():
        raise FileNotFoundError(f"pdftocairo did not create {tmp_png}")
    with Image.open(tmp_png) as im:
        cropped = crop_whitespace_rgba(im, threshold=threshold, pad=pad)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(png_path, optimize=True, compress_level=9)
    tmp_png.unlink(missing_ok=True)
    return optimize_manuscript_png(png_path)


def export_external_pdf_submission_formats(
    pdf_path: Path,
    *,
    stem_name: str | None = None,
    target_dpi: int = SUBMISSION_TIFF_DPI,
    update_png: bool = False,
    crop_png: bool = True,
) -> tuple[Path, Path, Path | None]:
    """Convert an external PDF to EPS + TIFF.

    By default does **not** overwrite a shipped manuscript PNG (Mo 8 fig1 is a
    cropped raster). Pass ``update_png=True`` to rebuild PNG from the PDF,
    optionally tight-cropped.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing figure PDF: {pdf_path}")

    all_dir = pdf_path.parent
    stem = stem_name or pdf_path.stem
    eps_path = manuscript_eps_dir(all_dir) / f"{stem}.eps"
    tiff_path = manuscript_tiff_dir(all_dir) / f"{stem}.tiff"
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    _run_subprocess(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=eps2write",
            f"-sOutputFile={eps_path}",
            str(pdf_path),
        ]
    )

    final_dpi = target_dpi
    for attempt_dpi in (target_dpi, 300, 250, 200):
        _run_subprocess(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=tiff24nc",
                f"-r{attempt_dpi}",
                "-sCompression=lzw",
                f"-sOutputFile={tiff_path}",
                str(pdf_path),
            ]
        )
        final_dpi = attempt_dpi
        if tiff_path.stat().st_size <= PLOS_TIFF_MAX_BYTES:
            break

    if tiff_path.stat().st_size > PLOS_TIFF_MAX_BYTES:
        print(
            f"Warning: {tiff_path} is {tiff_path.stat().st_size / (1024 * 1024):.1f} MB "
            f"(PLOS limit {PLOS_TIFF_MAX_BYTES // (1024 * 1024)} MB)",
            file=sys.stderr,
        )

    png_path: Path | None = None
    if update_png:
        png_path = all_dir / f"{stem}.png"
        if crop_png:
            rasterise_pdf_cropped_png(pdf_path, png_path, dpi=final_dpi)
        else:
            tmp_stem = all_dir / f".{stem}_pdf_export"
            _run_subprocess(
                [
                    "pdftocairo",
                    "-png",
                    "-r",
                    str(final_dpi),
                    "-singlefile",
                    str(pdf_path),
                    str(tmp_stem),
                ]
            )
            tmp_png = Path(f"{tmp_stem}.png")
            if not tmp_png.is_file():
                raise FileNotFoundError(f"pdftocairo did not create {tmp_png}")
            shutil.move(tmp_png, png_path)
            optimize_manuscript_png(png_path)

    return eps_path, tiff_path, png_path


def manuscript_fig1_png(exp_dir: Path) -> Path:
    """External paper Figure 1 schematic at ``all/fig1.png``."""
    return manuscript_figures_dir(exp_dir) / "fig1.png"


def manuscript_fig1_pdf(exp_dir: Path) -> Path:
    """External paper Figure 1 schematic at ``all/fig1.pdf``."""
    return manuscript_figures_dir(exp_dir) / "fig1.pdf"


def manuscript_fig7_e_image(exp_dir: Path) -> Path:
    """External panel (e) schematic — place as ``all/e.png``."""
    return manuscript_figures_dir(exp_dir) / "e.png"


def ensure_fig7_e_image(exp_dir: Path) -> Path:
    """Return panel (e) image, preferring the repository-level figure input."""
    target = manuscript_fig7_e_image(exp_dir)
    if target.is_file():
        return target
    repo_input = Path(__file__).resolve().parents[3] / "figures" / "e.png"
    if repo_input.is_file():
        return repo_input
    legacy = (
        fig4_comparison_dir(exp_dir)
        / "plots"
        / "max_budget_comparison"
        / "fig 2"
        / "e.png"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if legacy.is_file():
        shutil.copy2(legacy, target)
    return target


def ensure_fig8_e_image(exp_dir: Path) -> Path:
    return ensure_fig7_e_image(exp_dir)

FIG8_BUILD_PANELS = FIG7_BUILD_PANELS
