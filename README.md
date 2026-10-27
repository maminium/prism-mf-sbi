# prism-mf-sbi

Companion code and processed data for the manuscript:

**Low-fidelity simulator selection for multi-fidelity inference of epithelial vertex models**

Mohammad Amini, Jack L. Jennings, Wei W. Xing, Alexander G. Fletcher

The repository implements **PRISM** (Paired Ranking Index for Simulator Matching) to rank candidate low-fidelity (LF) Chaste vertex simulators before neural training, together with high-fidelity (HF) baselines and **multi-fidelity simulation-based inference (MF-SBI)**. Processed tables and posteriors under `data/` match the Zenodo companion deposit so **Figs 1–8** can be regenerated without rerunning large-scale jobs.

| Model | Experiment id | Chaste project |
| --- | --- | --- |
| Proliferating vertex | `proliferating_vertex` | `chaste/proliferating_vertex` |
| Non-proliferating vertex | `nonproliferating_vertex` | `chaste/nonproliferating_vertex` |

**Code:** https://github.com/maminium/prism-mf-sbi

## Requirements

- Python **3.11+**
- Dependencies are declared in `pyproject.toml` (PyTorch, [sbi](https://github.com/mackelab/sbi), NumPy, pandas, SciPy, Matplotlib, PyYAML)
- **Optional:** a local [Chaste](https://chaste.github.io/) build only if you want new forward simulations or full inference smoke tests (not needed to reproduce figures)

## Reproduce figures

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
./reproduce_figures.sh
```

Outputs: `figures/fig1.png`–`fig8.png`, plus TIFF and EPS under `figures/TIFF/` and `figures/EPS/`.

## Repository layout

```text
chaste/              Chaste user projects (not a full Chaste install)
configs/             Experiment and method YAML (paper simulation budgets)
data/                Processed inputs for Figs 1–8 (Zenodo-aligned)
figures/             Manuscript figures (PNG / TIFF / EPS)
prism_vertex/        Python package: simulators, PRISM, inference
scripts/
  figures/           plot_fig1 … plot_fig8, plot_all
  inference/         run_method, compute_metrics
  lf_selection/      PRISM runners
  verify_pipeline.py End-to-end checks (Chaste required for sim steps)
```

## Inference (optional)

After building the Chaste test executables (see below), point the environment at your binaries and run a method:

```bash
export PRISM_VERTEX_CHASTE_EXE=~/chaste/build/projects/ProliferatingVertex/test/TestPaperCommandLineSpeedSimulation
export PRISM_VERTEX_CHASTE_CWD=~/chaste
export PRISM_VERTEX_TESTOUTPUT_ROOT=~/chaste/build/testoutput

python scripts/inference/run_method.py \
  --experiment proliferating_vertex \
  --method mf_sbi \
  --obs_id 0 --seed 0 \
  --budget_hf 500 --budget_lf 10000 \
  --num_posterior_samples 2000
```

**Methods:** `hf_snpe`, `hf_npe`, `hf_smc_abc`, `mf_sbi`

**Paper budgets:** HF-only baselines use `N_HF = 1000`; MF-SBI uses `N_HF = 500` with LF budgets `{500, 1000, 5000, 10000}`. HF training labels are means over `K = 10` independent runs; LF `K_2` / `K_5` configurations average 2 or 5 replicates.

For the non-proliferating model, set `PRISM_VERTEX_NONPROLIF_CHASTE_EXE` to your built `TestYanVertexCommandLine`.

## Chaste setup

This repo is **not** a Chaste installation. Install Chaste separately, then link these projects:

```bash
./setup_chaste_project.sh ~/chaste
cd ~/chaste/build && cmake -DCMAKE_BUILD_TYPE=Release ~/chaste
make -j project_ProliferatingVertex TestPaperCommandLineSpeedSimulation
make -j project_NonproliferatingVertex TestYanVertexCommandLine
```

Model-specific build notes and literature references: `chaste/proliferating_vertex/README.md`, `chaste/nonproliferating_vertex/README.md`.

## Data

Files under `data/` are **processed derivatives** (summary statistics, metrics, posterior samples, calibration SBC/TARP tables) aligned with the Zenodo deposit. Original segmented tissue images from Yan et al. are not redistributed here.

## Verify installation

```bash
python scripts/verify_pipeline.py
```

Runs import checks, short Chaste forward simulations (when executables are configured), smoke tests for all four inference methods, and confirms Figs 1–8 plus export formats.

## Citation and license

If you use this software, please cite the accompanying manuscript (metadata in [`CITATION.cff`](CITATION.cff)).

Released under the **BSD 3-Clause License** — see [`LICENSE`](LICENSE).
