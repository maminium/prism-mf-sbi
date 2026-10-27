# Non-proliferating vertex tissue simulator

Chaste user project **NonproliferatingVertex** for the anisotropic epithelial
mechanics model used as the non-proliferating / transfer case in the prism-mf-sbi
companion repository.

## Model

- Short mechanical relaxation **without proliferation**, adapted from the
  anisotropic epithelial constitutive law used in image-based / Bayesian
  calibration of pupal-wing geometry (Ogita et al. 2022; Yan et al. 2025).
- **Entry point:** `test/TestYanVertexCommandLine.hpp`
- **HF labels for inference:** mean of **\(K=10\) independent HF runs** (same
  protocol as the proliferating model).
- **Observation:** processed five-dimensional geometry summary of published
  pupal-wing snapshots (Yan et al.); raw images/meshes are **not** redistributed.

## Build

```bash
../../setup_chaste_project.sh ~/chaste
cd ~/chaste/build && cmake -DCMAKE_BUILD_TYPE=Release ~/chaste
make -j project_NonproliferatingVertex TestYanVertexCommandLine
```

See the repository README for the shared build steps.

## Data policy

Ship only processed derivatives under `data/nonproliferating_vertex/`. Obtain
original tissue images/meshes from the Yan et al. source if needed.

## References

1. **Ogita et al. (2022).** *Image-based parameter inference for epithelial mechanics.*  
   *PLOS Computational Biology* 18(6):e1010209.  
   https://doi.org/10.1371/journal.pcbi.1010209

2. **Yan et al. (2024).** *Bayesian parameter inference for epithelial mechanics.*  
   *Journal of Theoretical Biology* 595:111960.  
   https://doi.org/10.1016/j.jtbi.2024.111960  
   (Empirical pupal-wing geometry summaries used as the observation.)

3. **Cooper, Spiteri & Mirams (2020).** *Chaste: Cancer, Heart and Soft Tissue Environment.*  
   *Journal of Open Source Software* 5(47):1848.  
   https://doi.org/10.21105/joss.01848  
   https://chaste.github.io/
