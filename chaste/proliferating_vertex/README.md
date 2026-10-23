# Proliferating vertex tissue simulator

Chaste user project **ProliferatingVertex** for the proliferating Farhadifar-type
vertex model used in the prism-mf-sbi companion repository.

## Model and summaries

- **Entry point:** `test/TestPaperCommandLineSpeedSimulation.hpp`  
  CLI: `-opt1` … `-opt5` for \((\Lambda, \Gamma)\) and seed.
- **HF reference:** \(t_{\mathrm{end}}=700\), \(\Delta t=0.01\), each training label /
  observation is the **mean of \(K=10\) independent HF runs**.
- **Summary:** six-dimensional packing vector (area ratios for polygon classes
  \(n=4,\ldots,8\) + mean elongation), following Kursawe et al. (2018).

## Build

This folder is **not** a Chaste install. Symlink into your Chaste tree and build:

```bash
../../setup_chaste_project.sh ~/chaste
cd ~/chaste/build && cmake -DCMAKE_BUILD_TYPE=Release ~/chaste
make -j project_ProliferatingVertex TestPaperCommandLineSpeedSimulation
```

See the repository README for the shared build steps.

## References

1. **Farhadifar et al. (2007).** *The influence of cell mechanics, cell-cell interactions, and proliferation on epithelial packing.*  
   *Current Biology* 17(24):2095–2104.  
   https://doi.org/10.1016/j.cub.2007.11.049

2. **Kursawe, Baker & Fletcher (2018).** *Approximate Bayesian computation reveals the importance of repeated measurements for parameterising cell-based models of growing tissues.*  
   *Journal of Theoretical Biology* 443:66–81.  
   https://doi.org/10.1016/j.jtbi.2018.01.020  
   (Same biological model and packing summaries used here.)

3. **Cooper, Spiteri & Mirams (2020).** *Chaste: Cancer, Heart and Soft Tissue Environment.*  
   *Journal of Open Source Software* 5(47):1848.  
   https://doi.org/10.21105/joss.01848  
   https://chaste.github.io/
