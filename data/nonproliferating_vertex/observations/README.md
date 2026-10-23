# Empirical observation (pupal wing)

Processed summaries from **23** published segmented images of the *Drosophila*
pupal wing (16.5–18.5 h APF). These arrays are the empirical observation
`x_o` used in the non-proliferating experiment (Figs 6–7).

| File | Role |
|---|---|
| `x_o_w165_all.npy` | Per-image features `(23, 8)` |
| `x_o_w165_mean.npy` / `_std.npy` / `_sem.npy` | Mean / SD / SEM over images |
| `x_o_w165_out5d.npy` | 5-D observation used for inference in the paper |
| `x_o_w165_out6d.npy` | 6-D observation used to define `theta_hat` |
| `observation_metadata.json` | Feature names and provenance |

See also `../reference/` for the per-image summary table and the data-matched
reference parameter `theta_hat`.
