# Paper experiments

These scripts produce the paper's figures and tables using the `sarsonar` package. Each one runs with
no arguments and reads its defaults from [`config.json`](config.json). Those defaults are the
SAR, side scan and forward looking sonar baselines in the paper's Default Parameters appendix.
Output goes to `experiments/figures/`, or to `$SARSONAR_FIGURES_DIR` if it is set.

```bash
pip install "pytorch-sar-sonar[experiments]"   # or, from a clone: pip install -e ".[experiments]"
cd experiments
python sar_overview_figure.py
```

## Data

| dataset | used for | `config.json` key | environment variable |
|---|---|---|---|
| srncars ([Scene Representation Networks](https://github.com/vsitzmann/scene-representation-networks)): `cars_train/val/test/<obj_id>/{rgb,pose}` | object ids, poses and RGB views | `srn_cars_dir` | `SRN_CARS_DIR` |
| [ShapeNetCore v2](https://shapenet.org) cars, `02958343/<obj_id>/models/model_normalized.obj` | the meshes | `shapenet_cars_dir` | `SHAPENET_CARS_DIR` |
| CVDomes CAD models as `.obj` with their Blender material names | Per-Triangle Materials only | `cvdomes_dir` | `CVDOMES_DIR` |

Relative paths are resolved against this folder. The default layout is `experiments/data/...`.
The sphere in the sphere-size figure is `tests/data/sphere.obj`.

## Scripts

Approximate times are for one NVIDIA L40S.

| paper | script | writes | time |
|---|---|---|---|
| Fig. modality comparison | `modality_comparison.py` | `modality_comparison.png` | 10 min |
| Fig. 1 (SAR overview) | `sar_overview_figure.py` | `sar_overview/sar_overview*.png` | 2 min |
| Fig. effective windows | `window_analysis.py` | `window_analysis_bw10_fs2000.png` | seconds |
| Fig. materials | `convert_cvdomes.py`, then `render_cvdomes.py` | `.mp` files beside each CVDomes `.obj`; `cvdomes_materials_*.png` | 1 min |
| Fig. validation, Fig. RCS (c), and the Validation numbers | `sar_validation.py` | `sar_validation.png`; numbers printed | 1 min |
| Fig. RCS (a, b) | `calculate_rcs.py` | `rcs/rcs_flatplate_polar.png`, `rcs/rcs_polar.png` | 40 min |
| Fig. sphere size; Appendix SAR examples | `sar_paper_figures.py` | `sar_stitched_<sweep>.png` | 2.5 h |
| Figs. bandwidth, ray density, scatter close-up | `fsbw_and_raydensity_exp.py` | `fsbw_and_raydensity/*.png` | 1 h |
| Table compute time | `compute_time_experiment.py` | `compute_time_experiment.json`; LaTeX rows printed | 50 min |
| Fig. side-scan azimuth rays; Appendix sonar beam width | `sss_paper_figures.py` | `side_scan_stitched_<sweep>.png` | 5 min |
| Fig. sonar range density; Appendix sonar beam width | `fls_paper_figures.py` | `fls_stitched_<sweep>.png` | 10 min |
| Fig. transmit waveform | `waveform_figure.py` | `waveform_stitched_sar_sss.png` | 12 min |

Running a sweep script with no arguments renders every sweep in it. Name sweeps to render only those:
`python sar_paper_figures.py wavelength snrdb`. `-replot` rebuilds a figure from the raw
amplitudes saved on an earlier run, without rendering again. `fsbw_and_raydensity_exp.py` and
`waveform_figure.py` also take `-replot`. `calculate_rcs.py` takes its map names, `plate` and
`car`, and `-replot` the same way.

`compute_time_experiment.py -quick` checks the timing harness in about a minute. The full run's
numbers depend on the GPU and CPU.

## Shared code

* `common.py` handles `config.json`, srncars lookups, and one render per modality from a baseline
  dict, e.g. `render_sar({**SAR_BASELINE, 'wavelength': 0.05}, seed=8134)`. It also holds the sweep runner.
* `paper_figure_layout.py` is the stitched-panel layout and the per-panel display compression.
* `validation_point_target.py` and `validation_plate.py` are the point target and flat plate
  harnesses that `sar_validation.py` runs. It also takes the plate's elevation cut from
  `calculate_rcs.py`.

The SAR panels seed numpy and torch before each render, so the receiver noise (`snr_db`) and
trajectory noise come out the same on every run.
