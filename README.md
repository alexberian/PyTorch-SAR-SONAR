# pytorch-sar-sonar

Ray traced **synthetic aperture radar**, **side scan sonar** and **forward looking sonar** images of
triangle meshes, in pure PyTorch. It runs on CPU or any CUDA GPU, with no compiled extensions.

```bash
pip install pytorch-sar-sonar
```

Install PyTorch first for your platform and CUDA version (https://pytorch.org/get-started/locally/);
otherwise pip pulls the default build.

## Quick start

```bash
sarsonar car.obj -m sss -o side_scan.png --azimuth 30 --elevation 40
sarsonar car.obj -m fls -o forward_looking.png --azimuth 30 --elevation 40
sarsonar car.obj -m sar -o sar.png --azimuth 30 --elevation 40
```

```python
import sarsonar

mesh = sarsonar.load_mesh('car.obj', device='cuda')    # y-up .obj, centered, unit diagonal, on a ground plane

sar = sarsonar.render_sar(mesh, azimuth_deg=30, elevation_deg=40)        # (128,128) raw amplitude
sss = sarsonar.render_side_scan(mesh, azimuth_deg=30, elevation_deg=40)
fls = sarsonar.render_fls(mesh, azimuth_deg=30, elevation_deg=40)

sarsonar.save_png(sss, 'side_scan.png', compression='db')   # 'db', 'asinh' or 'linear'
```

Any renderer argument can be changed from the command line with `--set`, e.g.
`--set num_pings=256 --set "waveform='lfm'"`. Run `sarsonar --help` for the rest.
`examples/toy_car.obj` is a small mesh to try it on.

## Modalities

| | sensor | image rows | image columns | `render_*` default distance |
|---|---|---|---|---|
| `render_sar` | spotlight SAR, a linear aperture of planar wavefront pulses, back projected | ground plane | ground plane | 1.3 |
| `render_side_scan` | straight track abeam of the object, one ray fan per ping | slant range | along track | 7.5 |
| `render_fls` | one ray fan, beam steered across azimuth | range | azimuth | 7.5 |

Every default is the baseline from the paper. `sar_image`, `side_scan_image` and `fls_image`
take explicit poses or sensor positions and document every parameter.

## Conventions

* **World:** z-up, with the object at the origin. Azimuth runs counterclockwise from +x; elevation is measured up from the ground plane.
* **`load_mesh`:**
  * Rotates y-up `.obj` files to z-up by default; pass `rotate_xyz=(0, 0, 0)` for files that are already z-up.
  * Centers the object and scales its bounding box diagonal to 1, the size of a ShapeNet normalized model, which the default sensor geometries frame. Pass `normalize=False` to keep your own units, and scale the sensor geometry to match.
  * Stands the object on a 200 × 200 ground plane by default (`ground=False` to drop it).
* **Materials:** each triangle carries `r, a, i, d, s` weights:
  * `r` reflectivity, `a` absorption, `s` scattering, normalized so `r + a + s = 1`
  * `i` directional and `d` diffuse scattering, normalized so `i + d = 1`

  Set them per object with `obj_raids` / `ground_raids`. For per-triangle values, put a `.mp` file next to the `.obj` with one `r a i d s` line per triangle, after fan triangulation of any polygons.

## Tests

```bash
pip install -e ".[test]"
pytest
```

`tests/test_golden.py` checks the scatter accumulators against fingerprints recorded from the
implementation the paper's figures were rendered with.

## Citation

If you use this in your research, please cite:

```bibtex
TODO
```

## License

PolyForm Noncommercial 1.0.0. Free for noncommercial research use; for a commercial license,
contact berian@arizona.edu.
