'''
Figure "materials": a CVDomes car rendered with each of the two .mp files convert_cvdomes.py writes
next to its .obj -- one constant PEC material, and the per-triangle CVDomes materials. Writes the
four subfloats, first-bounce energy of the center pulse on top and the SAR image below, as
figures/cvdomes_materials_<material>_<row>.png, plus a titled stitch of all four, and prints the
per-triangle vs constant levels the paper quotes.

    python convert_cvdomes.py
    python render_cvdomes.py
'''
import os

import numpy as np
import torch

import sarsonar
from sarsonar.display import db_compress

from common import CVDOMES_DIR, default_device, figures_path
from convert_cvdomes import PEC_RAIDS, cvdomes_ground_raids
from paper_figure_layout import stitch_panels

OBJ_NAME = 'Camry_06212012.obj'

# .mp suffix -> panel title
MATERIALS = {'_constant': 'constant', '': 'per-triangle'}
SAR_DB_FLOOR = -40

CENTER_AZIMUTH   = 210  # degrees
CENTER_ELEVATION = 32   # degrees
SENSOR_DISTANCE  = 1.3

RESOLUTION_MM = 100
SEED = 0  # of the receiver noise

MESH_KWARGS = dict(
    ground       = True,
    normalize    = False,
    scale        = 0.05,
    obj_raids    = PEC_RAIDS,   # unused, every render has a .mp
    ground_raids = cvdomes_ground_raids(),
    x_flip       = False,
    rotate_xyz   = (90.0, 0.0, 90.0),
)

SAR_KWARGS = dict(
    num_pulses           = 30,
    azimuth_spread_deg   = 90,
    trajectory_type      = 'circular',
    trajectory_noise_var = 0,
    spatial_bw           = 3680 / RESOLUTION_MM,
    spatial_fs           = 3680 / RESOLUTION_MM,
    waveform             = 'sinc',
    wavelength           = 0.5,
    use_sig_magnitude    = False,
    snr_db               = 50,
    image_width          = 128,
    image_height         = 128,
    image_plane_width    = 1,
    image_plane_height   = 1,
    grid_width           = 1.2,
    grid_height          = 1.2,
    n_ray_width          = 128,
    n_ray_height         = 128,
    region_radius        = 1.7,
    imaging_algorithm    = 'cbp',
    signal_interpolation = 'bilinear',
    cbp_batch_size       = 4096,
    num_bounce           = 2,
)


def render(mesh, pose):
    '''First-bounce energy (H,W) of the center pulse and the SAR image amplitude (H,W).'''
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sar_image, pulses = sarsonar.sar_image(mesh, pose, return_pulses=True, **SAR_KWARGS)
    energy = pulses['first_bounce_maps'][(0, SAR_KWARGS['num_pulses'] // 2)]['energy'].cpu().numpy()
    return energy, np.abs(sar_image[0].detach().cpu().numpy())


def main():
    device = default_device()
    obj_path = os.path.join(CVDOMES_DIR, OBJ_NAME)
    pose = sarsonar.pose_matrix(CENTER_AZIMUTH, CENTER_ELEVATION, SENSOR_DISTANCE, device=device).reshape(1, 4, 4)

    energies, sar_images = [], []
    for suffix in MATERIALS:
        mp_path = os.path.splitext(obj_path)[0] + suffix + '.mp'
        mesh = sarsonar.load_mesh(obj_path, material_file=mp_path, device=device, **MESH_KWARGS)
        energy, sar_image = render(mesh, pose)
        energies.append(energy)
        sar_images.append(sar_image)

    # both columns share one scale per row, so the materials compare by level as well as shape
    energy_peak = max(e.max() for e in energies)
    sar_peak = max(a.max() for a in sar_images)
    print('per-triangle re constant: SAR peak %.1f dB, SAR energy %.1f dB, first-bounce energy %.1f dB' % (
        20 * np.log10(sar_images[1].max() / sar_images[0].max()),
        10 * np.log10((sar_images[1] ** 2).sum() / (sar_images[0] ** 2).sum()),
        10 * np.log10(energies[1].sum() / energies[0].sum())))
    panels = [e / energy_peak for e in energies] + [db_compress(a, sar_peak, SAR_DB_FLOOR) for a in sar_images]
    keys = ['%s_%s' % (t, row) for row in ('energy', 'sar') for t in MATERIALS.values()]
    titles = [k.replace('_', ', ') for k in keys]
    style = dict(vmin=[0, 0, SAR_DB_FLOOR, SAR_DB_FLOOR], vmax=[1, 1, 0, 0],
                 cbar_label=['energy / shared peak'] * 2 + ['dB re shared peak'] * 2,
                 cbar_tick_fmt=['%.2g', '%.2g', '%.0f dB', '%.0f dB'])
    name = OBJ_NAME.split('_')[0]
    stitch_panels(panels, titles, figures_path('cvdomes_%s_materials.png' % name), ncols=2, **style)

    # one file per subfloat, the titles go in the caption
    for i, key in enumerate(keys):
        stitch_panels([panels[i]], [''], figures_path('cvdomes_materials_%s.png' % key),
                      **{k: v[i] for k, v in style.items()})


if __name__ == '__main__':
    main()
