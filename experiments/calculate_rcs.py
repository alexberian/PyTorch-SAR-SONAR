"""
Monostatic radar cross section over sensor azimuth and elevation: panels (a) and (b) of the
paper's RCS figure. sar_validation.py draws panel (c) with calculate_rcs and draw_cut from here.

RCS is |sum_k E_k|^2, the squared magnitude of the coherent sum of the scatter phasors that
accumulate_scatters returns for one pulse, in arbitrary units. Every look direction is one pulse of
one orthographic ray grid aimed at the origin, so the scene and octree are built once per map.
Each map is drawn in polar coordinates with theta = az and r = 90 - el, so the zenith is the
center and the horizon the rim.

    plate   (a) a 1 x 1 plate on the xy plane with the car's materials, at wavelength 0.05 on a
            0.25 deg grid over elevation 45-90; the ray grid widens to the plate's corner-on
            width at the baseline ray spacing
    car     (b) the SAR baseline car without the ground, at the baseline wavelength on a 2 deg grid

    python calculate_rcs.py              # both maps
    python calculate_rcs.py car          # only the ones named
    python calculate_rcs.py -replot      # redraw from the saved .npz, no ray tracing
"""
import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt

from sarsonar.geometry import spherical_to_cartesian
from sarsonar.mesh import Mesh, make_ground
from sarsonar.raytrace import build_octree
from sarsonar.scatter import accumulate_scatters

from common import SAR_BASELINE, baseline_mesh, default_device, figures_path


RCS_BASELINE = dict(SAR_BASELINE)

# the paper's maps, written to rcs/<file>.png; wavelength None is the baseline's
MAPS = {
    'plate': dict(flatplate=1.0, wavelength=0.05, step=0.25, el_range=(45.0, 90.0), db_range=65.0,
                  file='rcs_flatplate_polar'),
    'car': dict(flatplate=None, wavelength=None, step=2.0, el_range=(0.0, 90.0), db_range=40.0,
                file='rcs_polar'),
}


def flat_plate_scene(size, baseline, device=None):
    '''A size x size plate on the xy plane (normal +z, centered at the origin) with the baseline's object materials.'''
    verts, faces = make_ground(size, 2, ground_level=0.0, max_triangle_len=size / 8, device=device)
    raids = torch.tensor(baseline['obj_raids'], device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)
    return Mesh.from_tensors(verts, faces, raids)


def flat_plate_baseline(size, baseline):
    '''baseline with the ray grid widened to a size plate's corner-on width, at the same ray spacing.'''
    grid = max(baseline['grid_width'], 1.05 * np.sqrt(2) * size)
    n_ray = int(np.ceil(baseline['n_ray_width'] * grid / baseline['grid_width']))
    return {**baseline, 'grid_width': grid, 'grid_height': grid, 'n_ray_width': n_ray, 'n_ray_height': n_ray}


def calculate_rcs(az_deg, el_deg, baseline=RCS_BASELINE, flatplate=None,
                  sensor_distance=1.3, pulses_per_call=64, device=None):
    '''
    Monostatic RCS at each (az, el) pair.

    inputs:
        az_deg, el_deg (N,): look directions in deg, paired elementwise
        flatplate (float): image a flatplate x flatplate plate on the xy plane instead of the car,
            which is imaged without the ground
        sensor_distance (float): where the ray grid sits; the wavefront is planar, so it only
            has to clear the car
        pulses_per_call (int): look directions per accumulate_scatters call, for the progress line

    outputs:
        sigma (N,): |sum of scatter phasors|^2, arbitrary units
    '''
    device = device or default_device()
    if flatplate is None:
        mesh = baseline_mesh({**baseline, 'make_ground': False}, device=device)
    else:
        mesh = flat_plate_scene(flatplate, baseline, device=device)
    octree = build_octree(mesh)

    az = torch.as_tensor(np.asarray(az_deg), device=device, dtype=torch.float32)
    el = torch.as_tensor(np.asarray(el_deg), device=device, dtype=torch.float32)
    trajectory = spherical_to_cartesian(az, el, torch.full_like(az, sensor_distance))  # (N,3)

    sigma = []
    for start in range(0, len(az), pulses_per_call):
        _, energies, _ = accumulate_scatters(
            mesh, trajectory[start:start + pulses_per_call].unsqueeze(0),
            wavelength=baseline['wavelength'],
            grid_width=baseline['grid_width'], grid_height=baseline['grid_height'],
            n_ray_width=baseline['n_ray_width'], n_ray_height=baseline['n_ray_height'],
            num_bounce=baseline['num_bounce'],
            second_bounce_batch_size=2 ** 18,
            octree=octree,
        )
        sigma += [float(torch.abs(e.sum()) ** 2) if e.numel() else 0.0 for e in energies[0]]
        print('\r  rcs %d/%d look directions' % (len(sigma), len(az)), end='', flush=True)
    print()
    return np.array(sigma)


def po_flat_plate(az_deg, el_deg, size, wavelength):
    '''
    Physical-optics monostatic RCS of a size x size plate on the xy plane: the square-aperture
    sinc^2 x sinc^2 times cos^2 of the angle off normal, which is sin^2(el).
    '''
    az, el = np.radians(az_deg), np.radians(el_deg)
    u, v = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az)       # look direction's x and y
    return (4 * np.pi * size ** 4 / wavelength ** 2 * np.sin(el) ** 2
            * np.sinc(2 * size * u / wavelength) ** 2 * np.sinc(2 * size * v / wavelength) ** 2)


def to_db(sigma, db_range, normalize=False):
    '''RCS in dB (arbitrary reference, or relative to its peak), floored db_range below the peak.'''
    db = 10 * np.log10(np.maximum(sigma, 1e-30))
    if normalize:
        db = db - db.max()
    return np.maximum(db, db.max() - db_range)


def draw_cut(ax, angle_deg, sigmas, wavelengths, db_range, theories=None, xlabel='angle (deg)'):
    '''
    One RCS curve per wavelength on ax, with its physical-optics curve dotted in the same color.
    With theories, every curve is normalized to its own peak, since the simulated level does not
    carry PO's 4 pi A^2 / lambda^2 gain.
    '''
    normalize = theories is not None
    top = -np.inf
    for i, (sigma, lam) in enumerate(zip(sigmas, wavelengths)):
        db = to_db(sigma, db_range, normalize)
        top = max(top, db.max())
        ax.plot(angle_deg, db, color='C%d' % i, lw=1.2, label='λ = %g' % lam)
        if normalize:
            ax.plot(angle_deg, to_db(theories[i], db_range, normalize), color='C%d' % i, lw=1.2,
                    ls=':', label='λ = %g, theoretical RCS' % lam)
    ax.set_ylim(top - db_range, top + 0.05 * db_range)
    ax.set_xlim(angle_deg.min(), angle_deg.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel('normalized RCS (dB)' if normalize else 'RCS (dB, arb.)')
    ax.grid(alpha=0.3)
    if len(wavelengths) > 1 or normalize:
        ax.legend(fontsize=7, loc='lower left')


def plot_map(az, el, sigma, db_range, out_path):
    '''sigma (E,A) on the az/el grid, with theta = az and r = 90 - el.'''
    fig, ax = plt.subplots(figsize=(5.5, 4.6), subplot_kw=dict(projection='polar'))
    quad = ax.pcolormesh(np.radians(az), 90 - el, to_db(sigma, db_range),
                         shading='nearest', cmap='viridis')
    r_ticks = np.arange(0, 90 - el.min() + 1e-6, 30 if el.min() < 45 else 15)
    ax.set_rticks(r_ticks)
    ax.set_yticklabels(['%d°' % (90 - r) for r in r_ticks], fontsize=8)
    ax.set_rlim(0, 90 - el.min())
    ax.set_rlabel_position(22.5)
    fig.colorbar(quad, ax=ax, pad=0.1, label='RCS (dB, arb.)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_map(name, replot=False):
    '''Ray trace one of MAPS, or reload it with replot, and draw it.'''
    m = MAPS[name]
    baseline = dict(RCS_BASELINE)
    if m['wavelength'] is not None:
        baseline['wavelength'] = m['wavelength']
    if m['flatplate'] is not None:
        baseline = flat_plate_baseline(m['flatplate'], baseline)
    npz_path = figures_path('rcs', m['file'] + '.npz')
    png_path = figures_path('rcs', m['file'] + '.png')

    if replot:
        with np.load(npz_path) as f:
            az, el, sigma = f['az'], f['el'], f['sigma']
    else:
        el_all = np.arange(m['el_range'][0], m['el_range'][1] + 1e-6, m['step'])
        el, az = np.meshgrid(el_all, np.arange(0.0, 360.0, m['step']), indexing='ij')   # (E,A)
        print('=== %s: %d look directions, wavelength %g, %d bounces, %dx%d rays ==='
              % (name, az.size, baseline['wavelength'], baseline['num_bounce'],
                 baseline['n_ray_width'], baseline['n_ray_height']))
        sigma = calculate_rcs(az.ravel(), el.ravel(), baseline=baseline,
                              flatplate=m['flatplate']).reshape(az.shape)
        np.savez(npz_path, az=az, el=el, sigma=sigma, wavelength=baseline['wavelength'])

    plot_map(az[0], el[:, 0], sigma, m['db_range'], png_path)
    k = np.unravel_index(np.argmax(sigma), sigma.shape)
    print('%s: peak %.2f dB (arb.) at az %.1f el %.1f; saved %s'
          % (name, 10 * np.log10(sigma[k]), az[k], el[k], png_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('names', nargs='*', help='maps to run, of: %s (default: all)' % ', '.join(MAPS))
    parser.add_argument('-replot', action='store_true', help='redraw from the saved .npz, no ray tracing')
    args = parser.parse_args()
    unknown = set(args.names) - set(MAPS)
    if unknown:
        parser.error('unknown map(s) %s; choose from %s' % (', '.join(sorted(unknown)), ', '.join(MAPS)))
    for name in args.names or MAPS:
        run_map(name, replot=args.replot)


if __name__ == '__main__':
    main()
