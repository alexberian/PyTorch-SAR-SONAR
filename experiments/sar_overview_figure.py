'''
Figure 1 of the paper: one srncars pose rendered in spotlight-mode SAR at the paper baseline.

  (a) the pose's rgb image           (b) the SAR image, in range and cross-range
  (c) first-hit range of each ray    (d) first-bounce return weight E_k of each ray
  (e) energy-range scatter           (f) interposummed signal magnitude

(c)-(f) are one pulse, traced with two bounces, and (e)-(f) are cropped to the inner half of the
simulated window (sensor distance +/- region_radius), i.e. +/- region_radius/2. Each panel is
saved as its own png, figures/sar_overview/sar_overview*.png, for the paper's subfloats; (a) is
the srncars rgb image itself.

    python sar_overview_figure.py
    python sar_overview_figure.py -obj_id <id> -pose_num 000046 -pulse 10
'''
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

from common import SAR_BASELINE, display, figures_path, load_rgb, render_sar
from paper_figure_layout import panel_display


DEFAULT_OBJ_ID   = 'e6846796e15020e02bc9f17412005422'
DEFAULT_POSE_NUM = '000046'
PANEL_SIZE = (3.0, 2.6)  # inches, one paper panel
FONT_SIZE  = 13          # readable at 0.3 of the paper's text width


def render(obj_id, pose_num, baseline=SAR_BASELINE, seed=0, num_bounce=2):
    '''Render the pose once and return everything the panels draw, as numpy.'''
    b = {**baseline, 'obj_id': obj_id, 'pose_num': pose_num, 'num_bounce': num_bounce}
    sar, pulses = render_sar(b, seed=seed, return_pulses=True)

    n_ray = b['n_ray_width'] * b['n_ray_height']
    P = pulses['trajectory'].shape[1]
    maps = [pulses['first_bounce_maps'][(0, p)] for p in range(P)]
    depth = np.stack([m['depth'].cpu().numpy() for m in maps])                  # (P,H,W)
    returns = sum(r.numel() for r in pulses['ranges'][0])
    first_hits = int(sum((m['depth'] >= 0).sum() for m in maps))
    print('%d bounces: %d returns from %d first hits over %d pulses, %.0f%% from later bounces'
          % (num_bounce, returns, first_hits, P, 100 * (1 - first_hits / returns)))
    return dict(
        rgb    = load_rgb(obj_id, pose_num),
        sar    = sar.detach().cpu().numpy(),                                    # (H,W)
        depth  = np.where(depth < 0, np.nan, depth),                            # misses are -1
        energy = np.stack([m['energy'].cpu().numpy() for m in maps]) / n_ray,   # (P,H,W), as E_k
        scatter_range  = [r.cpu().numpy() / 2 for r in pulses['ranges'][0]],    # half round trip
        scatter_energy = [e.abs().cpu().numpy() for e in pulses['energies'][0]],
        signal   = pulses['signals'][0].abs().cpu().numpy(),                    # (P,Z)
        sample_z = pulses['sample_z'][0].cpu().numpy(),                         # (P,Z)
        sensor_distance = pulses['trajectory'][0].norm(dim=-1).cpu().numpy(),   # (P,)
        half_window = b['region_radius'] / 2,
        grid   = (b['grid_width'], b['grid_height']),
        plane  = (b['image_plane_width'], b['image_plane_height']),
        display = display(b),
    )


def limits(data, pulses):
    '''Color and axis limits covering the given pulses.'''
    energy = data['energy'][pulses]
    scatter_max = max(data['scatter_energy'][p][_in_window(data, p, data['scatter_range'][p])].max()
                      for p in pulses)
    signal_max = max(data['signal'][p][_in_window(data, p, data['sample_z'][p])].max()
                     for p in pulses)
    return dict(
        depth  = (np.nanmin(data['depth'][pulses]), np.nanmax(data['depth'][pulses])),
        energy = np.percentile(energy, 99.9),
        energy_power = _power(energy.max()),
        scatter_power = _power(scatter_max),
        scatter_max = scatter_max,
        signal_power = _power(signal_max),
        signal_max = signal_max,
    )


def _window(data, p):
    '''Range shown in (e) and (f): the inner half of the sensor distance +/- region_radius window.'''
    d = data['sensor_distance'][p]
    return d - data['half_window'], d + data['half_window']


def _in_window(data, p, z):
    lo, hi = _window(data, p)
    return (z >= lo) & (z <= hi)


def _power(peak):
    return int(np.floor(np.log10(peak)))


def _scale_label(label, power):
    return r'%s ($\times 10^{%d}$)' % (label, power)


# ---------------------------------------------------------------------------- panels

def draw_rgb(fig, ax, data, p, lim):
    ax.imshow(data['rgb'])
    ax.axis('off')


def draw_sar(fig, ax, data, p, lim):
    # the sensor is below the image, so up is range and across is cross-range, both about the
    # scene center
    panel, vmin, vmax, label, fmt = panel_display(data['sar'], **data['display'])
    w, h = data['plane']
    im = ax.imshow(panel, cmap='gray', vmin=vmin, vmax=vmax, extent=(-w / 2, w / 2, -h / 2, h / 2))
    _ticks(ax)
    ax.set_xlabel(r'cross-range ($\ell$)')
    ax.set_ylabel(r'range ($\ell$)')
    fig.colorbar(im, ax=ax, label=label, format=fmt)


def draw_depth(fig, ax, data, p, lim):
    im = ax.imshow(data['depth'][p], cmap='gray', extent=_grid_extent(data),
                   vmin=lim['depth'][0], vmax=lim['depth'][1])
    _grid_axes(ax)
    fig.colorbar(im, ax=ax, label=r'first-hit range ($\ell$)')


def draw_energy(fig, ax, data, p, lim):
    scale = 10.0 ** lim['energy_power']
    im = ax.imshow(data['energy'][p] / scale, cmap='gray', extent=_grid_extent(data),
                   vmin=0, vmax=lim['energy'] / scale)
    _grid_axes(ax)
    fig.colorbar(im, ax=ax, extend='max',
                 label=_scale_label(r'first-bounce $E_k$', lim['energy_power']))


def draw_scatter(fig, ax, data, p, lim):
    scale = 10.0 ** lim['scatter_power']
    ax.scatter(data['scatter_range'][p], data['scatter_energy'][p] / scale, s=1, linewidths=0,
               rasterized=True)
    ax.set_xlim(*_window(data, p))
    ax.set_ylim(-0.05 * lim['scatter_max'] / scale, 1.05 * lim['scatter_max'] / scale)
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(_scale_label(r'$|E_k|$', lim['scatter_power']))


def draw_signal(fig, ax, data, p, lim):
    scale = 10.0 ** lim['signal_power']
    ax.plot(data['sample_z'][p], data['signal'][p] / scale)
    ax.set_xlim(*_window(data, p))
    ax.set_ylim(-0.05 * lim['signal_max'] / scale, 1.05 * lim['signal_max'] / scale)
    ax.set_xlabel(r'range ($\ell$)')
    ax.set_ylabel(_scale_label(r'$|s(z)|$', lim['signal_power']))


def _grid_extent(data):
    # the ray grid's own coordinates, with +y up, so the maps read like the rgb image
    w, h = data['grid']
    return (-w / 2, w / 2, -h / 2, h / 2)


def _grid_axes(ax):
    _ticks(ax)
    ax.set_xlabel(r'$x$ ($\ell$)')
    ax.set_ylabel(r'$y$ ($\ell$)')


def _ticks(ax):
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])


# (letter, paper png, drawer) in the paper's reading order. (a) is the paper's own rgb.png, which
# carries the pose's az/el, so only (b)-(f) are written for it
PANELS = (('a', None,                         draw_rgb),
          ('b', 'sar_overview.png',           draw_sar),
          ('c', 'sar_overview_depth.png',     draw_depth),
          ('d', 'sar_overview_energy.png',    draw_energy),
          ('e', 'sar_overview_scatters.png',  draw_scatter),
          ('f', 'sar_overview_signal.png',    draw_signal))


def save_paper_panels(data, pulse, output_dir):
    lim = limits(data, [pulse])
    with plt.rc_context({'font.size': FONT_SIZE}):
        for _, name, draw in PANELS:
            if name is None:
                continue
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            draw(fig, ax, data, pulse, lim)
            fig.savefig(os.path.join(output_dir, name), dpi=200, bbox_inches='tight')
            plt.close(fig)
    print(f'Saved paper panels of pulse {pulse} to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n\n')[0])
    parser.add_argument('-obj_id', default=DEFAULT_OBJ_ID, help='srncars object id, any split')
    parser.add_argument('-pose_num', default=DEFAULT_POSE_NUM, help='pose/rgb file stem')
    parser.add_argument('-pulse', type=int, default=None,
                        help='pulse shown in (c)-(f); the middle one if omitted')
    args = parser.parse_args()

    data = render(args.obj_id, args.pose_num)
    P = data['signal'].shape[0]
    save_paper_panels(data, P // 2 if args.pulse is None else args.pulse,
                      os.path.dirname(figures_path('sar_overview', 'x')))


if __name__ == '__main__':
    main()
