'''
Figure "transmit waveform": a SAR row and a side scan sonar row, the same waveform in each column,
written to figures/waveform_stitched_sar_sss.png.

The first four columns are four waveforms at the baseline bandwidth; the last is the chirp at twice
it, with F_s = 2 B_s throughout. Each row is its config.json baseline with only these changed, and
displayed in that baseline's own compression.

    python waveform_figure.py
    python waveform_figure.py -replot
'''
import argparse

import numpy as np

from common import SAR_BASELINE, SSS_BASELINE, display, figures_path, render_panel
from paper_figure_layout import panel_display, stitch_panels

TITLES = ['Sinc Interpolation', 'Gaussian Pulse', 'LFM Chirp', 'Barker 13', 'LFM Chirp, 2x BW']


def waveform_panels(baseline):
    '''Each column's overrides of one baseline.'''
    bw = baseline['spatial_bw']
    return [dict(waveform=w, spatial_bw=b, spatial_fs=2 * b)
            for w, b in zip(['sinc', 'gaussian', 'lfm', 'barker13', 'lfm'], [bw] * 4 + [2 * bw])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n\n')[0])
    parser.add_argument('-replot', action='store_true', help='restitch the saved panels, without rendering')
    args = parser.parse_args()

    panels = []
    for modality, baseline, row in (('sar', SAR_BASELINE, 'SAR'), ('sss', SSS_BASELINE, 'Side Scan')):
        for i, overrides in enumerate(waveform_panels(baseline)):
            raw_path = figures_path('waveform', '%s_%03d.npy' % (modality, i))
            if args.replot:
                amplitude = np.load(raw_path)
            else:
                print('=== %s: %s ===' % (row, TITLES[i]))
                amplitude, _ = render_panel(modality, {**baseline, **overrides})
                np.save(raw_path, amplitude)
            panels.append(panel_display(amplitude, **display(baseline)))

    images, vmins, vmaxs, labels, fmts = map(list, zip(*panels))
    titles = ['SAR: ' + t for t in TITLES] + ['Side Scan: ' + t for t in TITLES]
    stitch_panels(images, titles, figures_path('waveform_stitched_sar_sss.png'), cmap='gray',
                  vmin=vmins, vmax=vmaxs, cbar_label=labels, cbar_tick_fmt=fmts, ncols=len(TITLES))


if __name__ == '__main__':
    main()
