'''
Command line renderer:

    sarsonar car.obj -m fls -o fls.png --azimuth 30 --elevation 40
    sarsonar car.obj -m sar -o sar.png --set num_pulses=128 --set waveform="'lfm'"
'''
import argparse
import ast
import logging

import numpy as np

from .display import COMPRESSIONS, save_png
from .fls import render_fls
from .mesh import default_device, load_mesh
from .sar import render_sar
from .sss import render_side_scan

RENDERERS = {'sar': render_sar, 'sss': render_side_scan, 'fls': render_fls}
# each renderer's default distance; a SAR image is a planar wavefront, so its distance only sets the range window
DEFAULT_DISTANCE = {'sar': 1.3, 'sss': 7.5, 'fls': 7.5}
DEFAULT_COMPRESSION = {'sar': 'linear', 'sss': 'db', 'fls': 'db'}


def _parse_set(items):
    '''key=value pairs, values read as Python literals, falling back to strings.'''
    kwargs = {}
    for item in items:
        key, _, value = item.partition('=')
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def main(argv=None):
    p = argparse.ArgumentParser(prog='sarsonar', description='Render a SAR or sonar image of an .obj file.')
    p.add_argument('obj', help='path to the .obj file')
    p.add_argument('-m', '--modality', choices=sorted(RENDERERS), default='sar',
                   help='sar (spotlight SAR), sss (side scan sonar) or fls (forward looking sonar)')
    p.add_argument('-o', '--out', default=None, help='output .png, default <modality>.png')
    p.add_argument('--npy', default=None, help='also save the raw amplitude image to this .npy')
    p.add_argument('--azimuth', type=float, default=0.0, help='sensor azimuth in degrees')
    p.add_argument('--elevation', type=float, default=40.0, help='sensor elevation in degrees')
    p.add_argument('--distance', type=float, default=None, help='sensor distance to the origin')
    p.add_argument('--up', choices=('y', 'z'), default='y', help='up axis of the .obj file')
    p.add_argument('--no-ground', action='store_true', help='render without the ground plane')
    p.add_argument('--no-normalize', action='store_true',
                   help='keep the object\'s own size and position instead of centering it at unit diagonal')
    p.add_argument('--compression', choices=COMPRESSIONS, default=None, help='display compression of the png')
    p.add_argument('--db-floor', type=float, default=-60.0, help='black point of the dB display')
    p.add_argument('--device', default=None, help='torch device, default cuda when available')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='any other renderer argument, e.g. --set num_pings=64; repeatable')
    p.add_argument('-v', '--verbose', action='store_true', help='log timing')
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(message)s')
    device = args.device or default_device()
    mesh = load_mesh(args.obj, device=device,
                     rotate_xyz=(90.0, 0.0, 0.0) if args.up == 'y' else (0.0, 0.0, 0.0),
                     normalize=not args.no_normalize, ground=not args.no_ground)

    m = args.modality
    distance = DEFAULT_DISTANCE[m] if args.distance is None else args.distance
    image = RENDERERS[m](mesh, azimuth_deg=args.azimuth, elevation_deg=args.elevation,
                         distance=distance, **_parse_set(args.set))
    image = image.detach().abs().cpu().numpy()

    out = args.out or '%s.png' % m
    save_png(image, out, compression=args.compression or DEFAULT_COMPRESSION[m], db_floor=args.db_floor)
    print('wrote %s' % out)
    if args.npy:
        np.save(args.npy, image)
        print('wrote %s' % args.npy)


if __name__ == '__main__':
    main()
