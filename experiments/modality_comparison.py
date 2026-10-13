'''
Figure "modality comparison": four random srncars cars, one row each, as RGB, spotlight SAR, side
scan sonar and forward looking sonar, written to figures/modality_comparison.png.

Each car is drawn at one random view inside the 20..60 deg elevation band and rendered at the three
config.json baselines from that view, each image stretched to 8 bits in its own baseline's
compression, the way the srncars sonar/SAR dataset stores them.

    python modality_comparison.py
'''
import argparse
import os
import zlib

import numpy as np
from matplotlib import pyplot as plt

from sarsonar.display import asinh_compress, db_compress
from sarsonar.raytrace import build_octree

from common import (FLS_BASELINE, SAR_BASELINE, SRN_CARS_DIR, SSS_BASELINE, baseline_mesh,
                    display, figures_path, load_rgb, mesh_path, pose_az_el, pose_nums, render_fls,
                    render_sar, render_side_scan)

ELEVATION_BAND = (20.0, 60.0)  # deg; the dataset's band


def to_uint8(amplitude, compression, db_floor, asinh_k_ratio):
    '''One image to 8 bits, referenced to itself; 'linear' is a min-max stretch.'''
    amplitude = np.nan_to_num(np.asarray(amplitude, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(amplitude.max())
    if compression == 'db' and peak > 0.0:
        return ((db_compress(amplitude, peak, db_floor) - db_floor) / -db_floor * 255.0).astype(np.uint8)
    if compression == 'asinh':
        ref = float(np.percentile(amplitude, 99.9))
        if ref <= 0.0:
            return np.zeros(amplitude.shape, dtype=np.uint8)
        return (asinh_compress(amplitude, asinh_k_ratio * ref, ref) * 255).astype(np.uint8)
    low = float(amplitude.min())
    return ((amplitude - low) / max(peak - low, 1e-12) * 255.0).clip(0, 255).astype(np.uint8)


def render_view(obj_id, pose_num):
    '''SAR, side scan and FLS 8-bit images of one view, off one mesh load.'''
    mesh = baseline_mesh(SAR_BASELINE, mesh_path(obj_id))
    octree = build_octree(mesh)
    # seeded per view, as the dataset is, so its receiver noise is reproducible
    seed = zlib.crc32(('%s/%s' % (obj_id, pose_num)).encode())
    view = dict(obj_id=obj_id, pose_num=pose_num)
    sar = render_sar({**SAR_BASELINE, **view}, seed=seed, mesh=mesh, octree=octree)
    sss = render_side_scan({**SSS_BASELINE, **view}, mesh=mesh, octree=octree)[0]
    fls = render_fls({**FLS_BASELINE, **view}, mesh=mesh, octree=octree)[0]
    return [to_uint8(image.detach().cpu().numpy(), **display(b))
            for image, b in ((sar, SAR_BASELINE), (sss, SSS_BASELINE), (fls, FLS_BASELINE))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n\n')[0])
    parser.add_argument('-num_examples', type=int, default=4)
    parser.add_argument('-split', default='cars_train')
    parser.add_argument('-seed', type=int, default=8134)
    args = parser.parse_args()

    # the three baselines must share one scene for one mesh load to serve them
    for key in ('object_x_flip', 'object_rotate_xyz', 'obj_raids', 'ground_raids'):
        assert SAR_BASELINE[key] == SSS_BASELINE[key] == FLS_BASELINE[key], key

    obj_ids = sorted(os.listdir(os.path.join(SRN_CARS_DIR, args.split)))
    rng = np.random.RandomState(args.seed)
    fig, axes = plt.subplots(args.num_examples, 4, figsize=(12, 3.2 * args.num_examples), squeeze=False)
    for row in range(args.num_examples):
        obj_id = obj_ids[rng.randint(0, len(obj_ids))]
        in_band = [p for p in pose_nums(obj_id)
                   if ELEVATION_BAND[0] <= pose_az_el(obj_id, p)[1] <= ELEVATION_BAND[1]]
        pose_num = in_band[rng.randint(0, len(in_band))]
        print('row %d: %s pose %s, elevation %.1f deg' % (row, obj_id, pose_num, pose_az_el(obj_id, pose_num)[1]))

        for col, panel in enumerate([load_rgb(obj_id, pose_num)] + render_view(obj_id, pose_num)):
            axes[row, col].imshow(panel, cmap='gray', vmin=0, vmax=255)
            axes[row, col].axis('off')

    fig.subplots_adjust(wspace=0.02, hspace=0.1)
    path = figures_path('modality_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved: %s' % path)


if __name__ == '__main__':
    main()
