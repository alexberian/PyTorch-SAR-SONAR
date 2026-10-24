'''
What every paper experiment shares: config.json, the srncars dataset, and one render per modality
from a config.json baseline.

    CONFIG                    the whole of config.json
    SAR_BASELINE, SSS_BASELINE, FLS_BASELINE
                              its three baselines, the defaults of Appendix "Default Parameters"
    SRN_CARS_DIR              srncars root, holding cars_train / cars_val / cars_test
    SHAPENET_CARS_DIR         ShapeNet cars synset, <obj_id>/models/model_normalized.obj per object
    CVDOMES_DIR               the CVDomes .obj models, for the materials experiment
    FIGURES_DIR               where every experiment writes, experiments/figures unless
                              SARSONAR_FIGURES_DIR is set

    render_sar / render_side_scan / render_fls
                              one image of an srncars object at one of its poses, from a baseline
                              dict with any of its keys overridden
    run_sweep                 render a list of panels and stitch them into one figure
'''
import json
import os
import re

import numpy as np
import torch

import sarsonar
from sarsonar.geometry import extract_pose_info

from paper_figure_layout import panel_display, stitch_panels

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EXPERIMENTS_DIR)
SPHERE_OBJ = os.path.join(REPO_DIR, 'tests', 'data', 'sphere.obj')


def _tuplify(value):
    # JSON has no tuples; raids and rotations are tuples to their callers
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuplify(v) for k, v in value.items()}
    return value


def load_config(path=os.path.join(EXPERIMENTS_DIR, 'config.json')):
    '''config.json, with whole-line // comments stripped.'''
    with open(path) as f:
        text = re.sub(r'^\s*//.*$', '', f.read(), flags=re.MULTILINE)
    return _tuplify(json.loads(text))


def _data_dir(key, env):
    path = os.environ.get(env, CONFIG[key])
    return path if os.path.isabs(path) else os.path.join(EXPERIMENTS_DIR, path)


CONFIG = load_config()
SAR_BASELINE = dict(CONFIG['sar_baseline'])
SSS_BASELINE = dict(CONFIG['side_scan_sonar_baseline'])
FLS_BASELINE = dict(CONFIG['forward_looking_sonar_baseline'])

SRN_CARS_DIR = _data_dir('srn_cars_dir', 'SRN_CARS_DIR')
SHAPENET_CARS_DIR = _data_dir('shapenet_cars_dir', 'SHAPENET_CARS_DIR')
CVDOMES_DIR = _data_dir('cvdomes_dir', 'CVDOMES_DIR')
FIGURES_DIR = os.environ.get('SARSONAR_FIGURES_DIR', os.path.join(EXPERIMENTS_DIR, 'figures'))

DISPLAY_KEYS = ('compression', 'db_floor', 'asinh_k_ratio')


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def figures_path(*parts):
    '''A path under FIGURES_DIR, creating its directory.'''
    path = os.path.join(FIGURES_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def display(baseline):
    '''A baseline's display compression, as panel_display takes it.'''
    return {k: baseline[k] for k in DISPLAY_KEYS}


# ---------------------------------------------------------------------------- srncars

def find_split_dir(obj_id):
    '''The srncars split directory holding obj_id.'''
    for split in ('cars_train', 'cars_val', 'cars_test'):
        path = os.path.join(SRN_CARS_DIR, split)
        if os.path.isdir(os.path.join(path, obj_id)):
            return path
    raise FileNotFoundError('object %s is in none of the srncars splits under %s; set srn_cars_dir '
                            'in config.json or SRN_CARS_DIR' % (obj_id, SRN_CARS_DIR))


def object_dir(obj_id):
    return os.path.join(find_split_dir(obj_id), obj_id)


def mesh_path(obj_id):
    return os.path.join(SHAPENET_CARS_DIR, obj_id, 'models', 'model_normalized.obj')


def pose_nums(obj_id):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(object_dir(obj_id), 'pose'))
                  if f.endswith('.txt'))


def load_pose(obj_id, pose_num, device='cpu'):
    '''(1,4,4) float32 camera-to-world pose of one srncars view.'''
    pose = np.loadtxt(os.path.join(object_dir(obj_id), 'pose', '%s.txt' % pose_num))
    return torch.tensor(pose.reshape(1, 4, 4).astype(np.float32), device=device)


def load_rgb(obj_id, pose_num):
    from PIL import Image
    return np.array(Image.open(os.path.join(object_dir(obj_id), 'rgb', '%s.png' % pose_num)))[..., :3]


def pose_az_el(obj_id, pose_num):
    '''Azimuth and elevation in deg of one srncars view.'''
    info = extract_pose_info(load_pose(obj_id, pose_num))
    return info[6].item(), info[5].item()


def nearest_pose_num(obj_id, azimuth_deg, elevation_deg):
    '''The view of obj_id whose look direction is closest to (azimuth_deg, elevation_deg).'''
    def look(az, el):
        az, el = np.deg2rad(az), np.deg2rad(el)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    target = look(azimuth_deg, elevation_deg)
    return max(pose_nums(obj_id), key=lambda p: float(np.dot(look(*pose_az_el(obj_id, p)), target)))


# ---------------------------------------------------------------------------- baseline renders

def baseline_mesh(b, path=None, device=None):
    '''
    The scene a baseline images: its srncars car (or path) in the file's own units, on the ground.

    The paper's meshes are ShapeNet normalized models, already unit diagonal, so normalize is off.
    '''
    return sarsonar.load_mesh(
        path or b.get('override_obj_path') or mesh_path(b['obj_id']),
        rotate_xyz=b['object_rotate_xyz'],
        normalize=False,
        scale=b.get('mesh_scale'),
        x_flip=b['object_x_flip'],
        level_with_ground=b.get('level_with_ground', True),
        ground=b.get('make_ground', True),
        obj_raids=b['obj_raids'],
        ground_raids=b['ground_raids'],
        material_file=b.get('material_file'),
        device=device or default_device(),
    )


def sar_kwargs(b):
    '''sar_image keyword arguments of a SAR baseline.'''
    return dict(
        num_pulses=b['num_pulse'],
        azimuth_spread_deg=b['azimuth_spread'],
        trajectory_type=b['trajectory_type'],
        trajectory_noise_var=b['trajectory_noise_var'],
        grid_width=b['grid_width'], grid_height=b['grid_height'],
        n_ray_width=b['n_ray_width'], n_ray_height=b['n_ray_height'],
        num_bounce=b['num_bounce'],
        spatial_bw=b['spatial_bw'], spatial_fs=b['spatial_fs'],
        waveform=b['waveform'],
        wavelength=b['wavelength'],
        region_radius=b['region_radius'],
        snr_db=b['snr_db'],
        use_sig_magnitude=b['use_sig_magnitude'],
        image_width=b['image_width'], image_height=b['image_height'],
        image_plane_width=b['image_plane_width'], image_plane_height=b['image_plane_height'],
        imaging_algorithm=b['imaging_algorithm'],
        signal_interpolation=b['signal_interpolation'],
        cbp_batch_size=b['cbp_batch_size'],
    )


def render_sar(b, seed=None, mesh=None, octree=None, return_pulses=False, device=None):
    '''
    SAR image of a baseline's srncars view, the aperture centered on the pose file's camera.

    inputs:
        b (dict): a SAR baseline, any key overridden
        seed (int): seeds numpy and torch first, for the receiver and trajectory noise
        mesh (Mesh): the scene, baseline_mesh(b) when None
    outputs:
        image (H,W) raw amplitude, plus sar_image's pulses dict with return_pulses
    '''
    device = device or default_device()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    mesh = mesh if mesh is not None else baseline_mesh(b, device=device)
    pose = load_pose(b['obj_id'], b['pose_num'], device)
    out = sarsonar.sar_image(mesh, pose, octree=octree, return_pulses=return_pulses, **sar_kwargs(b))
    if return_pulses:
        return out[0][0], out[1]
    return out[0]


def sonar_position(pose, elevation_deg=None, distance=None):
    '''
    (3,) sonar position along a pose's camera direction.

    inputs:
        pose (1,4,4): an srncars view; its camera center sets the azimuth, elevation and distance
        elevation_deg (float): replaces the pose's elevation, keeping its azimuth and distance
        distance (float): replaces its distance, keeping its look direction
    '''
    position = extract_pose_info(pose)[0].reshape(3)
    if elevation_deg is not None:
        # in float64, so cos(90 deg) keeps the azimuth's sign
        x, y, z = position.tolist()
        az, el = np.arctan2(y, x), np.radians(float(elevation_deg))
        position = torch.tensor(np.sqrt(x * x + y * y + z * z) * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]),
            dtype=torch.float32, device=pose.device)
    if distance is not None:
        position = torch.nn.functional.normalize(position, dim=-1) * distance
    return position


SSS_KEYS = ('track_length', 'num_pings', 'elevation_fov_deg', 'azimuth_beam_width_deg',
            'num_ray_width', 'num_ray_height', 'image_width', 'image_height',
            'image_cross_range_swath', 'image_range_swath', 'wavelength', 'num_bounce',
            'water_absorption', 'tvg_exponent', 'spatial_bw', 'spatial_fs',
            'waveform', 'use_sig_magnitude')

FLS_KEYS = ('image_azimuth_range', 'num_pings', 'num_range_values', 'ray_fov_az', 'ray_fov_el',
            'azimuth_beam_width_deg', 'num_ray_width', 'num_ray_height', 'image_range_swath',
            'wavelength', 'num_bounce', 'water_absorption', 'tvg_exponent', 'spatial_bw', 'spatial_fs',
            'waveform', 'use_sig_magnitude')


def _sonar_view(b, device):
    pose = load_pose(b['obj_id'], b['pose_num'], device)
    elevation = b.get('elevation_angle_deg')
    if elevation is None:
        elevation_check = extract_pose_info(pose)[5].item()
    else:
        elevation_check = float(elevation)
    assert 0.0 <= elevation_check <= 90.0, \
        'elevation %.1f deg puts the sensor below the seafloor or past overhead' % elevation_check
    return sonar_position(pose, elevation, b['sensor_distance'])


def render_side_scan(b, mesh=None, octree=None, device=None):
    '''
    Side scan image of a baseline's srncars view: a track through the pose's camera direction at
    the baseline's sensor_distance (and elevation_angle_deg, when set).

    outputs:
        image (H,W) raw amplitude, row_coords (H,) slant range far to near, col_coords (W,)
    '''
    device = device or default_device()
    mesh = mesh if mesh is not None else baseline_mesh(b, device=device)
    images, rows, cols = sarsonar.side_scan_image(mesh, _sonar_view(b, device), octree=octree,
                                                  **{k: b[k] for k in SSS_KEYS})
    return images[0], rows[0], cols


def render_fls(b, mesh=None, octree=None, device=None):
    '''
    Forward looking sonar image of a baseline's srncars view, from the pose's camera direction at
    the baseline's sensor_distance (and elevation_angle_deg, when set).

    outputs:
        image (H,W) raw amplitude, row_ranges (H,) far to near, ping_azimuths (W,) in deg
    '''
    device = device or default_device()
    mesh = mesh if mesh is not None else baseline_mesh(b, device=device)
    images, rows, azimuths = sarsonar.fls_image(mesh, _sonar_view(b, device).reshape(1, 3),
                                                octree=octree, **{k: b[k] for k in FLS_KEYS})
    return images[0], rows, azimuths


# ---------------------------------------------------------------------------- sweeps

# figure file prefix of each modality, as the paper's figure names have it
PREFIX = {'sar': 'sar', 'sss': 'side_scan', 'fls': 'fls'}
BASELINE = {'sar': SAR_BASELINE, 'sss': SSS_BASELINE, 'fls': FLS_BASELINE}
SAR_SEED = 8134  # every SAR panel starts from this seed, so its noise is the same draw


def render_panel(modality, b):
    '''One sweep panel: (raw amplitude (H,W) numpy, its axes for the stitched figure or None).'''
    if modality == 'sar':
        return render_sar(b, seed=SAR_SEED).detach().cpu().numpy(), None
    if modality == 'sss':
        return render_side_scan(b)[0].detach().cpu().numpy(), None
    image, rows, azimuths = render_fls(b)
    azimuths, rows = azimuths.cpu().numpy(), rows.cpu().numpy()
    # near range at the bottom, matching the row order of the image
    return image.detach().cpu().numpy(), [azimuths[0], azimuths[-1], rows[-1], rows[0]]


def run_sweep(modality, name, vary, titles, overrides=None, ncols=None, baseline=None, replot=False):
    '''
    Render one parameter sweep and stitch it into <prefix>_stitched_<name>.png.

    Every panel is the baseline with overrides, then its own entry of each vary list. Each panel
    is displayed in its own compression (a sweep may vary the compression itself), referenced to
    its own peak, with its own colorbar.

    inputs:
        modality (str): 'sar', 'sss' or 'fls'
        name (str): the sweep's name in the file names
        vary (dict): baseline key -> one value per panel
        titles (list of str): one per panel
        overrides (dict): baseline keys every panel shares
        ncols (int): panels per row, all in one row when None
        baseline (dict): defaults to the modality's config.json baseline
        replot (bool): stitch the raw amplitudes saved by an earlier run instead of rendering
    outputs:
        path (str): the stitched figure
    '''
    n = len(titles)
    assert all(len(v) == n for v in vary.values()), 'every vary list needs one entry per title'
    base = {**(baseline or BASELINE[modality]), **(overrides or {})}
    prefix = PREFIX[modality]

    panels, vmins, vmaxs, labels, fmts, extents = [], [], [], [], [], []
    for i in range(n):
        b = dict(base, **{k: v[i] for k, v in vary.items()})
        raw_path = figures_path('%s_%s' % (prefix, name), '%03d.npz' % i)
        if replot:
            with np.load(raw_path, allow_pickle=True) as f:
                amplitude, extent = f['image'], f['extent'].tolist()
        else:
            print('=== %s %s [%d/%d] %s ===' % (prefix, name, i + 1, n, titles[i]))
            amplitude, extent = render_panel(modality, b)
            np.savez(raw_path, image=amplitude, extent=np.array(extent, dtype=object))
        panel, vmin, vmax, label, fmt = panel_display(np.abs(amplitude), **display(b))
        panels.append(panel)
        vmins.append(vmin)
        vmaxs.append(vmax)
        labels.append(label)
        fmts.append(fmt)
        extents.append(extent)

    axes = dict(extents=extents, xlabel='Azimuth (deg)', ylabel='Range', show_axes=True) \
        if modality == 'fls' else {}
    return stitch_panels(panels, titles, figures_path('%s_stitched_%s.png' % (prefix, name)),
                         cmap='gray', vmin=vmins, vmax=vmaxs, cbar_label=labels,
                         cbar_tick_fmt=fmts, ncols=ncols, **axes)


def sweep_main(modality, sweeps, description):
    '''Command line of a sweep script: every sweep by default, or the ones named.'''
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('names', nargs='*',
                        help='sweeps to run, of: %s (default: all)' % ', '.join(sweeps))
    parser.add_argument('-replot', action='store_true',
                        help='restitch from the saved raw amplitudes, without rendering')
    args = parser.parse_args()
    unknown = set(args.names) - set(sweeps)
    if unknown:
        parser.error('unknown sweep(s) %s; choose from %s' % (', '.join(sorted(unknown)), ', '.join(sweeps)))
    for name in args.names or sweeps:
        run_sweep(modality, name, replot=args.replot, **sweeps[name])
