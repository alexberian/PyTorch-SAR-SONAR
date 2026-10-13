'''
Spotlight synthetic aperture radar.

The sensor flies a straight (or circular) aperture centered on a pose that looks at the origin. Each
pulse is a planar wavefront: a parallel grid of rays traced through the scene, whose scatters are
spread onto a range signal. The pulses are then back projected onto the ground plane.

Entry points:
    render_sar(mesh, azimuth_deg, elevation_deg, ...)  -- one view -> (H,W) image
    sar_image(mesh, poses, ...)                         -- a batch of (4,4) poses -> (T,H,W) images
'''
import logging

import numpy as np
import torch

from .geometry import extract_pose_info, pose_matrix, spherical_to_cartesian
from .imaging import projected_CBP, strip_map_imaging
from .mesh import as_mesh
from .scatter import accumulate_scatters
from .signals import apply_snr, interpolate_signal

log = logging.getLogger(__name__)


def generate_trajectory(pose, trajectory_type='linear', n_pulses=100, azimuth_spread_deg=None,
                        trajectory_noise_var=0):
    '''
    Sensor positions of an aperture centered on each pose.

    inputs:
        pose (T,4,4): the aperture center of each view
        trajectory_type (str): 'linear' flies along the pose's right vector, far enough that the
            end pulses sit azimuth_spread_deg/2 either side of the center as seen from the origin;
            'circular' holds distance and elevation and sweeps azimuth
        n_pulses (int): pulses per aperture
        azimuth_spread_deg (float): angular extent of the aperture, < 180 for 'linear'
        trajectory_noise_var (float): variance of Gaussian position noise added to the flown, but
            not the assumed, trajectory
    outputs:
        true_trajectory (T,P,3), perceived_trajectory (T,P,3), cam_azimuth_deg (T,)
    '''
    assert azimuth_spread_deg >= 0, 'the azimuth cannot be negative'

    device = pose.device
    T = pose.shape[0]
    P = n_pulses

    cam_center, cam_right, cam_up, cam_forward, cam_distance, cam_elevation_deg, cam_azimuth_deg = extract_pose_info(pose)

    if trajectory_type == 'circular':
        azimuth_offsets = torch.linspace(-azimuth_spread_deg / 2, azimuth_spread_deg / 2, P, device=device)  # (P,)
        azimuth_deg = cam_azimuth_deg.reshape(T, 1) + azimuth_offsets.reshape(1, P)  # (T,P)
        elevation_deg = cam_elevation_deg.reshape(T, 1).repeat(1, P)  # (T,P)
        distance = cam_distance.reshape(T, 1).repeat(1, P)  # (T,P)
        trajectory = spherical_to_cartesian(azimuth_deg, elevation_deg, distance)  # (T,P,3)

    elif trajectory_type == 'linear':
        assert azimuth_spread_deg < 180, 'with a linear trajectory, the azimuth cannot be greater than 180 degrees'
        mag_ground_pos = cam_center[:, :2].norm(dim=-1)  # (T,)
        dist_from_center = mag_ground_pos * np.tan(azimuth_spread_deg / 2 * np.pi / 180)  # (T,)
        start_pos = cam_center - dist_from_center.reshape(T, 1) * cam_right  # (T,3)
        trajectory = torch.linspace(0, 1, P, device=device).reshape(1, P, 1) * \
                     cam_right.reshape(T, 1, 3) * \
                     (2 * dist_from_center).reshape(T, 1, 1) + \
                     start_pos.reshape(T, 1, 3)  # (T,P,3)

    else:
        raise ValueError('trajectory_type should be either circular or linear, but got %s' % trajectory_type)

    if trajectory_noise_var > 0:
        true_trajectory = trajectory + torch.randn_like(trajectory) * np.sqrt(trajectory_noise_var)
    else:
        true_trajectory = trajectory
    return true_trajectory, trajectory, cam_azimuth_deg


def sar_image(mesh, poses,
              num_pulses=64,
              azimuth_spread_deg=135.0,
              trajectory_type='linear',
              trajectory_noise_var=0,

              # ray grid
              grid_width=1.2,
              grid_height=1.2,
              n_ray_width=208,
              n_ray_height=208,
              num_bounce=2,

              # signal
              spatial_bw=30,
              spatial_fs=60,
              waveform='gaussian',
              wavelength=0.5,
              region_radius=1.4,
              snr_db=None,
              rng=None,
              use_sig_magnitude=True,

              # image
              image_width=128,
              image_height=128,
              image_plane_width=1.2,
              image_plane_height=1.2,
              imaging_algorithm='cbp',
              signal_interpolation='bilinear',
              cbp_batch_size=4096,

              ray_batch_size=2**9,
              octree=None,
              return_pulses=False,
              ):
    '''
    SAR image(s) of a mesh, one per aperture center pose.

    inputs:
        mesh (Mesh): the scene
        poses (T,4,4): aperture center of each image, looking at the origin (see geometry.pose_matrix)
        num_pulses (int): pulses per aperture
        azimuth_spread_deg (float): angular extent of the aperture
        trajectory_type (str): 'linear' or 'circular', see generate_trajectory
        grid_width/grid_height (float): extent of each pulse's parallel ray grid
        n_ray_width/n_ray_height (int): rays across that grid
        num_bounce (int): reflections traced per ray
        spatial_bw (float): waveform bandwidth in cycles per unit range; resolution ~ 1/spatial_bw
        spatial_fs (float): range samples per unit range
        waveform (str): 'sinc', 'hamming', 'gaussian', 'lfm' or 'barker13'
        wavelength (float): carrier wavelength; None makes every scatter real valued
        region_radius (float): half extent of the range window around the scene origin; keep it
            >= the image plane's half diagonal
        snr_db (float): add complex Gaussian noise at this SNR; None is noiseless
        rng (np.random.Generator): noise source for snr_db
        use_sig_magnitude (bool): back project the signal envelope rather than the complex signal
        image_width/image_height (int): pixels
        image_plane_width/image_plane_height (float): ground plane extent of the image
        imaging_algorithm (str or sequence): 'cbp' or 'stripmap'; a sequence images one ray trace
            with each and returns a dict
        signal_interpolation (str): 'bilinear' or 'sinc'
        cbp_batch_size (int): pixels per pass under sinc interpolation
        ray_batch_size (int): rays per ray tracing batch; halves itself on CUDA OOM
        octree (Octree): built once per mesh and passed back in when rendering many poses
        return_pulses (bool): also return the per-pulse trajectory, scatters, signals and
            first-bounce maps
    outputs:
        images (T,H,W) raw amplitude, or {algorithm: (T,H,W)}
        pulses (dict): only with return_pulses. trajectory (T,P,3); ranges and energies [T][P] of
            (R',) round-trip scatters; signals and sample_z (T,P,Z); first_bounce_maps
            (t,p) -> {'depth','energy'} (n_ray_height,n_ray_width), depth -1 where a ray missed
    '''
    true_trajectory, perceived_trajectory, cam_azimuth_deg = generate_trajectory(
        poses, trajectory_type=trajectory_type, n_pulses=num_pulses,
        azimuth_spread_deg=azimuth_spread_deg, trajectory_noise_var=trajectory_noise_var)

    all_ranges, all_energies, first_bounce_maps = accumulate_scatters(
        mesh, true_trajectory,
        wavelength=wavelength,
        grid_width=grid_width,
        grid_height=grid_height,
        n_ray_width=n_ray_width,
        n_ray_height=n_ray_height,
        num_bounce=num_bounce,
        second_bounce_batch_size=ray_batch_size,
        debug_gif=return_pulses,
        octree=octree,
    )  # [T][P] of (R',)

    T, P = true_trajectory.shape[:2]
    signals, sample_z = [], []
    for t in range(T):
        signals_t, sample_z_t = [], []
        for p in range(P):
            sig_tp, sz_tp = interpolate_signal(
                all_ranges[t][p].unsqueeze(0) / 2,  # (1,R') round trip -> one-way range
                all_energies[t][p].unsqueeze(0),     # (1,R')
                region_radius,
                torch.linalg.norm(true_trajectory[t, p], dim=-1).reshape(1),
                spatial_bw=spatial_bw, spatial_fs=spatial_fs,
                waveform=waveform,
            )
            signals_t.append(sig_tp.squeeze(0))  # (Z,)
            sample_z_t.append(sz_tp.squeeze(0))  # (Z,)
        signals.append(torch.stack(signals_t))
        sample_z.append(torch.stack(sample_z_t))
    signals = torch.stack(signals)    # (T,P,Z)
    sample_z = torch.stack(sample_z)  # (T,P,Z)

    if snr_db is not None:
        T, P, Z = signals.shape
        signals = apply_snr(signals.reshape(T, P * Z), snr_db, rng=rng).reshape(T, P, Z)

    complex_signals = signals
    if use_sig_magnitude:
        signals = signals.abs()

    # every algorithm images the one signal above, so asking for several costs no extra ray trace
    one_algorithm = isinstance(imaging_algorithm, str)
    algorithms = (imaging_algorithm,) if one_algorithm else tuple(imaging_algorithm)
    images = {}
    for algorithm in algorithms:
        if algorithm == 'cbp':
            images[algorithm] = projected_CBP(
                signals, sample_z, perceived_trajectory, spatial_fs,
                image_plane_rotation_deg=cam_azimuth_deg + 90,
                image_width=image_width,
                image_height=image_height,
                image_plane_width=image_plane_width,
                image_plane_height=image_plane_height,
                batch_size=cbp_batch_size,
                signal_interpolation=signal_interpolation,
                coherent_integration=not use_sig_magnitude,
                wavelength=wavelength,
            )
        elif algorithm == 'stripmap':
            images[algorithm] = strip_map_imaging(
                complex_signals, wavelength, perceived_trajectory, sample_z, spatial_fs,
                planar_wave=True,
                image_plane_rotation_deg=cam_azimuth_deg + 90,
                image_width=image_width,
                image_height=image_height,
                image_plane_width=image_plane_width,
                image_plane_height=image_plane_height,
                batch_size=cbp_batch_size,
                signal_interpolation=signal_interpolation,
            )
        else:
            raise ValueError("Invalid imaging algorithm '%s', expected 'cbp' or 'stripmap'" % algorithm)

    image = images[imaging_algorithm] if one_algorithm else images
    if return_pulses:
        return image, dict(trajectory=true_trajectory, ranges=all_ranges, energies=all_energies,
                           signals=signals, sample_z=sample_z, first_bounce_maps=first_bounce_maps)
    return image


def render_sar(mesh, azimuth_deg=0.0, elevation_deg=30.0, distance=1.3, device=None, **kwargs):
    '''
    One SAR image of a mesh, with the aperture centered on (azimuth, elevation, distance).

    inputs:
        mesh (Mesh or str): a loaded Mesh, or an .obj path loaded with load_mesh's defaults
        azimuth_deg, elevation_deg, distance (float): the aperture center, looking at the origin
        device: where to render; defaults to the Mesh's device, or cuda when available for a path
        kwargs: forwarded to sar_image
    outputs:
        image (H,W): raw amplitude; the image plane is rotated with the view, so the top of the
            image points away from the sensor
    '''
    mesh = as_mesh(mesh, device)
    pose = pose_matrix(azimuth_deg, elevation_deg, distance, device=mesh.device).reshape(1, 4, 4)
    images = sar_image(mesh, pose, **kwargs)
    return {k: v[0] for k, v in images.items()} if isinstance(images, dict) else images[0]
