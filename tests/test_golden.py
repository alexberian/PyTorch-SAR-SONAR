'''
Golden fingerprints of the three scatter accumulators on a sphere over a ground plane, recorded from
the research code behind the paper. A count must match exactly; sums of real and imaginary parts
within RTOL.

Regenerate after an intended change with:  python tests/test_golden.py --record
'''
import sys

import torch

from sarsonar.geometry import cartesian_to_spherical, pose_matrix, spherical_to_cartesian
from sarsonar.mesh import load_mesh
from sarsonar.scatter import (accumulate_scatters, accumulate_scatters_perspective,
                              accumulate_scatters_side_scan, centered_linspace)
from conftest import SPHERE

# float reductions reorder between CPU and CUDA, so CPU only gets close; its coherent energy sums
# least, since float32 range rounding becomes phase at wavelength 0.03 and the sums cancel heavily
RTOL = {'cuda': 1e-6, 'cpu': 1e-5}
CPU_ENERGY_RTOL = 1e-2

GOLDEN = {
    'planar 2 bounce coherent': {
        'energy': (10443, -9.0569233538e-03, 1.0106582551e-02, 1.1756748427e-04),
        'maps': (12288, 1.9467985562e+04, 0.0000000000e+00, 4.1005740166e+00),
        'range': (10443, 6.8901203158e+04, 0.0000000000e+00, 4.9523815155e+01),
    },
    'planar 1 bounce incoherent': {
        'energy': (6144, 4.2399384540e-01, 0.0000000000e+00, 1.1756747699e-04),
        'maps': (0, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00),
        'range': (6144, 3.8067632387e+04, 0.0000000000e+00, 8.2011499405e+00),
    },
    'perspective 2 bounce': {
        'arrival_azimuth': (3228, -4.1455462356e+03, 0.0000000000e+00, 1.7831898499e+02),
        'azimuth': (3228, -1.5193547815e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (3228, 7.8723181831e-03, -3.4986875895e-03, 1.2161598715e-04),
        'maps': (4096, 6.6351604041e+03, 0.0000000000e+00, 5.4100818634e+00),
        'range': (3228, 2.4311271969e+04, 0.0000000000e+00, 1.9580755615e+02),
    },
    'side scan absorption 0.05': {
        'arrival_azimuth': (4367, -1.2216744125e+03, 0.0000000000e+00, 1.3628024292e+02),
        'azimuth': (4367, 1.5338709664e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (4367, -1.4079777476e-03, 2.4556071680e-04, 9.1816727945e-05),
        'maps': (6144, 1.0861379572e+04, 0.0000000000e+00, 7.0999932289e+00),
        'range': (4367, 3.4078427491e+04, 0.0000000000e+00, 1.8021507263e+02),
    },
    'side scan absorption 0.00': {
        'arrival_azimuth': (4367, -1.2216744125e+03, 0.0000000000e+00, 1.3628024292e+02),
        'azimuth': (4367, 1.5338709664e+02, 0.0000000000e+00, 1.4999999046e+01),
        'energy': (4367, -2.1379681313e-03, 1.3775981986e-03, 1.2387866445e-04),
        'maps': (6144, 1.0861379572e+04, 0.0000000000e+00, 7.0999932289e+00),
        'range': (4367, 3.4078427491e+04, 0.0000000000e+00, 1.8021507263e+02),
    },
}


def _test_scene(device):
    '''Sphere over a ground plane: gives real second bounces and real occlusion.'''
    return load_mesh(SPHERE, rotate_xyz=(0.0, 0.0, 0.0), normalize=False, ground=True,
                     obj_raids=(1.0, 1.0, 0.9, 0.1, 1.0), ground_raids=(0.1, 0.1, 0.1, 0.9, 1.0),
                     device=device)


def _flatten(value, out):
    '''Collect every tensor in one of the nested wrapper returns into one flat list.'''
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)
    elif isinstance(value, dict):
        for key in sorted(value):
            _flatten(value[key], out)
    elif value is not None:
        out.append(value.reshape(-1))
    return out


def _fingerprint(**named):
    '''
    Order-insensitive summary of a wrapper's outputs: (count, sum of real parts, sum of
    imaginary parts, largest magnitude) per named output.

    Nothing downstream cares what order the scatters come back in, so sums are the right
    invariant. Real and imaginary parts are kept apart so a phase regression cannot hide
    behind a magnitude that still matches.
    '''
    fp = {}
    for name, value in sorted(named.items()):
        parts = _flatten(value, [])
        flat = torch.cat(parts) if parts else torch.zeros(0)
        wide = flat.to(torch.complex128) if flat.is_complex() else flat.to(torch.float64) + 0j
        fp[name] = (int(flat.numel()), float(wide.real.sum()), float(wide.imag.sum()),
                    float(flat.abs().max()) if flat.numel() else 0.0)
    return fp


def _side_scan_poses(device, num_pings=3, track_length=1.5):
    '''Straight-track broadside poses, built the way side_scan_image builds them.'''
    mean_sensor_position = spherical_to_cartesian(
        torch.tensor(20.0, device=device), torch.tensor(35.0, device=device),
        torch.tensor(4.0, device=device))                                          # (3,)
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)
    track_direction = torch.nn.functional.normalize(
        torch.linalg.cross(line_of_sight, world_up), dim=-1)
    up_vector       = torch.linalg.cross(track_direction, line_of_sight)

    ping_offsets = centered_linspace(track_length, num_pings, device)               # (P,)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))

    poses = torch.zeros(num_pings, 4, 4, device=device)
    poses[:, :3, 0] = track_direction
    poses[:, :3, 1] = up_vector
    poses[:, :3, 2] = line_of_sight
    poses[:, :3, 3] = trajectory
    poses[:,  3, 3] = 1.0
    return poses.unsqueeze(0)  # (1,P,4,4), one scene


def _wrapper_cases(device):
    '''Every wrapper under a config that exercises two bounces, occlusion and the energy tail.'''
    mesh = _test_scene(device)

    az = torch.tensor([[10.0, 12.0, 14.0], [100.0, 102.0, 104.0]], device=device)  # (T,P)
    trajectory = spherical_to_cartesian(az, torch.full_like(az, 30.0), torch.full_like(az, 4.0))
    fan_az = torch.tensor([[10.0, 40.0]], device=device)
    fan_trajectory = spherical_to_cartesian(fan_az, torch.full_like(fan_az, 30.0), torch.full_like(fan_az, 4.0))
    poses = _side_scan_poses(device)

    grid = dict(grid_width=1.2, grid_height=1.2, n_ray_width=32, n_ray_height=32,
                second_bounce_batch_size=2**18)
    fan = dict(fov_width_deg=30.0, fov_height_deg=30.0, n_ray_width=32, n_ray_height=32,
               second_bounce_batch_size=2**18)

    r, e, maps = accumulate_scatters(mesh, trajectory, num_bounce=2, wavelength=0.03, debug_gif=True, **grid)
    yield 'planar 2 bounce coherent', _fingerprint(range=r, energy=e, maps=maps)

    r, e, maps = accumulate_scatters(mesh, trajectory, num_bounce=1, wavelength=None, debug_gif=False, **grid)
    yield 'planar 1 bounce incoherent', _fingerprint(range=r, energy=e, maps=maps)

    r, e, a, rx, maps = accumulate_scatters_perspective(mesh, fan_trajectory, num_bounce=2, wavelength=0.03,
                                                        debug_gif=True, **fan)
    yield 'perspective 2 bounce', _fingerprint(range=r, energy=e, azimuth=a, arrival_azimuth=rx, maps=maps)

    for absorption in (0.05, 0.0):
        r, e, a, rx, maps = accumulate_scatters_side_scan(mesh, poses, num_bounce=2, wavelength=0.03,
                                                          debug_gif=True, water_absorption=absorption, **fan)
        yield ('side scan absorption %.2f' % absorption,
               _fingerprint(range=r, energy=e, azimuth=a, arrival_azimuth=rx, maps=maps))


def test_golden_fingerprints(device):
    failures = []
    for label, fp in _wrapper_cases(device):
        golden = GOLDEN[label]
        for name in sorted(fp):
            got, want = fp[name], golden[name]
            if got[0] != want[0]:
                failures.append('%s %s: count %d vs %d' % (label, name, got[0], want[0]))
                continue
            worst = max(abs(g - w) / max(abs(w), 1e-30) for g, w in zip(got[1:], want[1:]))
            on_cpu = torch.device(device).type == 'cpu'
            if worst > (CPU_ENERGY_RTOL if on_cpu and name == 'energy' else RTOL['cpu' if on_cpu else 'cuda']):
                failures.append('%s %s: rel %.3e' % (label, name, worst))
    assert not failures, '\n'.join(failures)


def test_perspective_matches_side_scan(device):
    '''
    The two fan wrappers are the same physics with the pose supplied rather than inferred, so handing
    side scan the pose perspective builds for itself must give the same scatters, bit for bit.
    '''
    mesh = _test_scene(device)
    az = torch.tensor([[15.0, 45.0]], device=device)
    trajectory = spherical_to_cartesian(az, torch.full_like(az, 30.0), torch.full_like(az, 4.0))

    T, P = trajectory.shape[:2]
    poses = torch.zeros(T, P, 4, 4, device=device)
    for t in range(T):
        for p in range(P):
            a, e, d = cartesian_to_spherical(trajectory[t, p])
            poses[t, p] = pose_matrix(a, e, d, device=device)
            poses[t, p, :3, 3] = trajectory[t, p]

    kwargs = dict(wavelength=0.03, fov_width_deg=30.0, fov_height_deg=30.0,
                  n_ray_width=32, n_ray_height=32, num_bounce=2, second_bounce_batch_size=2**18)
    per = accumulate_scatters_perspective(mesh, trajectory, **kwargs)
    sss = accumulate_scatters_side_scan(mesh, poses, **kwargs)
    for i in range(4):  # ranges, energies, azimuths, arrival azimuths
        for got, want in zip(_flatten(sss[i], []), _flatten(per[i], [])):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def record_goldens(device):
    print('GOLDEN = {')
    for label, fp in _wrapper_cases(device):
        print("    '%s': {" % label)
        for name in sorted(fp):
            n, re_, im, absmax = fp[name]
            print("        '%s': (%d, %.10e, %.10e, %.10e)," % (name, n, re_, im, absmax))
        print('    },')
    print('}')


if __name__ == '__main__' and '--record' in sys.argv:
    record_goldens('cuda' if torch.cuda.is_available() else 'cpu')
