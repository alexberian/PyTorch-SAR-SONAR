"""
Flat-plate harness for sar_validation.py.

Where validation_point_target.py bypasses the ray tracer to isolate the signal and imaging chain,
this drives the whole forward model -- ray tracer, scatter accumulation, interpolate_signal --
against the physical-optics closed form for a flat square plate, the canonical primitive every SAR
simulator is expected to reproduce.

Geometry. The plate sits at the scene origin with its normal along the line of sight; the sensor
sits at elevation 0 so that sweeping sensor azimuth by phi is *exactly* a rotation about an
in-plane axis of the plate. That makes the PO reference one-dimensional and exact:

    sigma(phi) = 4 pi A^2 / lambda^2 * cos^2(phi) * sinc^2( 2 L sin(phi) / lambda )

with A = L^2 and np.sinc(x) = sin(pi x)/(pi x), so the first null is at sin(phi) = lambda / (2 L).

Observable. Every experiment reports

    sigma_hat = | sum_r E_r exp(j 2 pi z_r / lambda) |^2

the squared magnitude of the coherent sum of the scatter phasors, i.e. the monostatic RCS in
arbitrary units, taken *before* image formation.

    run_area_sweep            RCS vs plate area, expects sigma ~ A^2
    run_wavelength_sweep      RCS vs wavelength, PO expects sigma ~ 1/lambda^2
    run_phase_sampling_sweep  error vs rays per phase cycle across the plate
"""
import numpy as np
import torch

from sarsonar.geometry import spherical_to_cartesian
from sarsonar.mesh import Mesh
from sarsonar.scatter import accumulate_scatters


BASELINE = dict(
    plate_side=0.40,        # scene units
    elevation_deg=0.0,      # sensor elevation; 0 makes the azimuth sweep an exact in-plane rotation
    sensor_distance=1.3,
    wavelength=0.05,        # L/lambda = 8
    grid_width=1.0,         # ray grid, held FIXED so hit count tracks plate area
    n_ray=512,
    n_subdiv=8,             # triangles per plate edge
    scatter_fraction=0.9,   # s in the r+s+a=1 budget
)

# --------------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------------

def _plate_frame(tilt_deg, elevation_deg, device):
    """
    Orthonormal frame for a plate whose normal is tilted `tilt_deg` off the line of
    sight, rotated about the plate's own vertical in-plane axis.

    Returns (n, u, v): n the plate normal, u the in-plane axis the tilt rotates
    within (so range varies linearly along u), v the in-plane axis it does not.
    """
    az = torch.tensor(0.0, device=device)
    el = torch.tensor(float(elevation_deg), device=device)
    one = torch.tensor(1.0, device=device)

    n0 = spherical_to_cartesian(az, el, one)                      # (3,) origin -> sensor
    z = torch.tensor([0.0, 0.0, 1.0], device=device)
    u0 = torch.nn.functional.normalize(torch.linalg.cross(z, n0), dim=0)
    v0 = torch.linalg.cross(n0, u0)                               # completes a right-handed frame

    phi = float(tilt_deg) * np.pi / 180.0
    n = float(np.cos(phi)) * n0 + float(np.sin(phi)) * u0
    u = float(np.cos(phi)) * u0 - float(np.sin(phi)) * n0
    return n, u, v0


def make_plate(side, tilt_deg=0.0, center=(0.0, 0.0, 0.0), scatter_fraction=0.9,
               diffuse=False, n_subdiv=8, elevation_deg=0.0, frame=None, device='cuda'):
    """
    One square plate as (verts, faces, raids).

    Materials use the (r, a, i, d, s) convention, r + s + a = 1 and i + d = 1.
    `scatter_fraction` is s, the fraction of incident *power* the surface scatters.
    r = 0 because these experiments are single-bounce.

    `frame` overrides the tilt-derived (n, u, v) basis.
    """
    n, u, v = frame if frame is not None else _plate_frame(tilt_deg, elevation_deg, device)

    t = torch.linspace(-side / 2, side / 2, n_subdiv + 1, device=device)
    su, sv = torch.meshgrid(t, t, indexing='ij')                  # (K,K)
    c = torch.tensor(center, device=device, dtype=torch.float32)
    verts = (c.reshape(1, 1, 3)
             + su.unsqueeze(-1) * u.reshape(1, 1, 3)
             + sv.unsqueeze(-1) * v.reshape(1, 1, 3))             # (K,K,3)
    K = n_subdiv + 1
    verts = verts.reshape(-1, 3)

    # two triangles per grid square, wound consistently (as in make_ground)
    i, j = torch.meshgrid(torch.arange(K - 1, device=device),
                          torch.arange(K - 1, device=device), indexing='ij')
    a_ = (i * K + j).reshape(-1)
    b_ = ((i + 1) * K + j).reshape(-1)
    c_ = (i * K + j + 1).reshape(-1)
    d_ = ((i + 1) * K + j + 1).reshape(-1)
    faces = torch.cat([torch.stack([a_, b_, c_], dim=-1),
                       torch.stack([b_, d_, c_], dim=-1)], dim=0).long()

    s = float(scatter_fraction)
    # (r, a, i, d, s): pure specular unless `diffuse`, which gives an aspect-independent
    # return
    raids = (0.0, 1.0 - s, 0.0 if diffuse else 1.0, 1.0 if diffuse else 0.0, s)
    raids = torch.tensor(raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)
    return verts, faces, raids


def build_scene(plates):
    """Concatenate plate primitives into one Mesh."""
    all_verts, all_faces, all_raids = [], [], []
    offset = 0
    for verts, faces, raids in plates:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_raids.append(raids)
        offset += verts.shape[0]
    return Mesh.from_tensors(torch.cat(all_verts), torch.cat(all_faces), torch.cat(all_raids))


# --------------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------------

def scatter_from_plates(plates, sensor_az_deg, elevation_deg=0.0, sensor_distance=1.3,
                        wavelength=0.05, grid_width=1.0, n_ray=512,
                        num_bounce=1, device='cuda'):
    """
    Ray-trace a plate scene from one or more sensor azimuths.

    `sensor_az_deg` is array-like; each entry becomes one pulse, so the whole aspect
    sweep is a single accumulate_scatters call over one octree.

    Returns (ranges, energies, trajectory): lists of P 1-D tensors plus the (1,P,3)
    sensor positions. `energies` already carries the exp(+j 2 pi z / lambda) phasor.
    """
    mesh = build_scene(plates)

    az = torch.as_tensor(np.atleast_1d(sensor_az_deg), device=device, dtype=torch.float32)
    el = torch.full_like(az, float(elevation_deg))
    dist = torch.full_like(az, float(sensor_distance))
    trajectory = spherical_to_cartesian(az, el, dist).reshape(1, -1, 3)   # (1,P,3)

    ranges, energies, _ = accumulate_scatters(
        mesh, trajectory,
        wavelength=wavelength,
        grid_width=grid_width, grid_height=grid_width,
        n_ray_width=n_ray, n_ray_height=n_ray,
        num_bounce=num_bounce,
        second_bounce_batch_size=2 ** 18,
    )
    return ranges[0], energies[0], trajectory


def coherent_rcs(energy):
    """sigma_hat = |sum of scatter phasors|^2, in arbitrary (uncalibrated) units."""
    if energy.numel() == 0:
        return 0.0
    return float(torch.abs(energy.sum()) ** 2)


def po_plate_rcs(side, tilt_rad, wavelength):
    """Monostatic physical-optics RCS of a square plate, rotated about an in-plane axis."""
    area = side ** 2
    tilt_rad = np.asarray(tilt_rad, dtype=np.float64)
    return (4 * np.pi * area ** 2 / wavelength ** 2
            * np.cos(tilt_rad) ** 2
            * np.sinc(2 * side * np.sin(tilt_rad) / wavelength) ** 2)


def _loglog_slope(x, y):
    """Least-squares slope of log(y) vs log(x), ignoring non-positive samples."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


# --------------------------------------------------------------------------------
# RCS vs plate area
# --------------------------------------------------------------------------------

def run_area_sweep(sides=(0.10, 0.14, 0.20, 0.28, 0.40), device='cuda', **overrides):
    """
    PO says sigma ~ A^2, i.e. field amplitude ~ area.

    The ray grid is held fixed while the plate grows, so the hit count tracks plate
    area directly; a grid scaled with the plate would divide the area back out.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    rows = []
    for side in sides:
        plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                           n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                           device=device)
        _, energy, _ = scatter_from_plates(
            [plate], [0.0], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=cfg['wavelength'],
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        rows.append(dict(side=side, area=side ** 2, n_hit=int(energy[0].numel()),
                         sigma=coherent_rcs(energy[0]),
                         po=float(po_plate_rcs(side, 0.0, cfg['wavelength']))))

    slope = _loglog_slope([r['area'] for r in rows], [r['sigma'] for r in rows])
    return dict(rows=rows, slope=slope, expected_slope=2.0, cfg=cfg)


# --------------------------------------------------------------------------------
# RCS vs wavelength
# --------------------------------------------------------------------------------

def run_wavelength_sweep(wavelengths=(0.2, 0.1, 0.05, 0.025, 0.0125), device='cuda', **overrides):
    """
    At specular every ray on a plate normal to the line of sight returns at the same
    range, so the coherent sum is wavelength-independent by construction. PO says
    sigma ~ 1/lambda^2. The gap is the aperture-diffraction factor geometric optics
    does not carry, and it is what an absolute RCS calibration would have to absorb.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']
    rows = []
    for lam in wavelengths:
        plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                           n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                           device=device)
        _, energy, _ = scatter_from_plates(
            [plate], [0.0], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=lam,
            grid_width=cfg['grid_width'], n_ray=cfg['n_ray'], device=device,
        )
        rows.append(dict(wavelength=lam, sigma=coherent_rcs(energy[0]),
                         po=float(po_plate_rcs(side, 0.0, lam))))

    return dict(rows=rows,
                slope=_loglog_slope([r['wavelength'] for r in rows], [r['sigma'] for r in rows]),
                po_slope=_loglog_slope([r['wavelength'] for r in rows], [r['po'] for r in rows]),
                cfg=cfg)


# --------------------------------------------------------------------------------
# phase sampling across the plate
# --------------------------------------------------------------------------------

def run_phase_sampling_sweep(u_values=(0.5, 1.5, 2.5, 4.5, 8.5, 12.5, 16.5,
                                       20.5, 24.5, 30.5, 40.5, 60.5),
                             tilt_deg=4.0, n_ray=128, device='cuda', **overrides):
    """
    Cross the phase-sampling limit Delta_ray < lambda / (4 sin theta) with an exact
    reference on the other side of it.

    The aspect is held fixed and the wavelength shrunk, so the number of phase cycles
    across the plate, u = 2 L sin(theta) / lambda, grows while the ray grid does not.
    Each lambda is chosen so u lands halfway between PO nulls -- on a sidelobe peak --
    which keeps the reference away from the zeros where any relative error blows up.

    Rays per phase cycle is lambda / (2 Delta_ray sin theta), so the rule is exactly
    "more than 2 rays per cycle". Below that the measured pattern must depart
    from PO, and the departure is aliasing, not physics.
    """
    cfg = dict(BASELINE); cfg.update(overrides)
    side = cfg['plate_side']
    phi = float(tilt_deg) * np.pi / 180.0
    d_ray = cfg['grid_width'] / (n_ray - 1)

    plate = make_plate(side, scatter_fraction=cfg['scatter_fraction'],
                       n_subdiv=cfg['n_subdiv'], elevation_deg=cfg['elevation_deg'],
                       device=device)

    rows = []
    for u in u_values:
        lam = 2 * side * np.sin(phi) / u
        _, energies, _ = scatter_from_plates(
            [plate], [0.0, tilt_deg], elevation_deg=cfg['elevation_deg'],
            sensor_distance=cfg['sensor_distance'], wavelength=lam,
            grid_width=cfg['grid_width'], n_ray=n_ray, device=device,
        )
        spec, off = coherent_rcs(energies[0]), coherent_rcs(energies[1])
        po_norm = float(np.cos(phi) ** 2 * np.sinc(u) ** 2)
        rows.append(dict(
            u=u, wavelength=lam,
            rays_per_cycle=lam / (2 * d_ray * np.sin(phi)),
            nyquist_ratio=d_ray / (lam / (4 * np.sin(phi))),
            measured_norm=off / spec if spec else float('nan'),
            po_norm=po_norm,
            error_db=10 * np.log10((off / spec) / po_norm) if spec and po_norm else float('nan'),
        ))
    return dict(rows=rows, tilt_deg=tilt_deg, n_ray=n_ray, d_ray=d_ray, cfg=cfg)
