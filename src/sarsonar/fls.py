'''
Forward looking sonar: one perspective ray trace, many beam-steered pings.

The sensor sits at one position with its boresight on the scene origin and fans rays over
ray_fov_az x ray_fov_el. That fan is ray traced once. Each ping then steers the two-way beam to its
own azimuth, spaced evenly over image_azimuth_range, and the beam-weighted scatters are spread onto
a range signal. Each ping is one column of the image, near range in the bottom row.

Keep image_azimuth_range inside ray_fov_az: a ping at the edge of the image still needs rays out to
a beam width or so past it.

Entry points:
    render_fls(mesh, azimuth_deg, elevation_deg, ...)  -- one view -> (H,W) image
    fls_image(mesh, sensor_positions, ...)             -- a batch of positions -> (T,H,W)
'''
import torch

from .geometry import spherical_to_cartesian
from .mesh import as_mesh
from .scatter import accumulate_scatters_perspective, centered_linspace
from .signals import interpolate_signal
from .sss import apply_beam_pattern


def fls_image(
    mesh,
    sensor_positions,
    image_azimuth_range = 9.15,
    num_pings = 128,
    num_range_values = 128,
    ray_fov_az = 10.0,
    ray_fov_el = 23.0,
    azimuth_beam_width_deg = 0.1,
    num_ray_width = 350,
    num_ray_height = 550,

    image_range_swath = 1.2,

    wavelength = None,
    num_bounce = 2,
    water_absorption = 0.0,
    tvg_exponent = 0.0,
    spatial_bw = 32,
    spatial_fs = 64,
    waveform = 'gaussian',
    use_sig_magnitude = True,

    ray_batch_size = 2**9,
    octree = None,
        ):
    '''
    Forward looking sonar image(s) of a mesh, one per sensor position.

    inputs:
        mesh (Mesh): the scene
        sensor_positions (T,3): where the sensor sits; its boresight always points at the origin
        image_azimuth_range (float): total azimuth span of the image in degrees, centered on
            boresight. Keep it inside ray_fov_az, with room for the beam at the edges
        num_pings (int): pings across that span, i.e. columns of the image, left to right
        num_range_values (int): rows of the image, far range in the top row
        ray_fov_az/ray_fov_el (float): azimuth and elevation extent of the ray fan, in degrees
        azimuth_beam_width_deg (float): two-way FWHM of the beam each ping steers, in degrees
        num_ray_width/num_ray_height (int): rays across the fan in azimuth and elevation
        image_range_swath (float): one-way range the rows span, from sensor_distance -
            image_range_swath/2 in the bottom row to sensor_distance + image_range_swath/2 in the
            top row, so moving the sensor moves the window with it
        wavelength (float): when given, scatter energies are complex and sum coherently
        water_absorption (float): two-way absorption in nepers per unit length, 0 = off
        tvg_exponent (float): time varying gain, signals multiplied by range^n before image
            formation, 0 = off
        ray_batch_size (int): rays per ray tracing batch; halves itself on CUDA OOM
        remaining arguments: as in sss.side_scan_image
    outputs:
        images (T,H,W): the sonar image(s), H = num_range_values, W = num_pings
        row_ranges (H,): one-way range of each row, far range first
        ping_azimuths (W,): steering azimuth of each column in degrees, left to right
    '''
    device = sensor_positions.device
    T = sensor_positions.shape[0]

    # ray trace once: a single perspective view per sensor position, boresight on the origin
    scatter_ranges, scatter_energies, scatter_azimuths, scatter_arrival_azimuths, _ = accumulate_scatters_perspective(
        mesh,
        sensor_positions.reshape(T, 1, 3),       # (T,1,3), one view per sensor position
        wavelength     = wavelength,
        fov_width_deg  = ray_fov_az,
        fov_height_deg = ray_fov_el,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = ray_batch_size,
        water_absorption = water_absorption,
        octree         = octree,
    )  # list[T][1] of (R',) each

    # the range swath brackets the scene origin along the line of sight
    sensor_distance = torch.linalg.norm(sensor_positions, dim=-1).mean().item()
    assert 0 < image_range_swath / 2 <= sensor_distance, \
        'image_range_swath %g must be positive and stay in front of the sensor at distance %g' % (image_range_swath, sensor_distance)
    range_near = sensor_distance - image_range_swath / 2
    range_far  = sensor_distance + image_range_swath / 2

    # one signal window shared by every ping, covering the rows plus a sample of margin
    window_center = torch.tensor([(range_near + range_far) / 2], device=device)  # (1,)
    window_radius = (range_far - range_near) / 2 + 1 / spatial_fs

    # each ping steers the beam to its own azimuth, left to right across the image
    ping_azimuths = centered_linspace(image_azimuth_range, num_pings, device)  # (P,)

    signals = []
    for t in range(T):
        scatter_z = scatter_ranges[t][0] / 2           # (R',) round trip -> one-way range
        energies  = scatter_energies[t][0]             # (R',)
        tx        = scatter_azimuths[t][0]             # (R',)
        rx        = scatter_arrival_azimuths[t][0]     # (R',)
        signals_t = []
        for ping_azimuth in ping_azimuths:
            # ping adjusted tx and rx azimuths through the two-way beam
            energy_p = apply_beam_pattern(energies, tx - ping_azimuth, rx - ping_azimuth,
                                          azimuth_beam_width_deg)  # (R',)

            # scatters the beam weights to exactly zero add nothing; dropping them bounds memory at high fs
            lit = energy_p != 0  # (R',)
            signal_p, sample_z = interpolate_signal(
                scatter_z[lit].unsqueeze(0),   # (1,R'')
                energy_p[lit].unsqueeze(0),    # (1,R'')
                window_radius,
                window_center,
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                waveform = waveform,
            )
            signals_t.append(signal_p.squeeze(0))  # (Z,)
        signals.append(torch.stack(signals_t))     # (P,Z)
    signals  = torch.stack(signals)  # (T,P,Z)
    sample_z = sample_z.squeeze(0)   # (Z,) every ping shares the one window

    # time varying gain on the signal, before image formation: each sample scaled by its range^n
    if tvg_exponent:
        signals = signals * sample_z ** tvg_exponent  # (T,P,Z)

    if use_sig_magnitude:
        signals = signals.abs()  # project the envelope; a complex image is not displayable

    # resample each ping's signal onto the rows, far range first; the columns are the pings themselves
    row_ranges = torch.linspace(range_far, range_near, num_range_values, device=device, dtype=sample_z.dtype)  # (H,)
    y_norm = (2 * (row_ranges - sample_z[0]) / (sample_z[-1] - sample_z[0]) - 1
              ).reshape(1, num_range_values, 1).expand(T, num_range_values, num_pings)  # (T,H,W)
    x_norm = torch.linspace(-1.0, 1.0, num_pings, device=device, dtype=sample_z.dtype
              ).reshape(1, 1, num_pings).expand(T, num_range_values, num_pings)         # (T,H,W)
    sample_grid = torch.stack((x_norm, y_norm), dim=-1)  # (T,H,W,2)

    grid = signals.transpose(1, 2).unsqueeze(1)  # (T,1,Z,P)
    if torch.is_complex(grid):
        images = torch.complex(
            torch.nn.functional.grid_sample(grid.real, sample_grid, align_corners=True, padding_mode='zeros'),
            torch.nn.functional.grid_sample(grid.imag, sample_grid, align_corners=True, padding_mode='zeros'),
        ).squeeze(1)  # (T,H,W)
    else:
        images = torch.nn.functional.grid_sample(
            grid, sample_grid, align_corners=True, padding_mode='zeros'
        ).squeeze(1)  # (T,H,W)

    return images, row_ranges, ping_azimuths
    #      (T,H,W), (H,) one-way range of each row (far to near), (W,) azimuth of each column (left to right)


def render_fls(mesh, azimuth_deg=0.0, elevation_deg=40.0, distance=7.5, device=None, **kwargs):
    '''
    One forward looking sonar image of a mesh, from a sensor at (azimuth, elevation, distance)
    looking at the origin.

    inputs:
        mesh (Mesh or str): a loaded Mesh, or an .obj path loaded with load_mesh's defaults
        azimuth_deg, elevation_deg, distance (float): the sensor position
        device: where to render; defaults to the Mesh's device, or cuda when available for a path
        kwargs: forwarded to fls_image
    outputs:
        image (H,W): raw amplitude, near range at the bottom, azimuth left to right
    '''
    assert elevation_deg >= 0.0, 'elevation %.1f deg puts the sensor below the seafloor' % elevation_deg
    mesh = as_mesh(mesh, device)
    position = spherical_to_cartesian(*(torch.tensor(float(x), device=mesh.device)
                                        for x in (azimuth_deg, elevation_deg, distance)))
    return fls_image(mesh, position.reshape(1, 3), **kwargs)[0][0]
