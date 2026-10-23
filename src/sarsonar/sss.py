'''
Side scan sonar.

The platform flies a straight track past the object, staring broadside and down at a fixed
elevation, and pings as it goes. Each ping is a fan of rays from the sensor; its scatters are
weighted by a two-way Gaussian azimuth beam and spread onto a range signal. The pings are stacked
as columns: rows run out in slant range (near range at the bottom), columns along the track.

Entry points:
    render_side_scan(mesh, azimuth_deg, elevation_deg, ...)  -- one track -> (H,W) image
    side_scan_image(mesh, sensor_position, ...)              -- full control -> (T,H,W)
'''
import numpy as np
import torch

from .geometry import spherical_to_cartesian
from .mesh import as_mesh
from .scatter import accumulate_scatters_side_scan, centered_linspace
from .signals import interpolate_signal


# a Gaussian's full width at half maximum is 2*sqrt(2*ln2) standard deviations
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def beam_spread_weights(angle_difference_deg, beam_width_deg):
    '''
    Gaussian beam pattern: the weight a beam gives a scatter angle_difference_deg off its
    boresight.

    beam_width_deg is the full width at half maximum of the pattern, so a scatter sitting half a
    beam width off boresight keeps half its energy. The pattern peaks at 1 rather than
    integrating to 1: a scatter dead on boresight keeps its full energy.

    inputs:
        angle_difference_deg (...): scatter azimuth off the beam's boresight, in degrees
        beam_width_deg (float): FWHM of the beam, in degrees
    outputs:
        weights (...): beam pattern value in (0, 1]
    '''
    sigma_deg = beam_width_deg / FWHM_PER_SIGMA
    return torch.exp(-0.5 * (angle_difference_deg / sigma_deg) ** 2)


def apply_beam_pattern(energy, tx_azimuth_deg, rx_azimuth_deg, beam_width_deg):
    '''
    Weight scatter energies by a two-way Gaussian beam: transmit at each scatter's launch azimuth,
    receive at its arrival azimuth. Runs after accumulate_scatters, which never applies a beam.

    Each one-way Gaussian is sqrt(2) wider than beam_width_deg, so on a first bounce, where the two
    azimuths agree, the product is the two-way beam of FWHM beam_width_deg.

    inputs:
        energy (R,): scatter energies, real or complex
        tx_azimuth_deg (R,): launch azimuth of the ray behind each scatter, off boresight, in degrees
        rx_azimuth_deg (R,): arrival azimuth of each scatter, off the same boresight, in degrees
        beam_width_deg (float): two-way FWHM of the beam, in degrees
    outputs:
        energy (R,): the weighted energies
    '''
    one_way_beam_width_deg = beam_width_deg * np.sqrt(2)
    return (energy * beam_spread_weights(tx_azimuth_deg, one_way_beam_width_deg)
                   * beam_spread_weights(rx_azimuth_deg, one_way_beam_width_deg))


def side_scan_image(
    mesh,
    sensor_position,
    track_length = 1.2,
    num_pings = 128,
    elevation_fov_deg = 23.0,
    azimuth_beam_width_deg = 0.1,
    num_ray_width = 1,
    num_ray_height = 550,

    image_width = 128,
    image_height = 128,
    image_cross_range_swath = 1.2,
    image_range_swath = 1.2,

    wavelength = None,
    num_bounce = 2,
    water_absorption = 0.0,
    tvg_exponent = 0.0,
    spatial_bw = 30,
    spatial_fs = 60,
    waveform = 'gaussian',
    use_sig_magnitude = True,

    ray_batch_size = 2**9,
    octree = None,
        ):
    '''
    Side scan sonar image of a mesh, from a track centered on sensor_position.

    inputs:
        mesh (Mesh): the scene
        sensor_position (3,): the track center. The track runs through it perpendicular to the line
            of sight to the origin, and every ping looks along that line of sight
        track_length (float): along-track extent flown; keep it >= image_cross_range_swath or the
            outer columns have no ping abeam of them
        num_pings (int): pings along the track
        elevation_fov_deg (float): vertical extent of each ping's ray fan
        azimuth_beam_width_deg (float): two-way FWHM of the along-track beam; the fan spans 3x this
        num_ray_width/num_ray_height (int): rays across the fan in azimuth and elevation
        image_width/image_height (int): pixels
        image_cross_range_swath (float): along-track extent of the image
        image_range_swath (float): slant range extent of the image, centered on the track center's
            distance to the origin
        wavelength (float): when given, scatter energies are complex and sum coherently
        num_bounce (int): reflections traced per ray
        water_absorption (float): two-way absorption in nepers per unit length, 0 = off
        tvg_exponent (float): time varying gain, signals multiplied by range^n, 0 = off
        spatial_bw (float): waveform bandwidth in cycles per unit range
        spatial_fs (float): range samples per unit range
        waveform (str): 'sinc', 'hamming', 'gaussian', 'lfm' or 'barker13'
        use_sig_magnitude (bool): image the signal envelope rather than the complex signal
        ray_batch_size (int): rays per ray tracing batch; halves itself on CUDA OOM
        octree (Octree): built once per mesh and passed back in when rendering many tracks
    outputs:
        images (T,H,W): raw amplitude, T = 1, near range in the bottom row
        row_coords (T,H): slant range of each row, far range first
        col_coords (W,): along-track offset of each column
    '''
    mean_sensor_position = sensor_position.reshape(3)

    # figure out each sensor position. The track is a straight line through
    # mean_sensor_position, running along the sensor's right vector, so the object stays off
    # to one side of the platform the whole way -- that sidelong look is what makes this a
    # side scan.
    device = mean_sensor_position.device
    line_of_sight   = torch.nn.functional.normalize(-mean_sensor_position, dim=-1)  # (3,) sensor -> origin
    world_up        = torch.tensor([0.0, 0.0, 1.0], device=device)                  # +z
    track_direction = torch.linalg.cross(line_of_sight, world_up)                   # (3,) sensor's right, unnormalized
    if track_direction.norm() < 1e-6:
        # a nadir look zeroes the cross product, so the right comes from the sensor's azimuth instead
        azimuth = torch.atan2(mean_sensor_position[1], mean_sensor_position[0])
        track_direction = torch.stack((-torch.sin(azimuth), torch.cos(azimuth), torch.zeros_like(azimuth)))
    track_direction = torch.nn.functional.normalize(track_direction, dim=-1)        # (3,)

    # calculate trajectory of the sensor
    ping_offsets = centered_linspace(track_length, num_pings, device)               # (P,)
    trajectory   = (mean_sensor_position.reshape(1, 3)
                    + ping_offsets.reshape(num_pings, 1) * track_direction.reshape(1, 3))  # (P,3)

    # figure out the camera matrix for each sensor position. Broadside: every ping shares the
    # single orientation built at the track center, so all the boresights are parallel and only
    # the translation column changes down the track. That is what separates this from a
    # spotlight geometry, where each pose steers onto the origin and the object therefore never
    # leaves azimuth 0; here the object drifts across the beam as the platform passes, and the
    # azimuth weighting below has something to bite on.
    # Columns are (right, up, forward, center), the srn_cars layout of pose_matrix.
    up_vector = torch.linalg.cross(track_direction, line_of_sight)                  # (3,)
    mean_pose = torch.zeros(4, 4, device=device)                                    # (4,4)
    mean_pose[:3, 0] = track_direction
    mean_pose[:3, 1] = up_vector
    mean_pose[:3, 2] = line_of_sight
    mean_pose[:3, 3] = mean_sensor_position
    mean_pose[ 3, 3] = 1.0
    poses = mean_pose.reshape(1, 4, 4).repeat(num_pings, 1, 1)                      # (P,4,4)
    poses[:, :3, 3] = trajectory                                                    # only the center moves

    # the ray fan only needs to cover the angles where the beam pattern still has weight. The
    # Gaussian is down to ~0.002 at 1.5 beam widths off boresight, so 3 beam widths total spends
    # rays where the energy actually is instead of on the tails.
    fan_azimuth_fov_deg   = azimuth_beam_width_deg * 3
    fan_elevation_fov_deg = elevation_fov_deg

    # use accumulate_scatters to ray trace on the object for each camera matrix
    scatter_ranges, scatter_energies, scatter_azimuths, scatter_arrival_azimuths, _ = accumulate_scatters_side_scan(
        mesh,
        poses.unsqueeze(0),                      # (1,P,4,4), one scene
        wavelength     = wavelength,
        fov_width_deg  = fan_azimuth_fov_deg,
        fov_height_deg = fan_elevation_fov_deg,
        n_ray_width    = num_ray_width,
        n_ray_height   = num_ray_height,
        num_bounce     = num_bounce,
        second_bounce_batch_size = ray_batch_size,
        water_absorption = water_absorption,
        octree         = octree,
    )  # list[T][P] of (R',) each

    # two-way beam pattern: transmit at the launch azimuth, receive at the arrival azimuth
    scatter_energies = [
        [apply_beam_pattern(energy, tx, rx, azimuth_beam_width_deg)
         for energy, tx, rx in zip(energies_t, tx_t, rx_t)]
        for energies_t, tx_t, rx_t in zip(scatter_energies, scatter_azimuths, scatter_arrival_azimuths)
    ]  # list[T][P] of (R',)

    # interpolate signal, on one range window shared by every ping so the columns line up.
    # Centered on the target, so the object sits mid-window. The window can start inside
    # swath_near, which is correct: the object stands off the seafloor and returns before it.
    target_range  = torch.linalg.norm(mean_sensor_position)
    assert 0 < image_range_swath / 2 <= float(target_range), \
        'image_range_swath %g must be positive and stay in front of the sensor at distance %g' % (image_range_swath, float(target_range))
    window_center = target_range.reshape(1)  # (1,)
    window_radius = image_range_swath / 2 + 1 / spatial_fs  # the image's rows plus a sample of margin
    signals = []
    sample_z = []
    for ranges_t, energies_t in zip(scatter_ranges, scatter_energies):
        signals_t = []
        sample_z_t = []
        for scatter_range, scatter_energy in zip(ranges_t, energies_t):
            signal_p, sample_z_p = interpolate_signal(
                scatter_range.unsqueeze(0) / 2,   # (1,R') round trip -> one-way range
                scatter_energy.unsqueeze(0),      # (1,R')
                window_radius,
                window_center,
                spatial_bw = spatial_bw, spatial_fs = spatial_fs,
                waveform = waveform,
            )
            signals_t.append(signal_p.squeeze(0))      # (Z,)
            sample_z_t.append(sample_z_p.squeeze(0))   # (Z,)
        signals.append(torch.stack(signals_t))         # (P,Z)
        sample_z.append(torch.stack(sample_z_t))       # (P,Z)
    signals  = torch.stack(signals)   # (T,P,Z)
    sample_z = torch.stack(sample_z)  # (T,P,Z)
    T = signals.shape[0]

    # time varying gain: a receiver ramp of R^n against the seafloor's fall with range. Absolute
    # rather than referenced to a range, so it rescales the image as well as tilting it. n=0 turns it off.
    if tvg_exponent:
        signals = signals * sample_z ** tvg_exponent  # (T,P,Z)

    if use_sig_magnitude:
        signals = signals.abs()  # project the envelope; a complex image is not displayable

    # stack the interpolated signals as columns -- one per ping -- and resample onto a fixed
    # pixel grid centered on the target range, rows in slant range and columns in cross range
    # (ping/along-track position). Unlike a ground plane projection this never resamples range
    # onto the seafloor, so the image shows slant range rather than ground range
    shared_sample_z = sample_z[:, 0, :]  # (T,Z) every ping shares the one range window

    row_frac = torch.linspace(0.5, -0.5, image_height, device=signals.device, dtype=shared_sample_z.dtype)  # (H,) far to near
    col_frac = torch.linspace(-0.5, 0.5, image_width, device=signals.device, dtype=ping_offsets.dtype)      # (W,) low to high
    row_coords = target_range.reshape(1, 1).expand(T, 1) + row_frac.reshape(1, image_height) * image_range_swath  # (T,H) far range first
    col_coords = col_frac * image_cross_range_swath  # (W,) low ping offset first

    # normalize the target coordinates into grid_sample's [-1,1] convention over the sampled
    # (P,Z) grid, so a target outside the sampled range/track window comes back zero
    z_lo, z_hi = shared_sample_z[:, :1], shared_sample_z[:, -1:]  # (T,1)
    p_lo, p_hi = ping_offsets[0], ping_offsets[-1]  # ()
    y_norm = (2 * (row_coords - z_lo) / (z_hi - z_lo) - 1).reshape(T, image_height, 1).expand(T, image_height, image_width)  # (T,H,W)
    x_norm = (2 * (col_coords - p_lo) / (p_hi - p_lo) - 1).reshape(1, 1, image_width).expand(T, image_height, image_width)   # (T,H,W)
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

    return images, row_coords, col_coords
    #      (T,H,W), (T,H) slant range of each row (far to near), (W,) cross range of each column (low to high)


def render_side_scan(mesh, azimuth_deg=0.0, elevation_deg=40.0, distance=7.5, device=None, **kwargs):
    '''
    One side scan sonar image of a mesh, from a track centered on (azimuth, elevation, distance).

    inputs:
        mesh (Mesh or str): a loaded Mesh, or an .obj path loaded with load_mesh's defaults
        azimuth_deg (float): direction of the track center from the origin
        elevation_deg (float): look angle down to the origin, in [0, 90]
        distance (float): track center distance to the origin
        device: where to render; defaults to the Mesh's device, or cuda when available for a path
        kwargs: forwarded to side_scan_image
    outputs:
        image (H,W): raw amplitude, near range at the bottom, along-track left to right
    '''
    assert 0.0 <= elevation_deg <= 90.0, 'elevation %.1f deg puts the sensor below the seafloor or past overhead' % elevation_deg
    mesh = as_mesh(mesh, device)
    position = spherical_to_cartesian(*(torch.tensor(float(x), device=mesh.device)
                                        for x in (azimuth_deg, elevation_deg, distance)))
    return side_scan_image(mesh, position, **kwargs)[0][0]
