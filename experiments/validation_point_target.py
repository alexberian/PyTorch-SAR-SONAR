"""
Point-scatterer harness for sar_validation.py.

Validates the coherent signal simulation and imaging chain against the closed-form response of a
single ideal point scatterer at a known location. The ray tracer is bypassed on purpose: one
analytic scatterer is injected directly into interpolate_signal, so anything that fails here is a
signal or imaging defect and not a geometry defect.

    aperture phase residual   arg( s_hat(R_p) * conj(A e^{j4 pi R_p / lambda}) ) across pulses
    peak location error       sub-pixel image peak vs the projected target position
    mainlobe width            -3 dB widths vs 1/(B_s cos el) and lambda/(4 sin(dtheta/2))
    PSLR, ISLR                peak and integrated sidelobe ratios
"""
import numpy as np
import torch

from sarsonar.geometry import pose_matrix
from sarsonar.sar import generate_trajectory
from sarsonar.signals import apply_snr, interpolate_signal


# --------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------

def simulate_point_target(
    target_xyz,
    azimuth_deg=0.0,
    elevation_deg=30.0,
    sensor_distance=1.3,
    azimuth_spread=90.0,
    num_pulses=64,
    wavelength=0.5,
    spatial_bw=73.0,
    spatial_fs=73.0,
    region_radius=1.7,
    waveform='sinc',
    amplitude=1.0,
    snr_db=None,
    trajectory_type='circular',
    range_model='spherical',
    device='cpu',
):
    """
    Build the exact phase history of one ideal point scatterer.

    Follows the conventions of the full pipeline so this measures the same code
    path the paper's figures use:
      - the scatter accumulator attaches exp(+j 2 pi / lambda * two_way_range)
        to the scatter energy
      - sar_image hands `interpolate_signal` the *one-way* range, two_way/2,
        and the sensor's distance to the origin

    `range_model` selects the propagation geometry:
      'spherical' -- the true range |traj - q| (correct physics, the default)
      'planar'    -- |traj| + q . forward, i.e. the same far-field approximation
                     projected_CBP assumes internally.
    Imaging a 'planar' history isolates the imager's approximation error: any
    residual peak displacement under 'planar' is an implementation bug, whereas
    displacement present only under 'spherical' is far-field model error.

    Returns a dict of the signal, its range samples, the trajectory, and the
    per-pulse ground-truth range to the target.
    """
    target = torch.tensor(target_xyz, device=device, dtype=torch.float32)  # (3,)

    # one pose at the requested look angle, then the pulse trajectory around it
    pose = pose_matrix(
        azimuth_deg, elevation_deg, sensor_distance, device=device
    ).reshape(1, 4, 4)
    true_trajectory, perceived_trajectory, cam_azimuth_deg = generate_trajectory(
        pose,
        trajectory_type=trajectory_type,
        n_pulses=num_pulses,
        azimuth_spread_deg=azimuth_spread,
    )  # (1,P,3), (1,P,3), (1,)
    P = true_trajectory.shape[1]

    # one-way range from each pulse to the target
    sensor_dist = torch.linalg.norm(true_trajectory[0], dim=-1)          # (P,)
    if range_model == 'spherical':
        target_range = torch.linalg.norm(
            true_trajectory[0] - target.reshape(1, 3), dim=-1
        )  # (P,)
    elif range_model == 'planar':
        forward = -true_trajectory[0] / sensor_dist.unsqueeze(-1)        # (P,3)
        target_range = sensor_dist + torch.sum(
            target.reshape(1, 3) * forward, dim=-1
        )  # (P,)
    else:
        raise ValueError("range_model should be 'spherical' or 'planar', but got %s"
                         % range_model)

    # the ideal scatterer: unit energy carrying the two-way propagation phase
    scatter_z = target_range.reshape(P, 1)                       # (P,1) one-way
    scatter_e = amplitude * torch.exp(
        1j * 2 * np.pi / wavelength * (2 * scatter_z)
    )                                                            # (P,1) complex

    signals, sample_z = interpolate_signal(
        scatter_z,
        scatter_e,
        region_radius,
        sensor_dist,                                             # (P,)
        spatial_bw=spatial_bw,
        spatial_fs=spatial_fs,
        waveform=waveform,
        batch_size=None,
    )  # (P,Z), (P,Z)

    if snr_db is not None:
        signals = apply_snr(signals.reshape(1, -1), snr_db).reshape(signals.shape)

    return dict(
        signals=signals.reshape(1, P, -1),
        sample_z=sample_z.reshape(1, P, -1),
        true_trajectory=true_trajectory,
        perceived_trajectory=perceived_trajectory,
        cam_azimuth_deg=cam_azimuth_deg,
        target_range=target_range.reshape(1, P),
        target=target,
        wavelength=wavelength,
        spatial_bw=spatial_bw,
        spatial_fs=spatial_fs,
        amplitude=amplitude,
        elevation_deg=elevation_deg,
        azimuth_spread=azimuth_spread,
    )


def reconstruct_at_range(signals, sample_z, query_z, spatial_fs):
    """
    Band-limited (sinc) reconstruction of the sampled signal at arbitrary ranges.

    This is the same interposummation operator CBP_2D applies internally, so the
    residual it produces is a direct measurement of interposummation's
    reconstruction fidelity.

    inputs:
        signals  (T,P,Z)
        sample_z (T,P,Z)
        query_z  (T,P)
    outputs:
        (T,P) complex
    """
    return torch.sum(
        signals * torch.sinc(spatial_fs * (query_z.unsqueeze(-1) - sample_z)),
        dim=-1,
    )


# --------------------------------------------------------------------------------
# metric: aperture phase residual
# --------------------------------------------------------------------------------

def aperture_phase_residual(sim):
    """
    Compare the simulated phase history against the analytic exp(-j4 pi R/lambda).

    Reconstructs the sampled signal back at the target's true range and divides out
    the ideal response. A coherent chain leaves a constant residual; drift or
    structure across the aperture is a coherence defect.
    """
    s_hat = reconstruct_at_range(
        sim['signals'], sim['sample_z'], sim['target_range'], sim['spatial_fs']
    )[0]  # (P,)

    R = sim['target_range'][0]                                        # (P,)
    ideal = sim['amplitude'] * torch.exp(1j * 4 * np.pi * R / sim['wavelength'])

    residual = s_hat * torch.conj(ideal) / (torch.abs(ideal) ** 2)
    phase_err_deg = torch.angle(residual) * 180 / np.pi               # (P,)
    mag_ratio = torch.abs(s_hat) / torch.abs(ideal)                   # (P,)

    # remove the constant part: a fixed offset is a harmless global phase, whereas
    # variation across the aperture is what destroys coherent integration
    centered = torch.angle(residual * torch.conj(torch.mean(residual)))
    centered_deg = centered * 180 / np.pi

    return dict(
        s_hat=s_hat,
        ideal=ideal,
        phase_err_deg=phase_err_deg,
        phase_err_centered_deg=centered_deg,
        mag_ratio=mag_ratio,
        rms_phase_err_deg=float(torch.sqrt(torch.mean(centered_deg ** 2))),
        max_phase_err_deg=float(torch.max(torch.abs(centered_deg))),
        mag_ripple_db=float(
            20 * torch.log10(torch.max(mag_ratio) / torch.min(mag_ratio))
        ),
        mean_mag_ratio=float(torch.mean(mag_ratio)),
    )


# --------------------------------------------------------------------------------
# image-domain metrics
# --------------------------------------------------------------------------------

def expected_peak_pixel(sim, image_width, image_height,
                        image_plane_width, image_plane_height):
    """
    Where the target should land, in (row, col).

    CBP_2D builds the pixel grid then rotates it into the world:
    world_xy = R(theta) @ pixel_xy, so the inverse map is
    pixel_xy = R(theta)^T @ world_xy.
    """
    theta = float(sim['cam_azimuth_deg'][0] + 90) * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    wx, wy = float(sim['target'][0]), float(sim['target'][1])

    px = c * wx + s * wy       # R^T @ world
    py = -s * wx + c * wy

    # x spans [-w/2, w/2] over columns; y spans [+h/2, -h/2] over rows
    col = (px + image_plane_width / 2) / image_plane_width * (image_width - 1)
    row = (image_plane_height / 2 - py) / image_plane_height * (image_height - 1)
    return row, col


def _subpixel_peak(cut, idx):
    """Quadratic-fit refinement of a discrete peak; returns the offset in samples."""
    if idx <= 0 or idx >= len(cut) - 1:
        return 0.0
    a, b, c = float(cut[idx - 1]), float(cut[idx]), float(cut[idx + 1])
    denom = a - 2 * b + c
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def _minus_3db_width(cut, peak_idx, pixel_size):
    """-3 dB (intensity) width of the mainlobe, linearly interpolated, in scene units."""
    peak = float(cut[peak_idx])
    if peak <= 0:
        return float('nan')
    half = peak / np.sqrt(2.0)

    def crossing(direction):
        i = peak_idx
        while 0 < i < len(cut) - 1:
            j = i + direction
            if cut[j] <= half:
                # linear interpolation between samples i and j
                num = float(cut[i]) - half
                den = float(cut[i]) - float(cut[j])
                return abs(i - peak_idx) + (num / den if den != 0 else 0.0)
            i = j
        return float('nan')

    return (crossing(-1) + crossing(+1)) * pixel_size


def _first_nulls(cut, peak_idx):
    """Indices of the first local minima either side of the peak (mainlobe extent)."""
    def null(direction):
        i = peak_idx
        while 0 < i + direction < len(cut) - 1:
            j = i + direction
            if cut[j] > cut[i]:      # started climbing again -> i was the null
                return i
            i = j
        return i
    return null(-1), null(+1)


def _pslr_islr(cut, peak_idx):
    """Peak- and integrated-sidelobe ratio of a 1-D cut, in dB."""
    lo, hi = _first_nulls(cut, peak_idx)
    peak = float(cut[peak_idx])

    sidelobes = np.concatenate([cut[:lo + 1], cut[hi:]])
    pslr = 20 * np.log10(np.max(sidelobes) / peak) if sidelobes.size else float('nan')

    power = cut.astype(np.float64) ** 2
    main_e = power[lo:hi + 1].sum()
    side_e = power.sum() - main_e
    islr = 10 * np.log10(side_e / main_e) if main_e > 0 else float('nan')
    return pslr, islr, (lo, hi)


def impulse_response_metrics(image, sim, image_plane_width, image_plane_height):
    """
    Peak location, resolution, PSLR and ISLR from the image.

    Range maps to the image row axis and cross-range to the column axis: with the
    image plane rotated by (azimuth + 90), the CBP projection coordinate reduces to
    r_coord = -py, so py (rows) is range and px (columns) is cross-range.
    """
    img = image.detach().cpu().numpy()
    H, W = img.shape
    peak_row, peak_col = np.unravel_index(np.argmax(img), img.shape)

    row_cut = img[:, peak_col]   # range
    col_cut = img[peak_row, :]   # cross-range

    row_px = image_plane_height / (H - 1)
    col_px = image_plane_width / (W - 1)

    sub_row = peak_row + _subpixel_peak(row_cut, peak_row)
    sub_col = peak_col + _subpixel_peak(col_cut, peak_col)
    exp_row, exp_col = expected_peak_pixel(
        sim, W, H, image_plane_width, image_plane_height
    )

    rng_res = _minus_3db_width(row_cut, peak_row, row_px)
    crs_res = _minus_3db_width(col_cut, peak_col, col_px)

    rng_pslr, rng_islr, rng_nulls = _pslr_islr(row_cut, peak_row)
    crs_pslr, crs_islr, crs_nulls = _pslr_islr(col_cut, peak_col)

    # closed-form predictions
    el = sim['elevation_deg'] * np.pi / 180
    dtheta = sim['azimuth_spread'] * np.pi / 180
    pred_rng = 1.0 / (sim['spatial_bw'] * np.cos(el))
    pred_crs = sim['wavelength'] / (4 * np.sin(dtheta / 2))

    return dict(
        image=img,
        peak_row=sub_row, peak_col=sub_col,
        expected_row=exp_row, expected_col=exp_col,
        row_err_px=sub_row - exp_row, col_err_px=sub_col - exp_col,
        peak_err_px=float(np.hypot(sub_row - exp_row, sub_col - exp_col)),
        peak_err_scene=float(np.hypot((sub_row - exp_row) * row_px,
                                      (sub_col - exp_col) * col_px)),
        range_res=rng_res, crossrange_res=crs_res,
        predicted_range_res=pred_rng, predicted_crossrange_res=pred_crs,
        range_res_ratio=rng_res / pred_rng,
        crossrange_res_ratio=crs_res / pred_crs,
        range_pslr_db=rng_pslr, crossrange_pslr_db=crs_pslr,
        range_islr_db=rng_islr, crossrange_islr_db=crs_islr,
        row_cut=row_cut, col_cut=col_cut,
        row_nulls=rng_nulls, col_nulls=crs_nulls,
        row_px=row_px, col_px=col_px,
    )
