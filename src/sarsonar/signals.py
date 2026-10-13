'''
Range signals: every scatter, an (energy, range) pair, is spread onto a uniformly sampled range
window through the transmit waveform's matched-filter response.
'''
import math

import numpy as np
import torch


# Barker-13 code: the longest known Barker sequence, whose autocorrelation has
# uniform sidelobes of magnitude 1 against a peak of 13 (-22.3 dB).
BARKER13 = (+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1)

# Hamming taper H(f) = A0 + 2*A1*cos(2*pi*f/B) on the band, whose inverse FT is three shifted sincs
HAMMING_A0, HAMMING_A1 = 0.54, 0.23


def _matched_filter_window(x, dz, device, dtype):
    """
    Build the effective matched-filter window w(z) = h(z) * x(z) for a transmit
    waveform x(z) sampled on a uniform grid of spacing dz, where h(z) = conj(x(-z))
    is the matched filter.  w(z) is just the autocorrelation of x and plays the same
    role as the ideal sinc window in interpolate_signal:
        s_mf(z_o) = w(z) * g(z) |_{z=z_o} = sum_r E_r w(-z_o - z_r/2).
    It is normalized to unit peak magnitude so it is directly comparable to the sinc
    window (unit peak, no range sidelobes).

    Inputs:
        x (np.ndarray): complex baseband transmit waveform x(z), centered on the grid
        dz (float): grid spacing in range units
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
    Returns:
        window (callable): window(x) -> complex tensor, same shape as x; linearly
            interpolates w(z) at the continuous range offsets x, 0 outside support
    """
    # matched-filter output w[k] = sum_j conj(x[j]) x[j+k] (autocorrelation of x);
    # np.correlate conjugates its second argument, peak lands at the center.
    w = np.correlate(x, x, mode='full')          # length 2*len(x)-1 (odd)
    w = w / np.max(np.abs(w))                     # unit peak magnitude

    M = len(w)
    z0 = -(M - 1) / 2.0 * dz                      # range of the first grid sample
    cdtype = torch.complex64 if dtype in (torch.float32, torch.float16) else torch.complex128
    w_grid = torch.tensor(w, device=device, dtype=cdtype)  # (M,)

    def window(x):
        # linear interpolation of w(z) at the continuous offsets x, 0 outside support
        idxf = (x - z0) / dz
        i0 = torch.floor(idxf)
        frac = (idxf - i0).to(cdtype)
        i0 = i0.long()
        valid = (i0 >= 0) & (i0 < M - 1)
        i0c = i0.clamp(0, M - 2)
        w0 = w_grid[i0c]
        w1 = w_grid[i0c + 1]
        return (w0 + (w1 - w0) * frac) * valid.to(cdtype)

    return window


def make_lfm_window(spatial_bw, device, dtype, oversample=32, time_bandwidth=100.0):
    """
    Precompute the matched-filter window for an LFM (linear frequency modulation)
    chirp transmit waveform x(z) = exp(j*pi*K*z^2) of bandwidth B = spatial_bw.
    The matched-filter output is approximately a sinc of mainlobe width 1/B (matching
    the sinc window's first null) with the classic ~ -13 dB range sidelobes.

    Inputs:
        spatial_bw (float): chirp bandwidth B; sets range resolution (~1/B)
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
        oversample (int): grid samples per resolution cell (1/B)
        time_bandwidth (float): time-bandwidth product B*T of the chirp
    Returns:
        window (callable): window(x) -> complex tensor
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    T = time_bandwidth / B                   # pulse duration (range units)
    K = B / T                                # chirp rate, so that B = K*T
    n = int(round(T / dz)) + 1
    z = (np.arange(n) - (n - 1) / 2.0) * dz  # centered support, length n
    x = np.exp(1j * np.pi * K * z ** 2)      # inst. freq K*z spans [-B/2, B/2]
    return _matched_filter_window(x, dz, device, dtype)


def make_barker13_window(spatial_bw, device, dtype, oversample=32):
    """
    Precompute the matched-filter window for a Barker-13 phase-coded transmit
    waveform, with each of the 13 chips a rectangular sub-pulse of width 1/B
    (bandwidth ~ B = spatial_bw).  The matched-filter output has a mainlobe of width
    ~1/B and uniform range sidelobes at -22.3 dB (1/13 of the peak).

    Inputs:
        spatial_bw (float): waveform bandwidth B; sets range resolution (~1/B)
        device: torch device of the scatter tensors
        dtype: real torch dtype of the scatter tensors
        oversample (int): grid samples per chip / resolution cell (1/B)
    Returns:
        window (callable): window(x) -> complex tensor
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    x = np.repeat(np.array(BARKER13, dtype=np.float64), oversample).astype(np.complex128)
    return _matched_filter_window(x, dz, device, dtype)


def make_transmit_waveform(waveform, spatial_bw, oversample=32, time_bandwidth=100.0):
    """
    Build the complex baseband transmit waveform x(z) for a given waveform name, sampled on
    a common fine range grid.  This exposes the pulse whose matched-filter
    autocorrelation is used as the window in interpolate_signal, so callers can
    characterize the transmitted pulse alongside its compressed matched-filter output.

    The 'lfm' and 'barker13' branches mirror the pulses built inside make_lfm_window
    and make_barker13_window.  'sinc', 'hamming' and 'gaussian' have no separate
    transmit pulse in the simulator (their windows are defined directly in the range
    domain), so the window itself is returned as the waveform.  All five are placed
    on the same grid z in [-T/2, T/2], dz = 1/(B*oversample), T = time_bandwidth/B,
    so their spectra are directly comparable.

    Inputs:
        waveform (str): 'sinc', 'hamming', 'gaussian', 'lfm' or 'barker13'
        spatial_bw (float): waveform bandwidth B
        oversample (int): grid samples per resolution cell (1/B)
        time_bandwidth (float): time-bandwidth product B*T; sets the grid span T
    Returns:
        x (np.ndarray): complex transmit waveform on the grid, unit peak magnitude
        z (np.ndarray): range grid the waveform is sampled on (centered on 0)
        dz (float): grid spacing (so the sampling rate is 1/dz)
    """
    B = float(spatial_bw)
    dz = 1.0 / (B * oversample)              # fine grid spacing in range units
    T = time_bandwidth / B                   # common grid span (= LFM pulse duration)
    n = int(round(T / dz)) + 1
    z = (np.arange(n) - (n - 1) / 2.0) * dz  # centered support, length n

    if waveform == "sinc":
        x = np.sinc(B * z).astype(np.complex128)
    elif waveform == "hamming":
        x = (HAMMING_A0 * np.sinc(B * z) + HAMMING_A1 * np.sinc(B * z - 1)
             + HAMMING_A1 * np.sinc(B * z + 1)).astype(np.complex128) / HAMMING_A0
    elif waveform == "gaussian":
        sigma_x = math.sqrt(math.log(2.0)) / (math.pi * B)
        x = np.exp(-0.5 * (z / sigma_x) ** 2).astype(np.complex128)
    elif waveform == "lfm":
        K = B / T                            # chirp rate, B = K*T (see make_lfm_window)
        x = np.exp(1j * np.pi * K * z ** 2)  # inst. freq K*z spans [-B/2, B/2]
    elif waveform == "barker13":
        chips = np.repeat(np.array(BARKER13, dtype=np.float64), oversample)  # 13/B wide
        x = np.zeros(n, dtype=np.complex128)
        start = (n - len(chips)) // 2        # center the 13-chip code in the grid
        x[start:start + len(chips)] = chips
    else:
        raise ValueError("waveform should be 'sinc', 'hamming', 'gaussian', 'lfm' or 'barker13', but got %s" % waveform)

    x = x / (np.max(np.abs(x)) + 1e-30)      # unit peak magnitude
    return x, z, dz


def interpolate_signal(scatter_z, scatter_e,
        region_radius, sensor_distance,
        spatial_bw = 20, spatial_fs = 20,
        batch_size = None, waveform = 'sinc',
):
    """
    Simulates the received signal for the SAR algorithm given energy-range scatter.
    z is range. Z is the number of samples in the output signal.

    Inputs:
        scatter_z (...,R): the range of each scatter point
        scatter_e (...,R): the energy of each scatter point
        region_radius (float): the radius of the region to consider for output signal samples
        sensor_distance (...,): the distance from the sensor to the origin for each pulse
        spatial_bw (float): spatial bandwidth of the radar
        spatial_fs (float): spatial sampling frequency of the radar
        batch_size (int): number of signals to process in a batch, None means no batching
        waveform (str): transmit waveform to use ('sinc', 'hamming', 'gaussian',
            'lfm' or 'barker13'). 'sinc' is the brick-wall band and 'hamming' the same
            band with a raised-cosine taper rolled onto its edges, which drops the
            first range sidelobe from -13.3 dB to -41.4 dB and the far tail from 1/x
            to ~1/x^3, at the cost of a 1.45x wider mainlobe; both occupy exactly
            [-spatial_bw/2, spatial_bw/2], unlike 'gaussian', whose -3 dB bandwidth is
            spatial_bw but whose skirts run well past it. 'lfm' and 'barker13' are
            pulse-compression matched-filter windows (autocorrelation of the transmit
            waveform)


    Returns:
        signal (tensor): simulated received signal .shape=(..., Z)
    """

    device = scatter_z.device

    # reshaping
    R = scatter_z.shape[-1]  # number of scatter points
    shape_prefix = scatter_z.shape[:-1]  # shape before the last dimension
    N = np.prod(shape_prefix)
    assert(len(scatter_z.shape) > 1), "scatter_z should have at least 2 dimensions, but got %d" % len(scatter_z.shape)
    assert(scatter_z.shape == scatter_e.shape), "scatter_z and scatter_e should have the same shape, but got %s and %s" % (scatter_z.shape, scatter_e.shape)

    # build the effective range window this waveform gives interpolate_signal
    if waveform == "sinc":
        window = lambda x: torch.sinc(spatial_bw * x)
    elif waveform == "hamming":
        # inverse FT of a Hamming-tapered [-bw/2, bw/2] band, scaled to unit peak
        window = lambda x: (HAMMING_A0 * torch.sinc(spatial_bw * x)
                            + HAMMING_A1 * torch.sinc(spatial_bw * x - 1)
                            + HAMMING_A1 * torch.sinc(spatial_bw * x + 1)) / HAMMING_A0
    elif waveform == "gaussian":
        # Gaussian pulse whose half-power (-3 dB) two-sided bandwidth equals
        # spatial_bw, matching the sinc window's occupied band [-bw/2, bw/2].
        # Frequency-domain std: |G(f)|^2 = 1/2 at f = bw/2  =>  sigma_f =
        # (bw/2)/sqrt(ln2), so the spatial-domain std is
        # sigma_x = 1/(2*pi*sigma_f) = sqrt(ln2)/(pi*bw).
        sigma_x = math.sqrt(math.log(2.0)) / (math.pi * spatial_bw)
        window = lambda x: torch.exp(-0.5 * (x / sigma_x) ** 2)
    elif waveform == "lfm":
        window = make_lfm_window(spatial_bw, device, scatter_z.dtype)
    elif waveform == "barker13":
        window = make_barker13_window(spatial_bw, device, scatter_z.dtype)
    else:
        raise ValueError("waveform should be 'sinc', 'hamming', 'gaussian', 'lfm' or 'barker13', but got %s" % waveform)


    # the center of each spatial sample, a window of region_radius either side of sensor_distance
    Z = int(2*region_radius*spatial_fs) + 1
    sample_z = (torch.linspace(0,1, Z, device=device, dtype=scatter_z.dtype) - 0.5 ) * (Z-1)/spatial_fs # (Z,)
    sample_z = sensor_distance.reshape(N,1) + sample_z # (N, 1) + (Z,) -> (N, Z)

    # calculate the received signal for each pulse
    # signal = sum_over_R_scatters( scatter_e * torch.sinc( spatial_bw * (scatter_z - sample_z) ) )
    # When we do the broadcasting we want the shape to be (N,R,Z) before the sum, then sum over the R dimension
    if batch_size is None:
        signal = torch.sum(
            scatter_e.reshape(N,R,1) * \
            window(scatter_z.reshape(N,R,1) - sample_z.reshape(N,1,Z)),
            dim=1
        ) # (N, Z)

    # calculate the received signal in batches to save memory
    else:
        signal = []

        reshaped_scatter_e = scatter_e.reshape(N,R,1)
        reshaped_scatter_z = scatter_z.reshape(N,R,1)
        reshaped_sample_z  = sample_z.reshape(N,1,Z)

        start = 0
        while start < N:
            end = min(start + batch_size, N)
            signal.append(
                torch.sum( reshaped_scatter_e[start:end] * \
                           window(reshaped_scatter_z[start:end] - reshaped_sample_z),
                           dim=2
            ))  # (N', Z)
            start = end

        signal = torch.cat(signal, dim=1) # (N, Z)

    # return stuff
    signal = signal.reshape(*shape_prefix, Z)  # (..., Z)
    sample_z = sample_z.reshape(*shape_prefix, Z)  # (..., Z)
    return signal, sample_z


def apply_snr(signal, snr_db, dim=-1, rng=None):
    """
    Apply a signal-to-noise ratio (SNR) to the input signal.

    Args:
        signal (torch.Tensor): The input signal tensor.
        snr_db (float): The desired SNR in decibels.
        dim (int): The dimension along which to compute the SNR.
        rng (np.random.Generator): noise source; None draws from the global np.random

    Returns:
        torch.Tensor: The signal with the applied SNR.
    """

    # N = Ni + Nr # real and imaginary parts are iid
    # SNR = Ps/Pn = E[|S|^2]/E[|N|^2]
    # E[|N|^2] = E[|S|^2]/SNR

    # E[|N|^2] = E[Nr^2 + Ni^2] = 2 * E[Nr^2]
    # E[|N|^2]/2 = E[Nr^2] = sigma^2
    # sigma = sqrt( E[|N|^2]/2 ) = sqrt( E[|S|^2]/(2*SNR) )

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute the signal power
    signal_power = torch.mean(signal.abs() ** 2, dim=dim, keepdim=True)
    noise_std_dev = torch.sqrt(signal_power / (2 * snr_linear)).cpu().numpy()

    # Generate noise
    normal = np.random.normal if rng is None else rng.normal
    noise = torch.tensor(
             normal(scale=noise_std_dev, size=signal.shape) + \
        1j * normal(scale=noise_std_dev, size=signal.shape)
    ).to(signal.device, dtype=signal.dtype)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal
