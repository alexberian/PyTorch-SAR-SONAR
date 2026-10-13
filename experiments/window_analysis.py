"""
Figure "effective windows": the range impulse response and spectrum of each transmit waveform,
written to figures/window_analysis_bw10_fs2000.png.

    python window_analysis.py
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from sarsonar.display import db_compress
from sarsonar.signals import interpolate_signal

from common import FIGURES_DIR


def analyze_waveforms(
        waveforms=('sinc', 'hamming', 'gaussian', 'lfm', 'barker13'),
        bw_list=(20.0,),
        fs_list=(40.0, 80.0),
        region_radius=1.0,
        db_floor=-60.0,
        device='cpu',
        save_dir=None,
):
    """
    Characterize the range impulse response (point-spread function) of each
    interpolate_signal waveform.  A single unit scatter at z=0 is fed to
    interpolate_signal with sensor_distance=0, so the recovered signal is exactly
    the effective window w(z) sampled at the radar sampling frequency:
        s(z_o) = sum_r E_r w(-z_o - z_r/2) = w(-z_o)   (E=1, z=0).
    For each (bw, fs) pair this plots the impulse response in the time/range domain
    (linear and dB, to expose mainlobe width and range sidelobes) and its spectrum
    in the frequency domain (dB, to expose the spectral support and any aliasing
    when fs < bw).  Figures are written to save_dir for use in the paper.

    Inputs:
        waveforms (iterable of str): transmit waveforms to compare
        bw_list (iterable of float): spatial bandwidths to sweep
        fs_list (iterable of float): spatial sampling frequencies to sweep
        region_radius (float): radius of the sampled range region
        db_floor (float): lower dB limit for the log-scale plots
        device (str): torch device to run on
        save_dir (str): directory to write the figures to, FIGURES_DIR when None
    """
    save_dir = save_dir or FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)

    # human-readable legend labels for each waveform
    waveform_labels = {
        'sinc': 'Sinc Interpolation',
        'hamming': 'Hamming-Tapered Band',
        'gaussian': 'Gaussian Pulse',
        'lfm': 'LFM Chirp',
        'barker13': 'Barker 13',
    }

    # one unit scatter at the origin: shape (1, R=1)
    scatter_z = torch.zeros(1, 1, device=device)
    scatter_e = torch.ones(1, 1, device=device, dtype=torch.complex64)
    sensor_distance = torch.zeros(1, device=device)

    def to_db(mag):  # normalized magnitude -> dB, floored for plotting
        return db_compress(mag, float(mag.max()), db_floor)

    for bw in bw_list:
        for fs in fs_list:
            fig, (ax_lin, ax_freq) = plt.subplots(1, 2, figsize=(12, 5))

            for wf in waveforms:
                signal, sample_z = interpolate_signal(
                    scatter_z, scatter_e, region_radius, sensor_distance,
                    spatial_bw=bw, spatial_fs=fs, waveform=wf,
                )
                s = signal[0].cpu().numpy()           # (Z,) complex impulse response
                z = sample_z[0].cpu().numpy()         # (Z,) range axis (sensor_distance=0)
                mag = np.abs(s)
                label = waveform_labels.get(wf, wf)

                # time/range domain
                ax_lin.plot(z, mag / (mag.max() + 1e-30), label=label, ms=3)

                # frequency domain: spectrum of the sampled impulse response
                Z = len(s)
                spec = np.fft.fftshift(np.fft.fft(s))
                freq = np.fft.fftshift(np.fft.fftfreq(Z, d=1.0 / fs))  # [-fs/2, fs/2)
                ax_freq.plot(freq, to_db(np.abs(spec)), label=label, ms=3)

            ax_lin.set_title('Range impulse response (linear)')
            ax_lin.set_xlabel('range z'); ax_lin.set_ylabel('|s| (normalized)')
            ax_lin.grid(True, alpha=0.3); ax_lin.legend()

            # mark the bandwidth edges +-bw/2 to show spectral support
            ax_freq.axvline(+bw / 2, color='k', ls='--', lw=0.8, alpha=0.5, label='+-bw/2')
            ax_freq.axvline(-bw / 2, color='k', ls='--', lw=0.8, alpha=0.5)
            ax_freq.set_title('Spectrum (dB),  dashed = +-bw/2')
            ax_freq.set_xlabel('spatial frequency'); ax_freq.set_ylabel('|S| (dB)')
            ax_freq.set_ylim(db_floor, 3); ax_freq.grid(True, alpha=0.3); ax_freq.legend()
            ax_freq.set_xlim(-bw , +bw)

            path = os.path.join(save_dir, 'window_analysis_bw%g_fs%g.png' % (bw, fs))
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            print('saved %s' % path)


if __name__ == '__main__':
    # B_s = 10 sampled finely enough that each window is drawn as a curve
    analyze_waveforms(bw_list=(10.0,), fs_list=(2000.0,))
