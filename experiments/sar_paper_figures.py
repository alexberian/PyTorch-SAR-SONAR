'''
The paper's spotlight SAR parameter sweeps: one stitched figure each, figures/sar_stitched_<name>.png.

    sphere_size            Figure "sphere size": a sphere shrinking against the resolution cell
    az_spread              Appendix: angular extent of the aperture
    num_pulse              Appendix: pulses per aperture
    snrdb                  Appendix: receiver noise
    trajectory_type        Appendix: linear vs circular aperture
    trajectory_noise_var   Appendix: sensor position error along the track

Every panel is config.json's sar_baseline with the swept keys changed, seeded alike so the noise
is the same draw.

    python sar_paper_figures.py                       # every sweep
    python sar_paper_figures.py az_spread snrdb       # just these
    python sar_paper_figures.py -replot               # restitch the saved panels
'''
import numpy as np

from common import SPHERE_OBJ, sweep_main


def _sweeps():
    # angular extent of the aperture
    az_vals = np.linspace(0, 360, 5).tolist()
    az_spread = dict(
        vary={'azimuth_spread': az_vals},
        overrides={'trajectory_type': 'circular'},  # a linear track can't fly 180 deg or more
        titles=['Azimuth spread: %.0f deg' % a for a in az_vals],
    )

    # along-track sampling density
    pulse_vals = [2 ** k for k in range(3, 11)]  # 8 .. 1024
    num_pulse = dict(
        vary={'num_pulse': pulse_vals},
        titles=['Pulses: %d' % p for p in pulse_vals],
    )

    # additive receiver noise
    snr_db_vals = np.linspace(0, 22, 5).tolist()
    snrdb = dict(
        vary={'snr_db': snr_db_vals},
        titles=['SNR: %.1f dB' % s for s in snr_db_vals],
    )

    # linear vs circular at matched spreads, then the full circle only a circular track can fly
    trajectory_types = ['linear', 'circular', 'linear', 'circular', 'circular']
    trajectory_spreads = [45, 45, 135, 135, 360]
    trajectory_type = dict(
        vary={'trajectory_type': trajectory_types, 'azimuth_spread': trajectory_spreads},
        titles=['%s, %d deg' % (t.capitalize(), a) for t, a in zip(trajectory_types, trajectory_spreads)],
    )

    # sensor position error the imager doesn't know about
    noise_vals = [0] + (10 ** np.linspace(-4, -2, 4, endpoint=True)).tolist()
    trajectory_noise_var = dict(
        vary={'trajectory_noise_var': noise_vals},
        titles=['Turbulence: %.2e' % v for v in noise_vals],
    )

    # a sphere, from the diagonal of a ShapeNet car down to well under a resolution cell
    scale_vals = [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]
    sphere_size = dict(
        vary={'mesh_scale': scale_vals},
        overrides={'override_obj_path': SPHERE_OBJ, 'make_ground': False},
        titles=['Scale: 1', 'Scale: 1/2', 'Scale: 1/4', 'Scale: 1/8', 'Scale: 1/16'],
    )

    return dict(sphere_size=sphere_size, az_spread=az_spread, num_pulse=num_pulse, snrdb=snrdb,
                trajectory_type=trajectory_type, trajectory_noise_var=trajectory_noise_var)


SWEEPS = _sweeps()

if __name__ == '__main__':
    sweep_main('sar', SWEEPS, __doc__.strip().split('\n\n')[0])
