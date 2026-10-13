'''
The paper's forward looking sonar sweeps: one stitched figure each, figures/fls_stitched_<name>.png.

    range_density   Figure "sonar ray density across range": one image from the sensor out to
                    twice the car's range, under a fan wide enough to reach nadir
    beam_width      Appendix "Sonar Beam Width": the two-way beam each ping steers

Every panel is config.json's forward_looking_sonar_baseline with the swept keys changed.

    python fls_paper_figures.py
    python fls_paper_figures.py range_density
    python fls_paper_figures.py -replot
'''
import numpy as np

from common import FLS_BASELINE, sweep_main


def _sweeps():
    # the two-way beam each ping steers, which is what resolves azimuth
    beam_width_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        vary={'azimuth_beam_width_deg': beam_width_vals},
        titles=['Beam Width: %.2f deg' % b for b in beam_width_vals],
    )

    # from the sensor out to twice the car's range, with a fan reaching nadir (2*(90-40.6) = 99 deg),
    # so the seafloor rays thin from dense near nadir to sparse far out
    b = FLS_BASELINE
    fs = b['spatial_fs']
    d = b['sensor_distance']
    range_swath = 2 * d - 2 / fs  # a sample in from 0, which the swath assert's float norm rejects
    # azimuth span whose arc at the car's range, span * d, matches the 2d range axis
    span = np.degrees(2.0)
    pings = 512
    # beam and azimuth rays per ping spacing as in the baseline
    base_spacing = b['image_azimuth_range'] / (b['num_pings'] - 1)
    beam = b['azimuth_beam_width_deg'] * (span / (pings - 1)) / base_spacing
    rays_per_deg = b['num_ray_width'] / b['ray_fov_az'] * b['azimuth_beam_width_deg'] / beam
    fov_az = span + 3 * beam  # the paper's margin
    range_density = dict(
        vary={'image_range_swath': [range_swath]},
        overrides={'ray_fov_el': 100.0,
                   'num_ray_height': 1040,
                   'image_azimuth_range': span,
                   'num_pings': pings,
                   'azimuth_beam_width_deg': beam,
                   'ray_fov_az': fov_az,
                   'num_ray_width': int(round(rays_per_deg * fov_az)),
                   'db_floor': -80.0,  # the far seafloor sits ~70 dB under the nadir peak
                   'num_range_values': round(range_swath * fs) + 1},  # a row per signal sample
        titles=[''],  # described in the paper's caption
    )

    return dict(range_density=range_density, beam_width=beam_width)


SWEEPS = _sweeps()

if __name__ == '__main__':
    sweep_main('fls', SWEEPS, __doc__.strip().split('\n\n')[0])
