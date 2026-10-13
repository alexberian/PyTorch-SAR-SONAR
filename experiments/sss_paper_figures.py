'''
The paper's side scan sonar sweeps: one stitched figure each, figures/side_scan_stitched_<name>.png.

    num_ray_width   Figure "side-scan azimuth rays": azimuth rays per ping under a 4 deg beam
    beam_width      Appendix "Sonar Beam Width": the along-track beam

Every panel is config.json's side_scan_sonar_baseline with the swept keys changed.

    python sss_paper_figures.py
    python sss_paper_figures.py beam_width
    python sss_paper_figures.py -replot
'''
import numpy as np

from common import sweep_main


def _sweeps():
    # the along-track beam, logarithmic since a degree is a big step at 0.1 and a small one at 10
    beam_width_vals = np.logspace(-1, 1, 5).tolist()
    beam_width = dict(
        vary={'azimuth_beam_width_deg': beam_width_vals},
        overrides={'num_ray_width': 250},  # the baseline's one ray leaves the beam nothing to weight
        titles=['Beam Width: %.2f deg' % b for b in beam_width_vals],
    )

    # azimuth rays at a fixed 4 deg beam: each ray images the scene once, shifted along track by
    # range * its angle, so too few rays repeat the scene rather than smear it. Odd counts keep one
    # ray on boresight; at range 7.5 the copies are 21, 7, 2.6, 0.9, 0.3 pixels apart
    ray_width_beam_deg = 4.0
    ray_width_vals = [9, 25, 65, 181, 501]  # 181 is the ~182 ray bound for a smooth image at 4 deg
    num_ray_width = dict(
        vary={'num_ray_width': ray_width_vals},
        overrides={'azimuth_beam_width_deg': ray_width_beam_deg},
        titles=['%d Az Rays, %.0f deg Beam' % (n, ray_width_beam_deg) for n in ray_width_vals],
    )

    return dict(num_ray_width=num_ray_width, beam_width=beam_width)


SWEEPS = _sweeps()

if __name__ == '__main__':
    sweep_main('sss', SWEEPS, __doc__.strip().split('\n\n')[0])
