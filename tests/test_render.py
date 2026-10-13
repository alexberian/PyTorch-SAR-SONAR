'''Small renders of every modality through the public API and the CLI.'''
import numpy as np
import pytest
import torch

import sarsonar
from sarsonar.cli import main
from conftest import SPHERE

SMALL = {
    'sar': dict(num_pulses=8, n_ray_width=32, n_ray_height=32, image_width=32, image_height=32),
    'sss': dict(num_pings=16, num_ray_height=64, image_width=32, image_height=32),
    'fls': dict(num_pings=32, num_range_values=32, num_ray_width=48, num_ray_height=64),
}
RENDER = {'sar': sarsonar.render_sar, 'sss': sarsonar.render_side_scan, 'fls': sarsonar.render_fls}


@pytest.fixture(scope='module')
def mesh(device):
    return sarsonar.load_mesh(SPHERE, device=device)


@pytest.mark.parametrize('modality', sorted(RENDER))
def test_render(mesh, modality):
    image = RENDER[modality](mesh, azimuth_deg=20.0, elevation_deg=40.0, **SMALL[modality])
    assert image.shape == (32, 32)
    assert torch.isfinite(image).all()
    assert image.abs().max() > 0
    gray = sarsonar.to_uint8(image)
    assert gray.dtype == np.uint8 and gray.max() == 255


def test_render_from_path_is_deterministic(device):
    a = sarsonar.render_fls(SPHERE, device=device, **SMALL['fls'])
    b = sarsonar.render_fls(SPHERE, device=device, **SMALL['fls'])
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cli(tmp_path, device):
    out, npy = tmp_path / 'fls.png', tmp_path / 'fls.npy'
    main([SPHERE, '-m', 'fls', '-o', str(out), '--npy', str(npy), '--device', device,
          '--set', 'num_pings=16', '--set', 'num_range_values=32',
          '--set', 'num_ray_width=48', '--set', 'num_ray_height=64'])
    from PIL import Image
    assert Image.open(out).size == (16, 32)
    assert np.load(npy).shape == (32, 16)
