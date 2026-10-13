'''
pytorch-sar-sonar: ray traced synthetic aperture radar, side scan sonar and forward looking sonar
images of triangle meshes, in PyTorch.

    import sarsonar
    mesh = sarsonar.load_mesh('car.obj', device='cuda')
    image = sarsonar.render_fls(mesh, azimuth_deg=30, elevation_deg=40)
    sarsonar.save_png(image, 'fls.png')
'''
from .display import compress, save_png, to_uint8
from .fls import fls_image, render_fls
from .geometry import cartesian_to_spherical, pose_matrix, spherical_to_cartesian
from .mesh import Mesh, load_mesh, make_ground, read_obj
from .raytrace import build_octree, ray_trace
from .sar import render_sar, sar_image
from .sss import render_side_scan, side_scan_image

__version__ = '0.1.0'

__all__ = [
    'Mesh', 'load_mesh', 'read_obj', 'make_ground',
    'render_sar', 'render_side_scan', 'render_fls',
    'sar_image', 'side_scan_image', 'fls_image',
    'pose_matrix', 'spherical_to_cartesian', 'cartesian_to_spherical',
    'build_octree', 'ray_trace',
    'compress', 'to_uint8', 'save_png',
]
