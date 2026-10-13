'''
Coordinate conventions. The world is z-up with the object at the origin; azimuth is measured in
the x-y plane from +x toward +y and elevation up from that plane, both in degrees.

Poses are 4x4 camera-to-world matrices in the srn_cars layout: columns are (right, up, forward,
center), with forward pointing from the sensor toward the origin.
'''
import numpy as np
import torch


def dot_product(a, b, dim=-1, keepdim=False):
    '''Dot product of a and b along dim.'''
    return torch.sum(a * b, dim=dim, keepdim=keepdim)


def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    '''
    inputs:
        azimuth_deg, elevation_deg, distance (...,): tensors
    outputs:
        cartesian (...,3)
    '''
    azimuth_rad = azimuth_deg * np.pi / 180
    elevation_rad = elevation_deg * np.pi / 180
    x = distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    z = distance * torch.sin(elevation_rad)
    return torch.stack((x, y, z), dim=-1)


def cartesian_to_spherical(cartesian):
    '''
    inputs:
        cartesian (...,3)
    outputs:
        azimuth_deg, elevation_deg, distance (...,)
    '''
    x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
    distance = torch.sqrt(x**2 + y**2 + z**2)
    azimuth_deg = torch.atan2(y, x) * 180 / np.pi
    elevation_deg = torch.atan2(z, torch.sqrt(x**2 + y**2)) * 180 / np.pi
    return azimuth_deg, elevation_deg, distance


def pose_matrix(azimuth_deg, elevation_deg, distance, device='cpu'):
    '''
    Pose of a sensor at (azimuth, elevation, distance) looking at the origin.

    inputs:
        azimuth_deg, elevation_deg, distance: floats or (...,) tensors
    outputs:
        pose (...,4,4): columns (right, up, forward, center)
    '''
    as_tensor = lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x, device=device, dtype=torch.float32)
    # 3.14159, not pi: keeps renders bit-identical to the paper's
    azimuth = 3.14159 / 180 * as_tensor(azimuth_deg)
    elevation = 3.14159 / 180 * as_tensor(elevation_deg)
    distance = as_tensor(distance)

    u = torch.stack([-torch.sin(azimuth), torch.cos(azimuth), torch.zeros_like(azimuth)], dim=-1)
    v = torch.stack([-torch.cos(azimuth) * torch.sin(elevation),
                     -torch.sin(azimuth) * torch.sin(elevation),
                     torch.cos(elevation)], dim=-1)
    w = -torch.stack([torch.cos(azimuth) * torch.cos(elevation),
                      torch.sin(azimuth) * torch.cos(elevation),
                      torch.sin(elevation)], dim=-1)
    u = torch.nn.functional.normalize(u, dim=-1)
    v = torch.nn.functional.normalize(v, dim=-1)
    w = torch.nn.functional.normalize(w, dim=-1)

    pose = torch.zeros((*w.shape[:-1], 4, 4), device=device)
    pose[..., :3, 0] = u
    pose[..., :3, 1] = v
    pose[..., :3, 2] = w
    pose[..., :3, 3] = -distance * w
    pose[..., 3, 3] = 1.0
    return pose


def extract_pose_info(poses):
    '''
    Sensor position, axes and spherical coordinates of (...,4,4) poses.

    outputs:
        center, right, up, forward (...,3), distance (...,), elevation_deg (...,), azimuth_deg (...,)
    '''
    right = poses[..., :3, 0]
    up = poses[..., :3, 1]
    forward = poses[..., :3, 2]
    center = poses[..., :3, 3]
    distance = torch.norm(center, dim=-1)
    azimuth = torch.where(right[..., 0] < 0, torch.acos(right[..., 1]), 2 * np.pi - torch.acos(right[..., 1]))
    elevation = torch.asin(center[..., 2] / distance)
    return center, right, up, forward, distance, elevation * 180 / np.pi, azimuth * 180 / np.pi


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg, device='cpu'):
    '''(3,3) extrinsic rotation, x first, then y, then z, angles in degrees.'''
    def rad(d):
        return torch.tensor(d, dtype=torch.float32, device=device) * (torch.pi / 180.0)

    cx, sx = torch.cos(rad(rx_deg)), torch.sin(rad(rx_deg))
    cy, sy = torch.cos(rad(ry_deg)), torch.sin(rad(ry_deg))
    cz, sz = torch.cos(rad(rz_deg)), torch.sin(rad(rz_deg))
    one, zero = torch.ones_like(cx), torch.zeros_like(cx)

    Rx = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, cx, -sx]),
                      torch.stack([zero, sx, cx])])
    Ry = torch.stack([torch.stack([cy, zero, sy]),
                      torch.stack([zero, one, zero]),
                      torch.stack([-sy, zero, cy])])
    Rz = torch.stack([torch.stack([cz, -sz, zero]),
                      torch.stack([sz, cz, zero]),
                      torch.stack([zero, zero, one])])
    return Rz @ Ry @ Rx
