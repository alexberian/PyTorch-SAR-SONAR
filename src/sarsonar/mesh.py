'''
Triangle meshes for the ray tracer: an OBJ reader, a ground plane, and per-triangle materials.

A scene is one Mesh: vertices, triangles, their normals and edges, and the r,a,i,d,s material of
every triangle. load_mesh builds one from an .obj file, optionally standing it on a large ground plane.
'''
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import rotation_matrix_xyz

log = logging.getLogger(__name__)


@dataclass
class Mesh:
    '''
    A triangle mesh ready to ray trace.

    attributes:
        verts (V,3): vertex positions, float32
        faces (F,3): vertex indices of each triangle, int64
        normals (F,3): unit normal of each triangle
        materials (F,5): r,a,i,d,s of each triangle (reflectivity, absorption, directional
            scattering, diffuse scattering, overall scattering), normalized so r+a+s = i+d = 1
        edge_1, edge_2 (F,3): v1 - v0 and v2 - v0 of each triangle, cached for Moller-Trumbore
    '''
    verts: torch.Tensor
    faces: torch.Tensor
    normals: torch.Tensor
    materials: torch.Tensor
    edge_1: torch.Tensor
    edge_2: torch.Tensor

    @property
    def device(self):
        return self.verts.device

    @property
    def num_faces(self):
        return self.faces.shape[0]

    def to(self, device):
        return Mesh(*(getattr(self, k).to(device) for k in
                      ('verts', 'faces', 'normals', 'materials', 'edge_1', 'edge_2')))

    @classmethod
    def from_tensors(cls, verts, faces, materials):
        '''Build a Mesh from vertices, triangles and (F,5) materials, deriving normals and edges.'''
        face_verts = verts[faces]  # (F,3,3)
        edge_1 = face_verts[:, 1] - face_verts[:, 0]  # (F,3)
        edge_2 = face_verts[:, 2] - face_verts[:, 0]  # (F,3)
        normals = torch.nn.functional.normalize(torch.cross(edge_1, edge_2, dim=1), dim=1)  # (F,3)
        return cls(verts, faces, normals, correct_material_properties(materials), edge_1, edge_2)


def read_obj(path):
    '''
    Vertices and triangles of a Wavefront .obj file. Everything but v and f lines is ignored.

    Polygons are fan triangulated, (0,i,i+1), in file order -- the same triangles in the same
    order pytorch3d's load_obj gives, so a .mp per-triangle material file lines up with either.

    outputs:
        verts (V,3) float32 ndarray, faces (F,3) int64 ndarray, 0-based
    '''
    verts, faces = [], []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                # 'f 4/1/1 3/2/1 2/1/1': the vertex index is the first field; negatives count back
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                if len(idx) < 3:
                    raise ValueError('%s: face with fewer than 3 vertices: %r' % (path, line))
                faces.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.size and (faces.min() < 0 or faces.max() >= len(verts)):
        raise ValueError('%s: face index out of range for %d vertices' % (path, len(verts)))
    return verts, faces


def correct_material_properties(properties):
    '''
    Normalize (...,5) r,a,i,d,s materials so r+a+s = 1 and i+d = 1.

    The baselines give unnormalized weights, e.g. (1, 1, 100, 0.1, 0.9), so this is the usual path
    rather than a repair.
    '''
    assert properties.shape[-1] == 5, 'Properties should have shape (..., 5)'
    r, a, i, d, s = properties.unbind(-1)
    if not torch.allclose(r + s + a, torch.ones_like(r)):
        total = r + s + a
        r, s, a = r / total, s / total, a / total
    if not torch.allclose(i + d, torch.ones_like(i)):
        total = i + d
        i, d = i / total, d / total
    return torch.stack((r, a, i, d, s), dim=-1)


def make_ground(size, ground_dim=2, ground_level=0.0, max_triangle_len=0.1, device='cpu'):
    '''
    A square ground plane of side `size`, tessellated so no triangle edge exceeds max_triangle_len.

    outputs:
        verts (V,3), faces (F,3), every triangle wound so its normal points along +ground_dim
    '''
    N = int(math.floor(size / max_triangle_len)) + 1
    assert N > 1, 'N should be greater than 1, but got %d' % N

    grid = torch.meshgrid(torch.linspace(-size / 2, size / 2, N, device=device, dtype=torch.float32),
                          torch.linspace(-size / 2, size / 2, N, device=device, dtype=torch.float32),
                          indexing='ij')  # (N,N), (N,N)
    verts = torch.stack(grid, dim=-1)  # (N,N,2)
    verts = torch.cat((verts, torch.full((N, N, 1), ground_level, device=device, dtype=torch.float32)), dim=-1)
    verts = verts.reshape(-1, 3)  # (V,3)

    h, w = torch.meshgrid(torch.arange(N - 1, device=device, dtype=torch.int64),
                          torch.arange(N - 1, device=device, dtype=torch.int64),
                          indexing='ij')  # (N-1,N-1)

    # square corners A=(h,w) B=(h+1,w) C=(h,w+1) D=(h+1,w+1); A,B,C and B,D,C wind the same way
    squares = torch.stack((h * N + w, (h + 1) * N + w, h * N + w + 1, (h + 1) * N + w + 1), dim=-1).reshape(-1, 4)
    faces = torch.cat((squares[..., [0, 1, 2]], squares[..., [1, 3, 2]]), dim=1).reshape(-1, 3)  # (F,3)

    verts = torch.roll(verts, shifts=ground_dim + 1, dims=1)  # put the plane's normal on ground_dim
    return verts, faces


def load_mesh(path,
              rotate_xyz=(90.0, 0.0, 0.0),
              normalize=True,
              scale=None,
              x_flip=False,
              level_with_ground=True,
              ground=True,
              ground_dim=2,
              obj_raids=(1.0, 1.0, 100.0, 0.1, 0.9),
              ground_raids=(1.0, 1.0, 1.0, 5, 0.1),
              material_file=None,
              device='cpu',
              ):
    '''
    Load an .obj into a Mesh, optionally standing it on a ground plane.

    The simulator is z-up. OBJ files are conventionally y-up, so the default rotate_xyz turns +y into
    +z; pass (0, 0, 0) for a file that is already z-up.

    inputs:
        path (str): the .obj file
        rotate_xyz (3,): extrinsic x, y, z rotations in degrees, applied before normalizing
        normalize (bool): center the object's bounding box on the origin in x,y and scale its
            diagonal to 1, the size of a ShapeNet normalized model, which the default sensor
            geometries are framed for
        scale (float): extra uniform scale, applied first
        x_flip (bool): mirror the object along x
        level_with_ground (bool): translate the object so its lowest point sits at 0 along ground_dim
        ground (bool): add a 200 x 200 ground plane at 0 along ground_dim
        obj_raids (5,): r,a,i,d,s of every object triangle, unless there is a .mp file
        ground_raids (5,): r,a,i,d,s of every ground triangle
        material_file (str): .mp file of one "r a i d s" line per object triangle; None uses the
            .obj's sibling .mp if there is one, else obj_raids
        device: where the Mesh lives
    outputs:
        mesh (Mesh)
    '''
    verts, faces = read_obj(path)
    verts = torch.from_numpy(verts).to(device)  # (V,3)
    faces = torch.from_numpy(faces).to(device)  # (F,3)

    if scale is not None:
        verts = verts * scale
    if x_flip:
        verts = verts * torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=verts.dtype)

    rx, ry, rz = rotate_xyz
    if rx != 0.0 or ry != 0.0 or rz != 0.0:
        R = rotation_matrix_xyz(rx, ry, rz, device=device).to(verts.dtype)
        verts = verts @ R.T

    if normalize:
        lo, hi = verts.min(dim=0).values, verts.max(dim=0).values  # (3,)
        center = (lo + hi) / 2
        center[ground_dim] = 0.0  # leveling handles the vertical
        verts = (verts - center) / (hi - lo).norm().clamp_min(1e-12)

    if level_with_ground:
        verts[:, ground_dim] -= verts[:, ground_dim].min()

    # per triangle materials from a .mp file if there is one
    mp_path = os.path.splitext(path)[0] + '.mp' if material_file is None else material_file
    if material_file is not None or os.path.exists(mp_path):
        raids = torch.tensor(np.loadtxt(mp_path, ndmin=2), device=device, dtype=torch.float32)  # (F,5)
        assert raids.shape == (faces.shape[0], 5), \
            '%s: %s rows, mesh has %d triangles' % (mp_path, tuple(raids.shape), faces.shape[0])
        log.info('per-triangle materials from %s', mp_path)
    else:
        raids = torch.tensor(obj_raids, device=device, dtype=torch.float32).reshape(1, 5).repeat(faces.shape[0], 1)

    if ground:
        size = 200
        ground_verts, ground_faces = make_ground(size, ground_dim, ground_level=0,
                                                 max_triangle_len=size / 100.0, device=device)
        faces = torch.cat([faces, ground_faces + verts.shape[0]], dim=0)
        verts = torch.cat([verts, ground_verts], dim=0)
        ground_mat = torch.tensor(ground_raids, device=device, dtype=torch.float32).reshape(1, 5)
        raids = torch.cat([raids, ground_mat.repeat(ground_faces.shape[0], 1)], dim=0)

    log.info('load_mesh: verts=%d faces=%d', verts.shape[0], faces.shape[0])
    return Mesh.from_tensors(verts, faces, raids)


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def as_mesh(mesh, device=None):
    '''A Mesh on device, loaded with load_mesh's defaults when given an .obj path.'''
    if not isinstance(mesh, Mesh):
        return load_mesh(str(mesh), device=device or default_device())
    return mesh if device is None else mesh.to(device)
