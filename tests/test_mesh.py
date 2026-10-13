import numpy as np
import torch

from sarsonar import load_mesh, read_obj
from conftest import SPHERE


def test_fan_triangulation_and_negative_indices(tmp_path):
    path = tmp_path / 'quad.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n'
                    'f 1/1/1 2/2/1 3/3/1 4/4/1\n'
                    'f -4 -3 -1\n')
    verts, faces = read_obj(str(path))
    assert verts.shape == (4, 3) and verts.dtype == np.float32
    # a quad fans into (0,1,2), (0,2,3); negative indices count back from the last vertex
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3], [0, 1, 3]])


def test_sphere_matches_pytorch3d_counts():
    # pytorch3d's load_obj gives 482 vertices and 960 triangles for this file
    verts, faces = read_obj(SPHERE)
    assert verts.shape == (482, 3)
    assert faces.shape == (960, 3)


def test_load_mesh_normalizes_and_levels():
    mesh = load_mesh(SPHERE, ground=False)
    lo, hi = mesh.verts.min(0).values, mesh.verts.max(0).values
    assert torch.isclose((hi - lo).norm(), torch.tensor(1.0), atol=1e-5)
    assert abs(float(lo[2])) < 1e-6
    torch.testing.assert_close((lo + hi)[:2] / 2, torch.zeros(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(mesh.normals.norm(dim=1), torch.ones(mesh.num_faces))
    torch.testing.assert_close(mesh.materials[:, [0, 1, 4]].sum(1), torch.ones(mesh.num_faces))


def test_ground_plane():
    mesh = load_mesh(SPHERE, ground=True)
    bare = load_mesh(SPHERE, ground=False)
    assert mesh.num_faces == bare.num_faces + 2 * 100 * 100
    # every ground triangle faces up
    assert torch.all(mesh.normals[bare.num_faces:, 2] > 0.999)


def test_per_triangle_materials(tmp_path):
    verts, faces = read_obj(SPHERE)
    obj = tmp_path / 'sphere.obj'
    obj.write_text(open(SPHERE).read())
    raids = np.tile([0.5, 0.2, 0.7, 0.3, 0.3], (len(faces), 1))
    np.savetxt(tmp_path / 'sphere.mp', raids)
    mesh = load_mesh(str(obj), ground=False)
    torch.testing.assert_close(mesh.materials, torch.tensor(raids, dtype=torch.float32))
