'''
Writes examples/toy_car.obj: a box body, a box cabin and four cylinder wheels, y-up like most .obj
files. Only a stand-in to render out of the box; ShapeNet models can't be redistributed.

    python scripts/make_toy_car.py
'''
import math
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'examples', 'toy_car.obj')


def box(x0, x1, y0, y1, z0, z1):
    '''Corners and outward-wound quads of an axis-aligned box, z-up.'''
    v = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    q = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    return v, q


def wheel(cx, cy, cz, radius, width, n=24):
    '''A cylinder along y: side quads plus one n-gon cap at each end.'''
    v = []
    for y in (cy - width / 2, cy + width / 2):
        v += [(cx + radius * math.cos(2 * math.pi * k / n), y, cz + radius * math.sin(2 * math.pi * k / n))
              for k in range(n)]
    sides = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    caps = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    return v, sides + caps


def main():
    parts = [box(-2.0, 2.0, -0.9, 0.9, 0.4, 1.2), box(-1.2, 0.8, -0.8, 0.8, 1.2, 1.8)]
    parts += [wheel(x, y, 0.35, 0.35, 0.3) for x in (-1.3, 1.3) for y in (-0.85, 0.85)]

    lines = ['# toy car, y-up, written by scripts/make_toy_car.py']
    faces, offset = [], 0
    for verts, polys in parts:
        # z-up (x, y, z) -> y-up (x, z, -y), which load_mesh's default rotation turns back
        lines += ['v %.6f %.6f %.6f' % (x, z, -y) for x, y, z in verts]
        faces += ['f ' + ' '.join(str(offset + i + 1) for i in p) for p in polys]
        offset += len(verts)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines + faces) + '\n')
    print('wrote %s: %d vertices, %d polygons' % (OUT, offset, len(faces)))


if __name__ == '__main__':
    main()
