"""
Compute time of spotlight-mode SAR rendering, for the paper's Compute Time subsection.

Times one SAR image (CBP, the paper's spotlight mode) per setting over
    mesh     the smallest and the largest srn cars_train mesh, by face count
    rays     three ray grids R_w = R_h
    device   GPU vs CPU
at 30 pulses, with every other parameter at config.json's sar_baseline. Each image is split into
the stages sarsonar.sar_image runs: ray tracing (accumulate_scatters, both bounces),
interposummation (interpolate_signal, summed over pulses) and CBP. Loading the mesh and building
its octree are timed separately, since a caller rendering many poses of one object does them once.

The first render of each (mesh, device) is a warm-up and is not counted (CUDA context, kernel
caches). Reported times are the median over the timed repeats.

Writes figures/compute_time_experiment.json, rewritten after every row so an overrun of the
time budget keeps what was measured, and prints the paper's table. The two meshes are
cars_train objects, so this needs the srncars dataset and the ShapeNet meshes.

Run:
    python compute_time_experiment.py            # full run, under an hour
    python compute_time_experiment.py -quick     # smallest rays and 2 pulses, to check it runs
"""
import argparse
import json
import platform
import statistics
import time

import torch

import sarsonar
import sarsonar.sar
from sarsonar.raytrace import build_octree

from common import SAR_BASELINE, baseline_mesh, figures_path, load_pose, mesh_path, nearest_pose_num, \
    pose_az_el, sar_kwargs


SAR = dict(SAR_BASELINE)

# smallest and largest cars_train meshes by face count (before the ground plane is added)
MESHES = {
    'small': '3bd66fc2782bd2019766e05e7d6c9088',   # 12,762 faces
    'large': '2c3e7991d4b900ab35fea498c4ba7c5a',   # 1,323,584 faces
}
# per side. CPU time grows ~13x per 4x rays, so the baseline's 208 would put the large mesh's CPU
# run alone over the hour
N_RAYS = (32, 64, 128)
NUM_PULSE = 30
REPEATS = {'cuda': 3, 'cpu': 1}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class StageTimer:
    """Wraps sar_image's stage functions so each call's wall time is summed per stage."""

    STAGES = {'ray_trace': 'accumulate_scatters',
              'interposummation': 'interpolate_signal',
              'imaging': 'projected_CBP'}

    def __init__(self, device):
        self.device = device
        self.totals = {}
        self.originals = {}

    def __enter__(self):
        for stage, fn_name in self.STAGES.items():
            fn = getattr(sarsonar.sar, fn_name)
            self.originals[fn_name] = fn
            setattr(sarsonar.sar, fn_name, self._wrap(stage, fn))
        return self

    def __exit__(self, *exc):
        for fn_name, fn in self.originals.items():
            setattr(sarsonar.sar, fn_name, fn)

    def _wrap(self, stage, fn):
        def timed(*args, **kwargs):
            _sync(self.device)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _sync(self.device)
            self.totals[stage] = self.totals.get(stage, 0.0) + time.perf_counter() - t0
            return out
        return timed


def load_scene(obj_id, device):
    """Mesh, pose and octree for one object, each step timed."""
    pose_num = nearest_pose_num(obj_id, *pose_az_el(SAR['obj_id'], SAR['pose_num']))
    pose = load_pose(obj_id, pose_num, device)

    _sync(device)
    t0 = time.perf_counter()
    mesh = baseline_mesh(SAR, mesh_path(obj_id), device=device)
    _sync(device)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    octree = build_octree(mesh)
    _sync(device)
    t_octree = time.perf_counter() - t0

    return pose, mesh, octree, dict(n_faces=mesh.num_faces, t_load=t_load, t_octree=t_octree)


def render_once(pose, mesh, octree, n_ray, num_pulse, device):
    kwargs = dict(sar_kwargs(SAR), n_ray_width=n_ray, n_ray_height=n_ray, num_pulses=num_pulse)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with StageTimer(device) as timer:
        _sync(device)
        t0 = time.perf_counter()
        sarsonar.sar_image(mesh, pose, octree=octree, **kwargs)
        _sync(device)
        total = time.perf_counter() - t0
    out = dict(total=total, **timer.totals)
    if device.type == 'cuda':
        out['peak_mem_gb'] = torch.cuda.max_memory_allocated(device) / 1024**3
    return out


def save(results, num_pulse, out_path, t_start):
    meta = dict(
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu=platform.processor() or platform.machine(),
        cpu_threads=torch.get_num_threads(),
        torch=torch.__version__,
        num_pulse=num_pulse,
        sar_baseline=SAR,
        sarsonar=sarsonar.__version__,
        wall_time_s=time.perf_counter() - t_start,
    )
    with open(out_path, 'w') as f:
        json.dump(dict(meta=meta, results=results), f, indent=1)
    return meta


def run(n_rays, num_pulse, devices, repeats, out_path, budget_s):
    t_start = time.perf_counter()
    results = []
    for device_name in devices:
        device = torch.device(device_name)
        for size, obj_id in MESHES.items():
            pose, mesh, octree, info = load_scene(obj_id, device)
            print('\n[%s] %s mesh %s: %d faces, load %.2fs, octree %.2fs'
                  % (device_name, size, obj_id, info['n_faces'], info['t_load'], info['t_octree']))

            # warm-up on the smallest ray grid, not counted
            render_once(pose, mesh, octree, n_rays[0], num_pulse, device)

            for n_ray in n_rays:
                runs = [render_once(pose, mesh, octree, n_ray, num_pulse, device)
                        for _ in range(repeats[device_name])]
                med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
                row = dict(device=device_name, mesh=size, obj_id=obj_id, n_ray=n_ray,
                           num_pulse=num_pulse, repeats=len(runs), all_totals=[r['total'] for r in runs],
                           **info, **med)
                results.append(row)
                print('  rays %4d^2: total %8.2fs  ray trace %8.2fs  interposummation %6.2fs  '
                      'CBP %6.2fs%s'
                      % (n_ray, med['total'], med['ray_trace'], med['interposummation'],
                         med['imaging'],
                         '  peak %.2f GB' % med['peak_mem_gb'] if 'peak_mem_gb' in med else ''))

                save(results, num_pulse, out_path, t_start)
                elapsed = time.perf_counter() - t_start
                if elapsed > budget_s:
                    raise RuntimeError('over the %.0fs budget after %.0fs' % (budget_s, elapsed))
            del mesh, octree
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    meta = save(results, num_pulse, out_path, t_start)
    print('\nwall time %.1f min, wrote %s' % (meta['wall_time_s'] / 60, out_path))
    return results


def print_latex_table(results):
    """Rows for the paper's table: mesh, rays, then CPU and GPU totals and the speedup."""
    by_key = {(r['mesh'], r['n_ray'], r['device']): r for r in results}
    print('\n% mesh & rays & CPU (s) & GPU (s) & speedup & GPU ray trace / interposummation / CBP (s)')
    for size in MESHES:
        for n_ray in sorted({r['n_ray'] for r in results}):
            cpu, gpu = by_key.get((size, n_ray, 'cpu')), by_key.get((size, n_ray, 'cuda'))
            if cpu is None or gpu is None:
                continue
            print('%s & $%d^2$ & %.1f & %.2f & %.0f$\\times$ & %.2f / %.2f / %.2f \\\\'
                  % (size, n_ray, cpu['total'], gpu['total'], cpu['total'] / gpu['total'],
                     gpu['ray_trace'], gpu['interposummation'], gpu['imaging']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-quick', action='store_true', help='smallest ray grid and 2 pulses only')
    parser.add_argument('-devices', nargs='+', default=['cuda', 'cpu'])
    parser.add_argument('-cpu_threads', type=int, default=None, help='torch CPU threads, default torch\'s')
    parser.add_argument('-budget_min', type=float, default=60.0)
    parser.add_argument('-out', default=figures_path('compute_time_experiment.json'))
    args = parser.parse_args()

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if args.quick:
        results = run(N_RAYS[:1], 2, args.devices, {'cuda': 1, 'cpu': 1},
                      args.out.replace('.json', '_quick.json'), args.budget_min * 60)
    else:
        results = run(N_RAYS, NUM_PULSE, args.devices, REPEATS, args.out, args.budget_min * 60)
    print_latex_table(results)
