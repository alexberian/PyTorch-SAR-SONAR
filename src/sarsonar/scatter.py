'''
One scatter accumulator for arbitrary rays, accumulate_scatters_from_rays, behind the three sensor
geometries the simulator uses: accumulate_scatters (planar wavefront, SAR),
accumulate_scatters_perspective (point sensor aimed at the origin, forward looking sonar) and
accumulate_scatters_side_scan (point sensor with its pose supplied, side scan sonar).
'''
import logging
import time

import numpy as np
import torch

from .geometry import cartesian_to_spherical, dot_product, pose_matrix
from .raytrace import build_octree, ray_trace

log = logging.getLogger(__name__)


def directional_scatter_polynomial_alpha5(cos_90_minus_elevation):
    '''Normalizer that keeps the directional scattering lobe energy conserving at alpha=5.'''
    return 1.46792856*cos_90_minus_elevation**1 + 1.81188202


def ray_trace_oom_safe(ray_origins, ray_directions, mesh,
                       octree=None, batch_size=2**20, min_batch_size=1):
    '''
    Call ray_trace, halving batch_size on CUDA OOM until it fits.

    The octree path builds a padded (K, max_Fl, 3) face matrix whose size scales
    with the ray batch B; on dense scenes this can exceed VRAM. Rather than tune
    batch_size per scene, we catch OutOfMemoryError, free the cache, and retry the
    whole call with a smaller batch. ray_trace allocates fresh output tensors each
    call, so a failed attempt leaves nothing to clean up but the allocator cache.
    '''
    R = ray_origins.shape[0]
    # clamp to R so the first halving is actually effective (callers may pass a
    # huge sentinel batch size like 2**100 == "all rays in one batch")
    bs = min(batch_size, R) if R > 0 else batch_size
    while True:
        try:
            return ray_trace(ray_origins, ray_directions, mesh, octree=octree, batch_size=bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_batch_size:
                raise
            bs = max(min_batch_size, bs // 2)
            log.warning('ray_trace_oom_safe: CUDA OOM, retrying with batch_size=%d', bs)


def centered_linspace(span, n, device, flip=False):
    '''
    n evenly spaced samples covering `span`, centered on 0.

    torch.linspace(-s/2, s/2, 1) returns the *left edge* rather than the center, which would
    aim a single ray at the corner of the fan instead of down boresight; n==1 is special-cased.

    inputs:
        span (float): total extent covered by the samples
        n (int): number of samples
        flip (bool): return the samples in descending order (top->bottom image rows)
    outputs:
        offsets (n,): sample offsets about 0
    '''
    if n == 1:
        return torch.zeros(1, device=device)
    lo, hi = -span / 2, span / 2
    if flip:
        lo, hi = hi, lo
    return torch.linspace(lo, hi, n, device=device)


def points_visible_to_far_sensor(points, to_sensor_directions, mesh,
                                 octree=None, surface_bias=1e-3, batch_size=2**20):
    '''
    Line of sight back to an infinitely far sensor, evaluated per point.

    Rays carry their own sensor, so the direction back to it can vary point to point even
    under the planar-wave assumption. A point is visible if a ray cast from it toward the
    sensor escapes the scene; if it hits a triangle first, the point is occluded.

    Inputs:
        points (N,3): points in space to test
        to_sensor_directions (N,3): unit vector from each point back toward its sensor
        mesh (Mesh): the scene
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each ray origin off its surface along the
            sensor direction, to avoid self-intersection
        batch_size (int): max rays per ray_trace batch

    Outputs:
        visible (N,): boolean tensor, True where the point can see its sensor
    '''
    if octree is None:
        octree = build_octree(mesh)

    # bias origins toward the sensor so the shadow ray doesn't re-hit the surface the
    # point sits on (leg~=0 self-intersection)
    origins = points + surface_bias * to_sensor_directions  # (N, 3)

    _, distance = ray_trace_oom_safe(origins, to_sensor_directions, mesh,
                                     octree=octree, batch_size=batch_size)  # (N,)

    # a hit (distance >= 0) means geometry blocks the path to the sensor
    visible = distance < 0  # (N,)
    return visible


def points_visible_to_point_sensor(points, sensor_positions, mesh,
                                   octree=None, surface_bias=1e-3, batch_size=2**20):
    '''
    Line of sight to a sensor at a finite location, evaluated per point.

    Only geometry *between* the point and the sensor occludes it -- a hit beyond the sensor
    is behind it and blocks nothing.

    Inputs:
        points (N,3): points in space to test
        sensor_positions (N,3): location of the sensor each point reports back to
        mesh (Mesh): the scene
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each ray origin off its surface toward the
            sensor, to avoid self-intersection
        batch_size (int): max rays per ray_trace batch

    Outputs:
        visible (N,): boolean tensor, True where the point can see its sensor
    '''
    if octree is None:
        octree = build_octree(mesh)

    to_sensor = sensor_positions - points                       # (N, 3)
    distance_to_sensor = to_sensor.norm(dim=-1)                 # (N,)
    directions = to_sensor / distance_to_sensor.unsqueeze(-1).clamp_min(1e-12)  # (N, 3)

    # bias origins toward the sensor so the shadow ray doesn't re-hit the surface the
    # point sits on (leg~=0 self-intersection)
    origins = points + surface_bias * directions  # (N, 3)

    _, distance = ray_trace_oom_safe(origins, directions, mesh,
                                     octree=octree, batch_size=batch_size)  # (N,)

    # a hit (distance >= 0) blocks the path only if it lands short of the sensor. Origins were
    # already pushed surface_bias along the way, so that much less of the path remains.
    visible = (distance < 0) | (distance > distance_to_sensor - surface_bias)  # (N,)
    return visible


def accumulate_scatters_from_rays(
        mesh,
        ray_origins,
        ray_directions,
        sensor_positions=None,
        planar_wave=False,
        wavelength=None,
        num_bounce=1,
        attenuation=0.0,
        octree=None,
        second_bounce_batch_size=2**100,
        surface_bias=1e-3,
        first_bounce_maps=False,
        stats=None,
    ):
    '''
    returns the energy and range of every scatter produced by an arbitrary set of rays

    One call renders one set of rays -- one pulse, one ping, one whatever. It knows nothing
    about ray grids, poses, or scene/pulse indexing; a caller that wants those loops over
    this, building the octree once and passing it in.

    inputs:
        mesh (Mesh): the scene, with its face normals and r,a,i,d,s materials
        ray_origins (R,3): where each ray starts
        ray_directions (R,3): unit direction each ray travels
        sensor_positions (R,3) or (3,): where each ray's return is collected. None means the
            sensor is wherever the ray started, which is the case for a fan leaving a point
            sensor; a parallel grid spread across a sensor plane must pass the sensor itself.
        planar_wave (bool): treat the sensor as infinitely far in the direction of its
            position, so the line of sight back to it is -normalize(sensor_position) and is
            fixed per ray. The return leg then becomes a projection onto that line of sight
            rather than a true distance. False treats the sensor as the finite point it is.
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        num_bounce (int): how many times each ray is allowed to reflect
        attenuation (float): absorption coefficient of the medium, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.
        octree: prebuilt Octree for the mesh, or None to build one here
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        first_bounce_maps (bool): also return the per-ray first-bounce depth and energy
        stats (dict): accumulated in place with timing and cos(theta/2) NaN diagnostics, for a
            caller that wants one summary line across many calls

    outputs:
        range (R',): round-trip path length of each surviving scatter, so ~2x the distance to
            the scatter on a first bounce. R' varies with how many rays hit and stayed visible
        energy (R',): energy of each surviving scatter, after attenuation and the wavelength
            phasor
        position (R',3): where each surviving scatter sits in world space. Anything measured
            relative to a sensor pose -- azimuth off boresight, say -- is the caller's job
        ray_index (R',): which transmitted ray produced each scatter, indexing into the R
            input rays. The bounce filtering otherwise loses that link, and without it a
            caller cannot attach a per-ray quantity (its fan azimuth, say) to the scatters
        first_bounce: {'depth': (R,), 'energy': (R,)} over the *transmitted* rays, misses
            getting -1 / 0, or None when first_bounce_maps is False
    '''
    device = ray_origins.device
    R = ray_origins.shape[0]
    face_normals = mesh.normals
    material_properties = mesh.materials

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    if octree is None:
        octree = build_octree(mesh)

    # every ray carries the sensor it reports back to; by default that is where it started
    if sensor_positions is None:
        sensor_positions = ray_origins
    else:
        sensor_positions = sensor_positions.reshape(-1, 3).expand(R, 3)

    # under the planar-wave assumption the sensor sits infinitely far away in the direction of
    # its position, so this is fixed per ray rather than evaluated per scatter
    if planar_wave:
        to_sensor_directions = torch.nn.functional.normalize(sensor_positions, dim=-1)  # (R,3)

    prev_origins    = ray_origins                                                 # (R,3)
    prev_directions = ray_directions                                              # (R,3)
    cumulative_legs = torch.zeros(R, device=device)                               # (R,)
    cumulative_reflectivity = torch.ones(R, device=device)                        # (R,) product of r of prior bounces
    ray_index       = torch.arange(R, device=device)                              # (R,) survives the bounce filtering
    outbound_range  = None  # sensor -> first hit, shared by every later bounce of that ray

    scatter_ranges      = torch.empty(0, device=device)
    scatter_energies    = torch.empty(0, device=device)
    scatter_positions   = torch.empty(0, 3, device=device)
    scatter_ray_indices = torch.empty(0, dtype=torch.long, device=device)
    first_bounce        = None

    t_bounce_totals = [0.0] * num_bounce

    # diagnostics for cos(theta/2): sqrt() of a negative silently yields NaN rather than
    # raising, so track how often it happens and how far below zero the argument gets.
    n_nan_cos    = 0
    n_nan_arg    = 0
    n_cos_total  = 0
    min_sqrt_arg = float('inf')

    # ray-trace all bounces
    for b in range(1, num_bounce + 1):
        t_b_start = sync_time()
        hit_indices, distance = ray_trace_oom_safe(prev_origins, prev_directions, mesh, octree=octree, batch_size=second_bounce_batch_size)

        # bank the ray-trace time now so a bounce that breaks below still reports it
        t_after_trace = sync_time()
        t_bounce_totals[b-1] += t_after_trace - t_b_start

        hit_b = distance >= 0
        # if no rays hit for this bounce, nothing else to do; report time above
        if not hit_b.any():
            break

        # filter state to rays that hit
        prev_origins    = prev_origins[hit_b]
        prev_directions = prev_directions[hit_b]
        distance        = distance[hit_b]
        hit_indices     = hit_indices[hit_b]
        cumulative_legs = cumulative_legs[hit_b]
        cumulative_reflectivity = cumulative_reflectivity[hit_b]
        ray_index       = ray_index[hit_b]
        sensor_positions = sensor_positions[hit_b]
        if planar_wave:
            to_sensor_directions = to_sensor_directions[hit_b]
        if outbound_range is not None:
            outbound_range = outbound_range[hit_b]

        hit_b_pos = prev_origins + distance.unsqueeze(-1) * prev_directions  # (N, 3)

        # vector from each scatter back to its sensor. A finite sensor makes the direction and
        # the leg per-scatter; a planar wavefront reuses the ray's fixed line of sight instead.
        to_sensor = sensor_positions - hit_b_pos                                        # (N, 3)
        if planar_wave:
            sensor_direction = to_sensor_directions                                     # (N, 3)
            return_leg = dot_product(to_sensor, sensor_direction)                       # (N,)
        else:
            return_leg = to_sensor.norm(dim=-1)                                         # (N,)
            sensor_direction = to_sensor / return_leg.unsqueeze(-1).clamp_min(1e-12)    # (N, 3)

        n = face_normals[hit_indices]  # (N, 3)

        # orient each normal to face the incoming ray (outward on the side the ray came
        # from). Mesh triangle winding is not guaranteed consistent, so face_normals may
        # point either way; the reflection below is invariant to n's sign, but poly_input
        # (dot(n, next_directions)) is linear in n and would otherwise flip sign on
        # back-facing normals, collapsing the directional-scatter denominator and spiking
        # the returned energy. prev_directions points into the surface, so an outward
        # normal has dot(prev_directions, n) <= 0; only flip the strictly back-facing ones
        # (a where() rather than -sign(), so an exactly grazing dot==0 leaves n intact
        # instead of being zeroed).
        n = torch.where(dot_product(prev_directions, n, keepdim=True) > 0, -n, n)

        # calculate reflected ray direction
        next_directions = prev_directions - 2 * dot_product(prev_directions, n, keepdim=True) * n

        # calculate returned energy
        s = material_properties[hit_indices, 4]
        i = material_properties[hit_indices, 2]
        d = material_properties[hit_indices, 3]
        poly_input = dot_product(n, next_directions)

        # half-angle identity: cos(theta/2) = sqrt((1 + cos theta)/2), where theta is the
        # angle between the reflected ray and the direction back to the sensor. Both are
        # unit vectors, so the argument is mathematically in [0, 1]; it only dips below 0
        # by float error when the reflected ray points ~directly away from the sensor
        # (cos theta = -1), where the true value is 0. NaN -> 0 is that exact limit.
        sqrt_arg = (1 + dot_product(sensor_direction, next_directions)) / 2
        cos_theta_over_2 = torch.sqrt(sqrt_arg)
        n_nan_cos   += int(torch.isnan(cos_theta_over_2).sum())
        n_cos_total += cos_theta_over_2.numel()
        # an already-NaN argument means NaN arrived from upstream (a bad normal or
        # direction), which is a real bug rather than boundary rounding; count it apart
        # and keep it out of the min so it cannot masquerade as a benign undershoot.
        n_nan_arg    += int(torch.isnan(sqrt_arg).sum())
        min_sqrt_arg  = min(min_sqrt_arg,
                            float(torch.nan_to_num(sqrt_arg, nan=float('inf')).min()))
        cos_theta_over_2 = torch.nan_to_num(cos_theta_over_2, nan=0.0)
        energy_b = cumulative_reflectivity * s * (
            (i * cos_theta_over_2**5) /
            directional_scatter_polynomial_alpha5(poly_input) +
            d / 2 / np.pi
        )  # (N,) attenuated by the reflectivity of all prior bounces

        # store per-ray depth and energy for the first bounce (misses get -1 / 0)
        if b == 1 and first_bounce_maps:
            depth_flat = torch.full((R,), -1.0, device=device, dtype=distance.dtype)
            depth_flat[hit_b] = distance
            energy_flat = torch.zeros(R, device=device, dtype=energy_b.dtype)
            energy_flat[hit_b] = energy_b
            first_bounce = {'depth': depth_flat, 'energy': energy_flat}

        # round-trip range: the outbound leg to the first hit, plus any inter-bounce legs, plus
        # the leg straight back. Rays from a point sensor start at it, so their traced distance
        # *is* the outbound leg; planar rays start on the wavefront plane, so theirs is the
        # same projection as the return.
        if b == 1:
            outbound_range = return_leg if planar_wave else distance  # cumulative_legs stays 0
        else:
            cumulative_legs = cumulative_legs + distance
        total_range = outbound_range + cumulative_legs + return_leg  # (N,)

        # cull occluded scatters: a hit only returns energy if its path back to the
        # sensor is unobstructed. First-bounce hits are the nearest intersection along
        # the incoming ray, so they are always visible; only later bounces can be
        # occluded (e.g. a ground point hidden behind the object). This filters what we
        # store, NOT the ray that propagates onward to the next bounce.
        if b > 1 and planar_wave:
            visible = points_visible_to_far_sensor(
                hit_b_pos, sensor_direction, mesh,
                octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
            )  # (N,)
        elif b > 1:
            visible = points_visible_to_point_sensor(
                hit_b_pos, sensor_positions, mesh,
                octree=octree, surface_bias=surface_bias, batch_size=second_bounce_batch_size,
            )  # (N,)
        else:
            visible = torch.ones(hit_b_pos.shape[0], dtype=torch.bool, device=device)

        scatter_ranges      = torch.cat((scatter_ranges,      total_range[visible]))
        scatter_energies    = torch.cat((scatter_energies,    energy_b[visible]))
        scatter_positions   = torch.cat((scatter_positions,   hit_b_pos[visible]))
        scatter_ray_indices = torch.cat((scatter_ray_indices, ray_index[visible]))

        # attenuate future bounces by this surface's reflectivity
        r = material_properties[hit_indices, 0]
        cumulative_reflectivity = cumulative_reflectivity * r

        # reflect for next bounce
        prev_directions = next_directions

        # bias the new origin off the surface along the normal so the reflected ray
        # cannot spuriously re-hit the surface it just left. Without this, a reflected
        # ray that grazes the (flat, finely tessellated) ground re-intersects an adjacent
        # coplanar triangle at leg~=0, dumping a duplicate full-energy scatter at the
        # first-bounce range. n faces the incoming ray, so the reflected ray always
        # departs into the +n half-space and pushing along +n is always the correct side.
        prev_origins = hit_b_pos + surface_bias * n

        t_bounce_totals[b-1] += sync_time() - t_after_trace

    # medium absorption: exponential in path length, which here is already the round trip
    if attenuation:
        scatter_energies = scatter_energies * torch.exp(-attenuation * scatter_ranges)

    # apply complex value to the energy according to wavelength
    if wavelength is not None:
        scatter_energies = scatter_energies * torch.exp(
            1j * 2 * np.pi / wavelength * scatter_ranges
        )

    if stats is not None:
        bounce_totals = stats.setdefault('t_bounce', [])
        while len(bounce_totals) < num_bounce:
            bounce_totals.append(0.0)
        for b, t in enumerate(t_bounce_totals):
            bounce_totals[b] += t
        stats['n_nan_cos']    = stats.get('n_nan_cos', 0) + n_nan_cos
        stats['n_nan_arg']    = stats.get('n_nan_arg', 0) + n_nan_arg
        stats['n_cos_total']  = stats.get('n_cos_total', 0) + n_cos_total
        stats['min_sqrt_arg'] = min(stats.get('min_sqrt_arg', float('inf')), min_sqrt_arg)

    return scatter_ranges, scatter_energies, scatter_positions, scatter_ray_indices, first_bounce
    #      (R',)           (R',)             (R',3)             (R',)                dict of (R,) or None


# ================================================================================
# sensor geometry wrappers
# ================================================================================


def spherical_ray_fan(right_vector, up_vector, forward_vector,
                      fov_width_deg, fov_height_deg, n_ray_width, n_ray_height, device):
    '''
    Unit directions of a ray fan spread over the field of view, (H,W) flattened row-major.

    The fan is parameterized spherically rather than as a pinhole image plane, so a ray's
    azimuth is exactly its azimuth offset and the angular sampling stays uniform -- range
    angle imaging bins on that angle, so an even fan matters more than a flat image plane.

    inputs:
        right/up/forward_vector (3,): the sensor's axes
        fov_width_deg/fov_height_deg (float): angular extent in azimuth (about the sensor's
            up axis) and elevation (about its right axis)
        n_ray_width/height (int): number of rays along the azimuth and elevation axis
    outputs:
        directions (H*W,3): unit direction of every ray in the fan
        azimuths (H*W,): each ray's azimuth off boresight in degrees, positive toward the
            sensor's right vector. This is the angle the fan was built from, not a value
            recovered from the direction, so it is exact.
    '''
    az_offsets = centered_linspace(fov_width_deg  * np.pi/180, n_ray_width,  device)                # (W,) left→right
    el_offsets = centered_linspace(fov_height_deg * np.pi/180, n_ray_height, device, flip=True)     # (H,) top→bottom
    grid_el, grid_az = torch.meshgrid(el_offsets, az_offsets, indexing='ij')  # (H, W)
    directions = (
          (torch.cos(grid_el) * torch.cos(grid_az)).unsqueeze(-1) * forward_vector
        + (torch.cos(grid_el) * torch.sin(grid_az)).unsqueeze(-1) * right_vector
        + torch.sin(grid_el).unsqueeze(-1)                        * up_vector
    )  # (H, W, 3), already unit length
    return directions.reshape(-1, 3), (grid_az * 180 / np.pi).reshape(-1)


def _print_stats(name, stats, t_overall, t_octree_build, t_setup_total):
    '''Log the per-call timing and cos(theta/2) NaN summary.'''
    bounce_times = '  '.join(f'bounce{b+1}={t:.3f}s'
                             for b, t in enumerate(stats.get('t_bounce', [])))
    log.info(f"{name}: overall={t_overall:.3f}s  octree={t_octree_build:.3f}s  "
             f"setup={t_setup_total:.3f}s  {bounce_times}")
    n_nan_cos = stats.get('n_nan_cos', 0)
    if n_nan_cos:
        n_cos_total = stats['n_cos_total']
        log.info(f"{name}: cos(theta/2) NaN -> 0 on {n_nan_cos}/{n_cos_total} scatters "
                 f"({100*n_nan_cos/n_cos_total:.3f}%); min sqrt arg={stats['min_sqrt_arg']:.3e}")
        if stats.get('n_nan_arg', 0):
            log.warning(f"{name}: {stats['n_nan_arg']} of those had a NaN argument (not "
                        f"boundary rounding) -- check face normals / ray directions for NaN")


def accumulate_scatters(mesh,
                        trajectory,
                        wavelength=None,
                        grid_width=1, grid_height=1,
                        n_ray_width=1, n_ray_height=1,
                        num_bounce = 1,
                        second_bounce_batch_size = 2**100,
                        surface_bias = 1e-3,
                        debug_gif = False,
                        octree = None,
                    ):
    '''
    returns the energy and range for a bunch of rays for each pulse

    Orthographic wrapper around accumulate_scatters_from_rays: the rays are a parallel grid
    of origins on a sensor plane, and the sensor is treated as infinitely far away, so ranges
    are distances to that plane rather than to a point.

    inputs:
        mesh (Mesh): the scene, with its face normals and r,a,i,d,s materials
        trajectory (T,P,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        grid_width/height (float): the size of the ray grid for the orthonormal camera
        n_ray_width/height (int): the number of rays on the ray grid along the width and height axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        octree (Octree): a previously built octree for this mesh, built here when None. The
            octree depends only on the mesh, so a caller rendering many poses of one object
            builds it once and passes it back in instead of paying the rebuild every call

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only)
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None
    '''
    device = trajectory.device
    T = trajectory.shape[0]  # no. of camera views
    P = trajectory.shape[1]  # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

    t_octree_start = sync_time()
    if octree is None:
        octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        for p in range(P):

            t_setup_start = sync_time()

            # compute camera axes from pose matrix (columns: right, up, forward)
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t, p])
            pose           = pose_matrix(cam_azimuth_deg, cam_elevation_deg, cam_distance, device=device)
            right_vector   = pose[:3, 0]  # (3,)
            up_vector      = pose[:3, 1]  # (3,)
            forward_vector = pose[:3, 2]  # (3,)

            # set up ray origins on the sensor plane
            x_offsets = torch.linspace(-grid_width/2,  grid_width/2,  n_ray_width,  device=device)  # (W,) left→right
            y_offsets = torch.linspace( grid_height/2, -grid_height/2, n_ray_height, device=device)  # (H,) top→bottom
            grid_y, grid_x = torch.meshgrid(y_offsets, x_offsets, indexing='ij')  # (H, W)
            first_bounce_origins = (trajectory[t, p].reshape(1, 1, 3)
                                    + grid_x.unsqueeze(-1) * right_vector
                                    + grid_y.unsqueeze(-1) * up_vector)  # (H, W, 3)

            ray_origins    = first_bounce_origins.reshape(-1, 3)                             # (H*W, 3)
            ray_directions = forward_vector.unsqueeze(0).expand(ray_origins.shape[0], -1)    # (H*W, 3)

            t_setup_total += sync_time() - t_setup_start

            # the rays are spread across the sensor plane, so the sensor is not where they
            # start -- it has to be named explicitly for the planar line of sight to be right
            ranges_p, energies_p, _, _, first_bounce = accumulate_scatters_from_rays(
                mesh,
                ray_origins, ray_directions,
                sensor_positions = trajectory[t, p],
                planar_wave      = True,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-pulse hit count, which would vary with aspect and taper the
            # aperture.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return scatter_ranges, scatter_energies, debugging_maps if debug_gif else None
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse), dict (t,p)->(H,W) or None


def accumulate_scatters_perspective(mesh,
                                    trajectory,
                                    wavelength=None,
                                    fov_width_deg=30.0, fov_height_deg=30.0,
                                    n_ray_width=1, n_ray_height=1,
                                    num_bounce = 1,
                                    second_bounce_batch_size = 2**100,
                                    surface_bias = 1e-3,
                                    water_absorption = 0.0,
                                    debug_gif = False,
                                    octree = None,
                                ):
    '''
    Perspective wrapper around accumulate_scatters_from_rays: the sensor is a *point* rather
    than an infinitely distant plane, so the rays fan out from it and ranges are true distances
    to and from it.

    The sensor pose is rebuilt from each position, which always aims boresight at the scene
    origin; use accumulate_scatters_side_scan when the look direction is given instead.

    inputs:
        mesh (Mesh): the scene, with its face normals and r,a,i,d,s materials
        trajectory (T,P,3): the locations of the sensor for each pulse for each target scene
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        fov_width_deg/fov_height_deg (float): angular extent of the ray fan, in azimuth (about
            the sensor's up axis) and elevation (about its right axis)
        n_ray_width/height (int): the number of rays in the fan along the azimuth and elevation axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        water_absorption (float): absorption coefficient of the water, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.
        octree (Octree): a previously built octree for this mesh, built here when None. The
            octree depends only on the mesh, so a caller rendering many poses of one object
            builds it once and passes it back in instead of paying the rebuild every call

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only).
            Round-trip path length, so ~2x the distance to the scatter on a first bounce
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per pulse (hit rays only),
            attenuated by water absorption over that round-trip range if enabled
        azimuth (T,)[P,][R']: list of lists of 1-D tensors; the launch azimuth in degrees off
            boresight, positive toward the sensor's right vector, of the ray that produced
            each scatter. On a second bounce that is where the ray left, not where the
            scatter ended up. This is the transmit beam pattern's angle
        arrival_azimuth (T,)[P,][R']: like azimuth, but of the direction from the sensor to the
            scatter, i.e. where the receiver hears it from. Equal to azimuth on a first bounce;
            this is the receive beam pattern's angle
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None. The energy map
            is the raw first-bounce energy, before absorption and the ray-count normalization
    '''
    device = trajectory.device
    T = trajectory.shape[0]  # no. of camera views
    P = trajectory.shape[1]  # no. of pulses per view

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

    t_octree_start = sync_time()
    if octree is None:
        octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    scatter_azimuths = []
    scatter_arrival_azimuths = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        scatter_azimuths.append([])
        scatter_arrival_azimuths.append([])
        for p in range(P):

            t_setup_start = sync_time()

            # compute camera axes from pose matrix (columns: right, up, forward)
            cam_azimuth_deg, cam_elevation_deg, cam_distance = cartesian_to_spherical(trajectory[t, p])
            pose           = pose_matrix(cam_azimuth_deg, cam_elevation_deg, cam_distance, device=device)
            right_vector   = pose[:3, 0]  # (3,)
            up_vector      = pose[:3, 1]  # (3,)
            forward_vector = pose[:3, 2]  # (3,)

            # set up the ray fan: every ray leaves the sensor point, spread over the field of view
            ray_directions, ray_azimuths = spherical_ray_fan(right_vector, up_vector, forward_vector,
                                               fov_width_deg, fov_height_deg,
                                               n_ray_width, n_ray_height, device)         # (H*W, 3), (H*W,)
            ray_origins    = trajectory[t, p].reshape(1, 3).expand(ray_directions.shape[0], -1)

            t_setup_total += sync_time() - t_setup_start

            # rays leave the sensor itself, so the default sensor_positions is what we want
            ranges_p, energies_p, positions_p, ray_indices_p, first_bounce = accumulate_scatters_from_rays(
                mesh,
                ray_origins, ray_directions,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                attenuation      = water_absorption,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # azimuth of each scatter: the launch azimuth of the ray that produced it, taken
            # straight off the fan rather than recovered from the scatter's position.
            azimuths_p = ray_azimuths[ray_indices_p]                                         # (R',)

            # arrival azimuth: the sensor's view of the scatter itself, in the fan's atan2 convention
            to_scatter = positions_p - trajectory[t, p]                                      # (R',3)
            arrival_azimuths_p = torch.atan2(to_scatter @ right_vector,
                                             to_scatter @ forward_vector) * 180 / np.pi      # (R',)

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-pulse hit count, which would vary with aspect and taper the
            # aperture.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)
            scatter_azimuths[t].append(azimuths_p)
            scatter_arrival_azimuths[t].append(arrival_azimuths_p)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters_perspective', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return (scatter_ranges, scatter_energies, scatter_azimuths, scatter_arrival_azimuths,
            debugging_maps if debug_gif else None)
    #      list[T][P] of 1-D tensors (R' hit rays, varies per pulse) x4, dict (t,p)->(H,W) or None


def accumulate_scatters_side_scan(mesh,
                                  poses,
                                  wavelength=None,
                                  fov_width_deg=30.0, fov_height_deg=30.0,
                                  n_ray_width=1, n_ray_height=1,
                                  num_bounce = 1,
                                  second_bounce_batch_size = 2**100,
                                  surface_bias = 1e-3,
                                  water_absorption = 0.0,
                                  debug_gif = False,
                                  octree = None,
                              ):
    '''
    Broadside wrapper around accumulate_scatters_from_rays: like the perspective wrapper, but
    the sensor's look direction is given rather than inferred.

    accumulate_scatters_perspective takes sensor *positions* and rebuilds a pose from each one,
    which always aims boresight at the scene origin. A side scan platform does not steer -- it
    flies a straight track staring off to one side -- so this wrapper takes the full pose matrix
    and reads the sensor axes straight out of its columns. Azimuths are therefore measured off
    a fixed boresight, so a scatter's azimuth tells you where it sits along the track rather
    than always being ~0.

    inputs:
        mesh (Mesh): the scene, with its face normals and r,a,i,d,s materials
        poses (T,P,4,4): the sensor pose for each ping for each target scene, columns
            (right, up, forward, center) in the srn_cars convention of pose_matrix
        wavelength (float): the wavelength of the radar signal, if none, there will be no complex value in the energy
        fov_width_deg/fov_height_deg (float): angular extent of the ray fan, in azimuth (about
            the sensor's up axis) and elevation (about its right axis)
        n_ray_width/height (int): the number of rays in the fan along the azimuth and elevation axis.
        surface_bias (float): distance to push each bounce's outgoing ray origin off the surface
            along the normal, to prevent self-intersection (spurious leg~=0 re-hits). Should be
            small relative to scene features but large relative to float error at the scene scale.
        water_absorption (float): absorption coefficient of the water, in nepers per unit length
            (0 disables it). Applied over the round trip, so it is already the two-way loss.
            From the dB/m absorption is usually tabulated in: nepers = dB / 8.686.
        octree (Octree): a previously built octree for this mesh, built here when None. The
            octree depends only on the mesh, so a caller rendering many poses of one object
            builds it once and passes it back in instead of paying the rebuild every call

    outputs:
        range (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only).
            Round-trip path length, so ~2x the distance to the scatter on a first bounce
        energy (T,)[P,][R']: list of lists of 1-D tensors; R' varies per ping (hit rays only),
            attenuated by water absorption over that round-trip range if enabled
        azimuth (T,)[P,][R']: list of lists of 1-D tensors; the launch azimuth in degrees off
            the fixed boresight, positive toward the sensor's right vector (i.e. forward along
            track), of the ray that produced each scatter. On a second bounce that is where the
            ray left, not where the scatter ended up. This is the transmit beam pattern's angle
        arrival_azimuth (T,)[P,][R']: like azimuth, but of the direction from the sensor to the
            scatter, i.e. where the receiver hears it from. Equal to azimuth on a first bounce;
            this is the receive beam pattern's angle
        debugging_maps: dict (t,p) -> {'depth','energy'} of (H,W) maps, or None. The energy map
            is the raw first-bounce energy, before absorption and the ray-count normalization
    '''
    device = poses.device
    T = poses.shape[0]  # no. of scenes
    P = poses.shape[1]  # no. of pings per scene

    def sync_time():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter()

    t_overall_start = sync_time()
    t_setup_total   = 0.0
    stats           = {}

    t_octree_start = sync_time()
    if octree is None:
        octree = build_octree(mesh)
    t_octree_build = sync_time() - t_octree_start

    debugging_maps = {}  # (t, p) -> {'depth': (H,W), 'energy': (H,W)}; only populated when debug_gif=True

    scatter_ranges = []
    scatter_energies = []
    scatter_azimuths = []
    scatter_arrival_azimuths = []
    for t in range(T):
        scatter_ranges.append([])
        scatter_energies.append([])
        scatter_azimuths.append([])
        scatter_arrival_azimuths.append([])
        for p in range(P):

            t_setup_start = sync_time()

            # read the camera axes straight out of the pose (columns: right, up, forward, center)
            right_vector    = poses[t, p, :3, 0]  # (3,)
            up_vector       = poses[t, p, :3, 1]  # (3,)
            forward_vector  = poses[t, p, :3, 2]  # (3,)
            sensor_position = poses[t, p, :3, 3]  # (3,)

            # set up the ray fan: every ray leaves the sensor point, spread over the field of view
            ray_directions, ray_azimuths = spherical_ray_fan(right_vector, up_vector, forward_vector,
                                               fov_width_deg, fov_height_deg,
                                               n_ray_width, n_ray_height, device)         # (H*W,3), (H*W,)
            ray_origins    = sensor_position.reshape(1, 3).expand(ray_directions.shape[0], -1)

            t_setup_total += sync_time() - t_setup_start

            # rays leave the sensor itself, so the default sensor_positions is what we want
            ranges_p, energies_p, positions_p, ray_indices_p, first_bounce = accumulate_scatters_from_rays(
                mesh,
                ray_origins, ray_directions,
                wavelength       = wavelength,
                num_bounce       = num_bounce,
                attenuation      = water_absorption,
                octree           = octree,
                second_bounce_batch_size = second_bounce_batch_size,
                surface_bias     = surface_bias,
                first_bounce_maps = debug_gif,
                stats            = stats,
            )

            # azimuth of each scatter: the launch azimuth of the ray that produced it, taken
            # straight off the fan rather than recovered from the scatter's position.
            azimuths_p = ray_azimuths[ray_indices_p]                                         # (R',)

            # arrival azimuth: the sensor's view of the scatter itself, in the fan's atan2 convention
            to_scatter = positions_p - sensor_position                                       # (R',3)
            arrival_azimuths_p = torch.atan2(to_scatter @ right_vector,
                                             to_scatter @ forward_vector) * 180 / np.pi      # (R',)

            # normalized by number of rays. The divisor is the constant *transmitted* ray
            # count, not the per-ping hit count, which would vary with aspect and taper the track.
            scatter_ranges[t].append(ranges_p)
            scatter_energies[t].append(energies_p/n_ray_width/n_ray_height)
            scatter_azimuths[t].append(azimuths_p)
            scatter_arrival_azimuths[t].append(arrival_azimuths_p)

            if first_bounce is not None:
                debugging_maps[(t, p)] = {
                    'depth':  first_bounce['depth'].reshape(n_ray_height, n_ray_width),
                    'energy': first_bounce['energy'].reshape(n_ray_height, n_ray_width),
                }

    _print_stats('accumulate_scatters_side_scan', stats,
                 sync_time() - t_overall_start, t_octree_build, t_setup_total)

    return (scatter_ranges, scatter_energies, scatter_azimuths, scatter_arrival_azimuths,
            debugging_maps if debug_gif else None)
    #      list[T][P] of 1-D tensors (R' hit rays, varies per ping) x4, dict (t,p)->(H,W) or None
