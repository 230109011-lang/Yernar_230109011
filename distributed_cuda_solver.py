import os
import sys
import json
import math
import hashlib
import subprocess
from types import SimpleNamespace
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
REPORT_EVERY = 100
LAMBDA_DAMP = 0.15
SOURCE_AMP = 5.0e-4
CFL_FRACTION = 0.8  # dt = CFL_FRACTION * dt_max

TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3  # tag = direction of travel


def make_event(timer):
    try:
        return cuda.event(timer=timer)
    except TypeError:
        return cuda.event()


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    row_rank, col_rank = cart_comm.Get_coords(cart_comm.Get_rank())
    # Shift returns (source, dest): along axis 0 with disp=+1, source is the "up"
    # neighbour (row-1) and dest is the "down" neighbour (row+1).
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return SimpleNamespace(
        comm=comm, cart_comm=cart_comm, rank=cart_comm.Get_rank(), size=size,
        row_rank=row_rank, col_rank=col_rank,
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, instance_seed=instance_seed, github_user=github_user,
        ny=GLOBAL_NY // 2, nx=GLOBAL_NX // 2,
        dx=1.0 / (GLOBAL_NX + 1), dy=1.0 / (GLOBAL_NY + 1),
    )


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (rank-local only)
# ==============================================================================
def initialize_subdomains(ctx):
    ny, nx = ctx.ny, ctx.nx
    # local index i in 0..ny+1  ->  global interior row gi = row_rank*ny + (i-1); y=(gi+1)*dy
    gi = ctx.row_rank * ny + (np.arange(ny + 2) - 1)
    gj = ctx.col_rank * nx + (np.arange(nx + 2) - 1)
    y = (gi + 1) * ctx.dy
    x = (gj + 1) * ctx.dx
    Y, X = np.meshgrid(y, x, indexing="ij")  # shape (258, 258)

    rng = np.random.default_rng(ctx.instance_seed)  # identical stream on every rank
    phase = rng.uniform(0.0, 1.0, size=2)
    n_pulses = 3
    cx = rng.uniform(0.2, 0.8, n_pulses)
    cy = rng.uniform(0.2, 0.8, n_pulses)
    amp = rng.uniform(0.6, 1.0, n_pulses)
    sig = rng.uniform(0.05, 0.09, n_pulses)
    sx, sy = rng.uniform(0.3, 0.7, 2)

    # kappa in [0.08, 0.24]
    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * (X + phase[0])) * np.cos(2 * np.pi * (Y + phase[1]))

    u = np.zeros_like(X)
    for k in range(n_pulses):
        u += amp[k] * np.exp(-((X - cx[k]) ** 2 + (Y - cy[k]) ** 2) / (2 * sig[k] ** 2))

    src = SOURCE_AMP * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * 0.04 ** 2))

    # Dirichlet u = 0 on outer global edges (ghost layer facing MPI.PROC_NULL)
    if ctx.nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx.nbr_down == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx.nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx.nbr_right == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa), np.ascontiguousarray(src))


# ==============================================================================
# TASK 3: GPU halo pack / unpack kernels + pinned MPI exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, ny, nx, send_up, send_down, send_left, send_right):
    t = cuda.grid(1)
    if t < nx:
        send_up[t] = u[1, t + 1]        # row 1   (top owned row)
        send_down[t] = u[ny, t + 1]     # row ny  (bottom owned row)
    if t < ny:
        send_left[t] = u[t + 1, 1]      # column 1
        send_right[t] = u[t + 1, nx]    # column nx


@cuda.jit
def unpack_halos_kernel(u, ny, nx, recv_up, recv_down, recv_left, recv_right):
    t = cuda.grid(1)
    if t < nx:
        u[0, t + 1] = recv_up[t]
        u[ny + 1, t + 1] = recv_down[t]
    if t < ny:
        u[t + 1, 0] = recv_left[t]
        u[t + 1, nx + 1] = recv_right[t]


def make_state(ctx, u0, kappa, src):
    ny, nx = ctx.ny, ctx.nx
    st = SimpleNamespace()
    st.stream_boundary = cuda.stream()
    st.stream_interior = cuda.stream()
    st.d_a = cuda.to_device(u0)
    st.d_b = cuda.to_device(u0)
    st.d_kappa = cuda.to_device(kappa)
    st.d_src = cuda.to_device(src)
    st.u0 = u0
    for name, n in (("up", nx), ("down", nx), ("left", ny), ("right", ny)):
        setattr(st, f"d_send_{name}", cuda.device_array(n, dtype=np.float64))
        setattr(st, f"d_recv_{name}", cuda.device_array(n, dtype=np.float64))
        setattr(st, f"h_send_{name}", cuda.pinned_array(n, dtype=np.float64))
        hr = cuda.pinned_array(n, dtype=np.float64)
        hr[:] = 0.0  # stays 0.0 on PROC_NULL sides -> homogeneous Dirichlet ghost
        setattr(st, f"h_recv_{name}", hr)
    st.ev_i = make_event(False)
    st.ev_b = make_event(False)
    return st


def reset_state(st):
    st.d_a.copy_to_device(st.u0)
    st.d_b.copy_to_device(st.u0)
    cuda.synchronize()


def exchange_halos_async(ctx, st, d_u, on_posted=None):
    """Stage A (pack + async D2H), Stage B (Isend/Irecv, optional concurrent work via
    on_posted), and the first half of Stage C (Waitall, async H2D, unpack)."""
    ny, nx = ctx.ny, ctx.nx
    sb = st.stream_boundary
    cart = ctx.cart_comm

    # ---- Stage A ----
    threads = 128
    blocks = (max(ny, nx) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, sb](
        d_u, ny, nx, st.d_send_up, st.d_send_down, st.d_send_left, st.d_send_right)
    for name in ("up", "down", "left", "right"):
        getattr(st, f"d_send_{name}").copy_to_host(getattr(st, f"h_send_{name}"), stream=sb)
    sb.synchronize()  # host send buffers are now valid

    # ---- Stage B ----
    reqs = []
    # Message tag = direction of travel; a rank's recv from "down" matches the
    # neighbour's send "up" (tag TAG_UP), and so on.
    reqs.append(cart.Irecv(st.h_recv_down, source=ctx.nbr_down, tag=TAG_UP))
    reqs.append(cart.Irecv(st.h_recv_up, source=ctx.nbr_up, tag=TAG_DOWN))
    reqs.append(cart.Irecv(st.h_recv_right, source=ctx.nbr_right, tag=TAG_LEFT))
    reqs.append(cart.Irecv(st.h_recv_left, source=ctx.nbr_left, tag=TAG_RIGHT))
    reqs.append(cart.Isend(st.h_send_up, dest=ctx.nbr_up, tag=TAG_UP))
    reqs.append(cart.Isend(st.h_send_down, dest=ctx.nbr_down, tag=TAG_DOWN))
    reqs.append(cart.Isend(st.h_send_left, dest=ctx.nbr_left, tag=TAG_LEFT))
    reqs.append(cart.Isend(st.h_send_right, dest=ctx.nbr_right, tag=TAG_RIGHT))

    if on_posted is not None:
        on_posted()  # e.g. launch deep-interior stencil on stream_interior

    # ---- Stage C (first half) ----
    MPI.Request.Waitall(reqs)
    for name in ("up", "down", "left", "right"):
        getattr(st, f"d_recv_{name}").copy_to_device(getattr(st, f"h_recv_{name}"), stream=sb)
    unpack_halos_kernel[blocks, threads, sb](
        d_u, ny, nx, st.d_recv_up, st.d_recv_down, st.d_recv_left, st.d_recv_right)


# ==============================================================================
# TASK 4: 2D shared-memory tiled stencil + dual-stream orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u_in, u_out, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    """Updates cells with r_start<=i<=r_end, c_start<=j<=c_end (inclusive)."""
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    inside = (i <= r_end) and (j <= c_end)

    if inside:
        smem[ty + 1, tx + 1] = u_in[i, j]                   # center cell
        if ty == 0:
            smem[0, tx + 1] = u_in[i - 1, j]                # top apron
        if ty == TILE_DIM - 1 or i == r_end:
            smem[ty + 2, tx + 1] = u_in[i + 1, j]           # bottom apron
        if tx == 0:
            smem[ty + 1, 0] = u_in[i, j - 1]                # left apron
        if tx == TILE_DIM - 1 or j == c_end:
            smem[ty + 1, tx + 2] = u_in[i, j + 1]           # right apron

    cuda.syncthreads()  # every thread of the block reaches this barrier

    if inside:
        c = smem[ty + 1, tx + 1]
        n = smem[ty, tx + 1]
        s = smem[ty + 2, tx + 1]
        w = smem[ty + 1, tx]
        e = smem[ty + 1, tx + 2]
        kc = kappa[i, j]
        kn = 0.5 * (kappa[i - 1, j] + kc)
        ks = 0.5 * (kappa[i + 1, j] + kc)
        kw = 0.5 * (kappa[i, j - 1] + kc)
        ke = 0.5 * (kappa[i, j + 1] + kc)
        lap = (ks * (s - c) - kn * (c - n)) * inv_dy2 + (ke * (e - c) - kw * (c - w)) * inv_dx2
        u_out[i, j] = c + dt * (lap - lam * c * c * c + src[i, j])


def launch_stencil(ctx, st, stream, d_in, d_out, r0, r1, c0, c1, dt):
    if r1 < r0 or c1 < c0:
        return
    grid = ((c1 - c0 + TILE_DIM) // TILE_DIM, (r1 - r0 + TILE_DIM) // TILE_DIM)
    tiled_stencil_kernel[grid, (TILE_DIM, TILE_DIM), stream](
        d_in, d_out, st.d_kappa, st.d_src, r0, r1, c0, c1,
        dt, 1.0 / ctx.dx ** 2, 1.0 / ctx.dy ** 2, LAMBDA_DAMP)


def step_serialized(ctx, st, d_in, d_out, dt):
    ny, nx = ctx.ny, ctx.nx
    exchange_halos_async(ctx, st, d_in, on_posted=None)
    launch_stencil(ctx, st, st.stream_boundary, d_in, d_out, 1, ny, 1, nx, dt)
    st.stream_boundary.synchronize()


def step_overlapped(ctx, st, d_in, d_out, dt):
    ny, nx = ctx.ny, ctx.nx
    sb, si = st.stream_boundary, st.stream_interior

    def launch_interior():
        launch_stencil(ctx, st, si, d_in, d_out, 2, ny - 1, 2, nx - 1, dt)

    exchange_halos_async(ctx, st, d_in, on_posted=launch_interior)

    # Boundary ring (ghosts are populated, ordered on stream_boundary)
    launch_stencil(ctx, st, sb, d_in, d_out, 1, 1, 1, nx, dt)             # top row
    launch_stencil(ctx, st, sb, d_in, d_out, ny, ny, 1, nx, dt)           # bottom row
    launch_stencil(ctx, st, sb, d_in, d_out, 2, ny - 1, 1, 1, dt)         # left column
    launch_stencil(ctx, st, sb, d_in, d_out, 2, ny - 1, nx, nx, dt)       # right column

    # Join the streams (device-side, no host stall) so the next step sees a full field.
    st.ev_i.record(si)
    st.ev_i.wait(sb)
    st.ev_b.record(sb)
    st.ev_b.wait(si)


# ==============================================================================
# TASK 5: Reductions, PTX audit, benchmark, manifest
# ==============================================================================
def global_diagnostics(ctx, d_cur, d_prev, step):
    ny, nx = ctx.ny, ctx.nx
    cur = d_cur.copy_to_host()[1:ny + 1, 1:nx + 1]
    e_loc = float(cur.sum()) * ctx.dx * ctx.dy
    if step == 0:
        r2_loc = 0.0
    else:
        prev = d_prev.copy_to_host()[1:ny + 1, 1:nx + 1]
        r2_loc = float(np.sum((cur - prev) ** 2)) * ctx.dx * ctx.dy
    send = np.array([e_loc, r2_loc], dtype=np.float64)
    recv = np.zeros(2, dtype=np.float64)
    ctx.cart_comm.Allreduce(send, recv, op=MPI.SUM)
    return float(recv[0]), math.sqrt(float(recv[1]))


def run_mode(ctx, st, mode, steps, dt):
    stepper = step_overlapped if mode == "overlapped" else step_serialized
    reset_state(st)
    d_cur, d_other = st.d_a, st.d_b
    trace = []
    e, r = global_diagnostics(ctx, d_cur, d_cur, 0)
    trace.append({"step": 0, "energy": e, "l2_residual": r})

    ev0, ev1 = make_event(True), make_event(True)
    gpu_ms = 0.0
    wall_ms = 0.0
    for chunk_start in range(0, steps, REPORT_EVERY):
        ctx.cart_comm.Barrier()
        cuda.synchronize()
        t0 = MPI.Wtime()
        ev0.record(st.stream_boundary)
        for _ in range(REPORT_EVERY):
            stepper(ctx, st, d_cur, d_other, dt)
            d_cur, d_other = d_other, d_cur
        st.stream_interior.synchronize()
        ev1.record(st.stream_boundary)
        ev1.synchronize()
        cuda.synchronize()
        wall_ms += (MPI.Wtime() - t0) * 1e3
        gpu_ms += cuda.event_elapsed_time(ev0, ev1)
        step = chunk_start + REPORT_EVERY
        e, r = global_diagnostics(ctx, d_cur, d_other, step)  # d_other = previous state
        trace.append({"step": step, "energy": e, "l2_residual": r})

    final = d_cur.copy_to_host()[1:ctx.ny + 1, 1:ctx.nx + 1].copy()
    return final, trace, gpu_ms, wall_ms


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        asm = "\n".join(asm.values())
    counts = {
        "st.shared.f64": asm.count("st.shared.f64"),
        "ld.shared.f64": asm.count("ld.shared.f64"),
        "bar.sync": asm.count("bar.sync"),
        "st.shared (any)": asm.count("st.shared"),
        "ld.shared (any)": asm.count("ld.shared"),
    }
    passed = counts["st.shared (any)"] > 0 and counts["ld.shared (any)"] > 0 and counts["bar.sync"] > 0
    return counts, passed


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    u0, kappa, src = initialize_subdomains(ctx)
    st = make_state(ctx, u0, kappa, src)
    cart = ctx.cart_comm

    kmax_loc = np.array([kappa[1:ctx.ny + 1, 1:ctx.nx + 1].max()])
    kmax = np.zeros(1)
    cart.Allreduce(kmax_loc, kmax, op=MPI.MAX)
    kmax = float(kmax[0])
    dt_max = ctx.dx ** 2 / (4.0 * kmax)
    dt = CFL_FRACTION * dt_max

    # JIT warm-up (excluded from timings)
    for fn in (step_serialized, step_overlapped):
        reset_state(st)
        fn(ctx, st, st.d_a, st.d_b, dt)
        cuda.synchronize()

    final_s, trace_s, gpu_ms_s, wall_ms_s = run_mode(ctx, st, "serialized", STEPS, dt)
    final_o, trace_o, gpu_ms_o, wall_ms_o = run_mode(ctx, st, "overlapped", STEPS, dt)

    diff_loc = np.array([float(np.max(np.abs(final_s - final_o)))])
    diff = np.zeros(1)
    cart.Allreduce(diff_loc, diff, op=MPI.MAX)
    parity = float(diff[0])

    timings_loc = np.array([wall_ms_s, wall_ms_o, gpu_ms_s, gpu_ms_o])
    timings = np.zeros(4)
    cart.Allreduce(timings_loc, timings, op=MPI.MAX)

    counts, ptx_ok = ptx_audit()

    energies = [t["energy"] for t in trace_o]
    monotonic = all(energies[k + 1] < energies[k] for k in range(len(energies) - 1))

    if ctx.rank == 0:
        print("=" * 72)
        print(f"GITHUB_USERNAME     : {github_user}")
        print(f"GPU UUID            : {ctx.gpu_uuid}")
        print(f"INSTANCE_SEED       : {ctx.instance_seed}")
        print(f"MPI ranks / grid    : {ctx.size} ranks, 2 x 2 Cartesian (non-periodic)")
        print(f"dx = dy             : {ctx.dx:.9e}")
        print(f"kappa_max           : {kmax:.9f}")
        print(f"dt_max = dx^2/(4k)  : {dt_max:.9e}")
        print(f"dt used             : {dt:.9e}  ({CFL_FRACTION:.2f} * dt_max)")
        print("-" * 72)
        print("Allreduce trace (overlapped run):")
        for t in trace_o:
            print(f"  step {t['step']:4d}  E = {t['energy']:.12e}  L2 residual = {t['l2_residual']:.6e}")
        print("Allreduce trace (serialized run):")
        for t in trace_s:
            print(f"  step {t['step']:4d}  E = {t['energy']:.12e}  L2 residual = {t['l2_residual']:.6e}")
        print(f"Energy strictly decreasing: {monotonic}")
        print("-" * 72)
        print(f"serialized_ms (wall, max over ranks) : {timings[0]:.3f}")
        print(f"overlapped_ms (wall, max over ranks) : {timings[1]:.3f}")
        print(f"serialized_ms (CUDA events)          : {timings[2]:.3f}")
        print(f"overlapped_ms (CUDA events)          : {timings[3]:.3f}")
        print(f"overlap speedup (wall)               : {timings[0] / timings[1]:.4f}x")
        print(f"parity_max_diff                      : {parity:.3e}  (< 1e-12 required)")
        print("-" * 72)
        for k, v in counts.items():
            print(f"PTX {k:18s}: {v}")
        print(f"ptx_audit_passed: {ptx_ok}")
        print("=" * 72)

        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx.gpu_uuid,
            "instance_seed": ctx.instance_seed,
            "mpi_ranks": ctx.size,
            "cartesian_grid": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [ctx.ny + 2, ctx.nx + 2],
            "steps": STEPS,
            "dx": ctx.dx,
            "kappa_max": kmax,
            "dt_max": dt_max,
            "dt": dt,
            "lambda": LAMBDA_DAMP,
            "energy_trace": trace_o,
            "energy_strictly_decreasing": monotonic,
            "serialized_ms": float(timings[0]),
            "overlapped_ms": float(timings[1]),
            "serialized_cuda_event_ms": float(timings[2]),
            "overlapped_cuda_event_ms": float(timings[3]),
            "parity_max_diff": parity,
            "parity_passed": bool(parity < 1e-12),
            "ptx_counts": counts,
            "ptx_audit_passed": bool(ptx_ok),
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("Wrote submission_manifest.json")

    ok = (parity < 1e-12) and ptx_ok
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: distributed_cuda_solver.py <github_username>")
    rc = run_solver(sys.argv[1])
    sys.exit(rc)
