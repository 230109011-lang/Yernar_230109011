import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
from types import SimpleNamespace
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
REPORT_EVERY = 100
SHM_ITERS = 40_000_000
SHM_SLOTS = 64
SHM_STRIDES = (1, 8, 16)          # 8 B, 64 B, 128 B spacing
SHM_REPEATS = 3                   # every stride is timed SHM_REPEATS times, best run is reported
LAMBDA_DAMP = 0.15
SOURCE_AMP = 5.0e-4
CFL_FRACTION = 0.8                # dt = CFL_FRACTION * dt_max
MANIFEST_PATH = "hybrid_lab_manifest.json"

TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3  # tag = direction of travel


def make_event(timer):
    """cuda.event() signature differs between numba versions."""
    try:
        return cuda.event(timer=timer)
    except TypeError:
        return cuda.event()


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def create_shared_window(comm):
    """Node-local communicator plus a 64-byte-aligned window of SHM_SLOTS uint64 slots."""
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    pad = 8  # 8 extra slots (64 B) so a 64-byte-aligned start always exists inside the window
    nbytes = (SHM_SLOTS + pad) * 8 if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.frombuffer(buf, dtype=np.uint64)
    offset = ((-raw.ctypes.data) % 64) // 8
    view = raw[offset: offset + SHM_SLOTS]
    aligned = (view.ctypes.data % 64 == 0)
    offsets = node_comm.allgather(int(offset))
    return node_comm, win, view, {
        "node_size": node_comm.Get_size(),
        "node_rank": node_rank,
        "alignment_offset_slots": int(offset),
        "all_ranks_same_offset": len(set(offsets)) == 1,
        "view_64byte_aligned": bool(aligned),
        "slots": SHM_SLOTS,
    }


def init_hybrid_context(github_user: str):
    assert github_user != "REPLACE_WITH_YOUR_GITHUB_USERNAME" and len(github_user.strip()) > 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{cpu_model}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}::{line_bytes}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    row_rank, col_rank = cart_comm.Get_coords(cart_comm.Get_rank())
    nbr_up, nbr_down = cart_comm.Shift(0, 1)      # (source, dest) = (row-1, row+1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)   # (source, dest) = (col-1, col+1)

    node_comm, win, shm_view, shm_info = create_shared_window(comm)
    if shm_info["node_size"] != 4:
        raise RuntimeError("All 4 ranks must live on one node for the shared-memory window.")

    return SimpleNamespace(
        comm=comm, cart_comm=cart_comm, node_comm=node_comm, win=win, shm_view=shm_view,
        shm_info=shm_info, rank=cart_comm.Get_rank(), size=size,
        row_rank=row_rank, col_rank=col_rank,
        nbr_up=nbr_up, nbr_down=nbr_down, nbr_left=nbr_left, nbr_right=nbr_right,
        gpu_uuid=gpu_uuid, cpu_model=cpu_model, cpu_sig=cpu_sig, line_bytes=line_bytes,
        instance_seed=instance_seed, github_user=github_user.strip(),
        ny=GLOBAL_NY // 2, nx=GLOBAL_NX // 2,
        dx=1.0 / (GLOBAL_NX + 1), dy=1.0 / (GLOBAL_NY + 1),
    )


# ==============================================================================
# TASK 2: Rank-local fields + halo pack / unpack kernels
# ==============================================================================
def initialize_fields(ctx):
    ny, nx = ctx.ny, ctx.nx
    gi = ctx.row_rank * ny + (np.arange(ny + 2) - 1)   # global interior row index of local row i
    gj = ctx.col_rank * nx + (np.arange(nx + 2) - 1)
    y = (gi + 1) * ctx.dy
    x = (gj + 1) * ctx.dx
    Y, X = np.meshgrid(y, x, indexing="ij")            # (258, 258), no global grid is allocated

    rng = np.random.default_rng(ctx.instance_seed)     # same stream on every rank
    phase = rng.uniform(0.0, 1.0, size=2)
    n_pulses = 3
    cx = rng.uniform(0.2, 0.8, n_pulses)
    cy = rng.uniform(0.2, 0.8, n_pulses)
    amp = rng.uniform(0.6, 1.0, n_pulses)
    sig = rng.uniform(0.05, 0.09, n_pulses)
    sx, sy = rng.uniform(0.3, 0.7, 2)

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * (X + phase[0])) * np.cos(2 * np.pi * (Y + phase[1]))
    u = np.zeros_like(X)
    for k in range(n_pulses):
        u += amp[k] * np.exp(-((X - cx[k]) ** 2 + (Y - cy[k]) ** 2) / (2 * sig[k] ** 2))
    src = SOURCE_AMP * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2 * 0.04 ** 2))

    if ctx.nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if ctx.nbr_down == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if ctx.nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if ctx.nbr_right == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0
    return np.ascontiguousarray(u), np.ascontiguousarray(kappa), np.ascontiguousarray(src)


@cuda.jit
def pack_halos_kernel(u, ny, nx, send_up, send_down, send_left, send_right):
    t = cuda.grid(1)
    if t < nx:
        send_up[t] = u[1, t + 1]        # top owned row (contiguous in memory)
        send_down[t] = u[ny, t + 1]     # bottom owned row
    if t < ny:
        send_left[t] = u[t + 1, 1]      # left owned column (strided in memory)
        send_right[t] = u[t + 1, nx]    # right owned column


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
        hr[:] = 0.0   # stays 0 on PROC_NULL sides -> homogeneous Dirichlet ghost
        setattr(st, f"h_recv_{name}", hr)
    st.ev_i = make_event(False)
    st.ev_b = make_event(False)
    return st


def reset_state(st):
    st.d_a.copy_to_device(st.u0)
    st.d_b.copy_to_device(st.u0)
    cuda.synchronize()


def exchange_halos_async(ctx, st, d_u, on_posted=None):
    ny, nx = ctx.ny, ctx.nx
    sb = st.stream_boundary
    cart = ctx.cart_comm

    threads = 128
    blocks = (max(ny, nx) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, sb](
        d_u, ny, nx, st.d_send_up, st.d_send_down, st.d_send_left, st.d_send_right)
    for name in ("up", "down", "left", "right"):
        getattr(st, f"d_send_{name}").copy_to_host(getattr(st, f"h_send_{name}"), stream=sb)
    sb.synchronize()

    # tag = direction of travel: my recv from "down" matches the neighbour's send "up"
    reqs = [
        cart.Irecv(st.h_recv_down, source=ctx.nbr_down, tag=TAG_UP),
        cart.Irecv(st.h_recv_up, source=ctx.nbr_up, tag=TAG_DOWN),
        cart.Irecv(st.h_recv_right, source=ctx.nbr_right, tag=TAG_LEFT),
        cart.Irecv(st.h_recv_left, source=ctx.nbr_left, tag=TAG_RIGHT),
        cart.Isend(st.h_send_up, dest=ctx.nbr_up, tag=TAG_UP),
        cart.Isend(st.h_send_down, dest=ctx.nbr_down, tag=TAG_DOWN),
        cart.Isend(st.h_send_left, dest=ctx.nbr_left, tag=TAG_LEFT),
        cart.Isend(st.h_send_right, dest=ctx.nbr_right, tag=TAG_RIGHT),
    ]

    if on_posted is not None:
        on_posted()   # launch deep-interior stencil on stream_interior while MPI is in flight

    MPI.Request.Waitall(reqs)
    for name in ("up", "down", "left", "right"):
        getattr(st, f"d_recv_{name}").copy_to_device(getattr(st, f"h_recv_{name}"), stream=sb)
    unpack_halos_kernel[blocks, threads, sb](
        d_u, ny, nx, st.d_recv_up, st.d_recv_down, st.d_recv_left, st.d_recv_right)


# ==============================================================================
# TASK 3: 2D shared-memory tiled stencil + dual-stream pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u_in, u_out, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    inside = (i <= r_end) and (j <= c_end)

    if inside:
        smem[ty + 1, tx + 1] = u_in[i, j]
        if ty == 0:
            smem[0, tx + 1] = u_in[i - 1, j]
        if ty == TILE_DIM - 1 or i == r_end:
            smem[ty + 2, tx + 1] = u_in[i + 1, j]
        if tx == 0:
            smem[ty + 1, 0] = u_in[i, j - 1]
        if tx == TILE_DIM - 1 or j == c_end:
            smem[ty + 1, tx + 2] = u_in[i, j + 1]

    cuda.syncthreads()

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
        launch_stencil(ctx, st, si, d_in, d_out, 2, ny - 1, 2, nx - 1, dt)   # deep interior [2..255]

    exchange_halos_async(ctx, st, d_in, on_posted=launch_interior)

    launch_stencil(ctx, st, sb, d_in, d_out, 1, 1, 1, nx, dt)             # top strip
    launch_stencil(ctx, st, sb, d_in, d_out, ny, ny, 1, nx, dt)           # bottom strip
    launch_stencil(ctx, st, sb, d_in, d_out, 2, ny - 1, 1, 1, dt)         # left strip
    launch_stencil(ctx, st, sb, d_in, d_out, 2, ny - 1, nx, nx, dt)       # right strip

    st.ev_i.record(si)
    st.ev_i.wait(sb)
    st.ev_b.record(sb)
    st.ev_b.wait(si)


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
        e, r = global_diagnostics(ctx, d_cur, d_other, step)
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
    passed = (counts["st.shared (any)"] > 0 and counts["ld.shared (any)"] > 0
              and counts["bar.sync"] > 0)
    return counts, passed


# ==============================================================================
# TASK 4: Multi-rank host cache-line false-sharing sweep (MPI-3 shared memory)
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters):
    # `z` is 0 at run time but is read from the shared window, so LLVM cannot know it.
    # The store uses index j = slot + z while the next iteration loads shm_view[slot]:
    # LLVM can prove neither j == slot nor j != slot, so it cannot keep the counter in a
    # register or fold the loop into one addition -> a real load+store on every iteration.
    z = np.int64(shm_view[slot] >> uint64(63))
    j = slot + z
    for _ in range(iters):
        shm_view[j] = shm_view[slot] + uint64(1)
    return shm_view[slot]


def benchmark_shm_strides(ctx, iters=SHM_ITERS):
    node_comm, win, shm = ctx.node_comm, ctx.win, ctx.shm_view
    me = node_comm.Get_rank()

    # JIT warm-up on a private array so compilation is never timed
    tmp = np.zeros(SHM_SLOTS, dtype=np.uint64)
    host_shm_accumulate(tmp, me, 1000)

    results = {}
    for stride in SHM_STRIDES:
        times = []
        exact = True
        for _ in range(SHM_REPEATS):
            win.Fence()
            if me == 0:
                shm[:] = 0
            win.Fence()
            node_comm.Barrier()
            t0 = MPI.Wtime()
            host_shm_accumulate(shm, me * stride, iters)
            t1 = MPI.Wtime()
            node_comm.Barrier()
            win.Fence()
            my_ms = (t1 - t0) * 1e3
            times.append(node_comm.allreduce(my_ms, op=MPI.MAX))   # slowest rank sets the latency
            ok = int(shm[me * stride]) == iters
            exact = exact and bool(node_comm.allreduce(int(ok), op=MPI.MIN))
        best = min(times)
        results[f"stride_{stride}"] = {
            "stride_elements": stride,
            "stride_bytes": stride * 8,
            "best_ms": best,
            "all_ms": times,
            "ns_per_update": best * 1e6 / iters,
            "exact_final_counters": bool(exact),
        }
    return results


def llvm_audit_cpu():
    ir = host_shm_accumulate.inspect_llvm()
    full = "\n".join(ir.values()) if isinstance(ir, dict) else str(ir)
    # Numba also emits a CPython wrapper that legitimately calls PyArg_*/PyLong_*.
    # Audit only the native kernel body.
    parts = full.split("\ndefine ")
    native = [p for p in parts[1:] if "@_ZN7cpython" not in p.split("(", 1)[0]]
    text = "\n".join(native) if native else full
    store_i64 = text.count("store i64")
    load_i64 = text.count("load i64")
    fallback = any(tok in text for tok in ("PyObject", "PyArg_Parse", "pyapi", "PyLong"))
    snippet = [ln.strip() for ln in text.splitlines()
               if ("store i64" in ln or "load i64" in ln) and "%" in ln][:6]
    return {
        "llvm_store_i64_count": store_i64,
        "llvm_load_i64_count": load_i64,
        "llvm_python_object_fallback": bool(fallback),
        "llvm_ok": bool(store_i64 > 0 and load_i64 > 0 and not fallback),
        "llvm_snippet": snippet,
    }


# ==============================================================================
# TASK 5: Cross-tier verification and manifest export
# ==============================================================================
def run_hybrid(github_user: str) -> int:
    ctx = init_hybrid_context(github_user)
    u0, kappa, src = initialize_fields(ctx)
    st = make_state(ctx, u0, kappa, src)
    cart = ctx.cart_comm

    kmax_loc = np.array([kappa[1:ctx.ny + 1, 1:ctx.nx + 1].max()])
    kmax = np.zeros(1)
    cart.Allreduce(kmax_loc, kmax, op=MPI.MAX)
    kmax = float(kmax[0])
    dt_max = ctx.dx ** 2 / (4.0 * kmax)
    dt = CFL_FRACTION * dt_max

    # JIT warm-up (not timed)
    for fn in (step_serialized, step_overlapped):
        reset_state(st)
        fn(ctx, st, st.d_a, st.d_b, dt)
        cuda.synchronize()

    final_s, trace_s, gpu_ms_s, wall_ms_s = run_mode(ctx, st, "serialized", STENCIL_STEPS, dt)
    final_o, trace_o, gpu_ms_o, wall_ms_o = run_mode(ctx, st, "overlapped", STENCIL_STEPS, dt)

    diff = np.zeros(1)
    cart.Allreduce(np.array([float(np.max(np.abs(final_s - final_o)))]), diff, op=MPI.MAX)
    parity = float(diff[0])

    timings = np.zeros(4)
    cart.Allreduce(np.array([gpu_ms_s, gpu_ms_o, wall_ms_s, wall_ms_o]), timings, op=MPI.MAX)

    ptx_counts, ptx_ok = ptx_audit()
    energies = [t["energy"] for t in trace_o]
    monotonic = all(energies[k + 1] < energies[k] for k in range(len(energies) - 1))

    cuda.synchronize()
    cart.Barrier()

    shm_results = benchmark_shm_strides(ctx)
    llvm = llvm_audit_cpu()
    shm_exact = all(v["exact_final_counters"] for v in shm_results.values())

    ok = bool(parity < 1e-12 and ptx_ok and llvm["llvm_ok"] and shm_exact and monotonic
              and ctx.shm_info["view_64byte_aligned"] and ctx.shm_info["all_ranks_same_offset"])

    if ctx.rank == 0:
        s1, s8, s16 = shm_results["stride_1"], shm_results["stride_8"], shm_results["stride_16"]
        manifest = {
            "github_username": ctx.github_user,
            "gpu_uuid": ctx.gpu_uuid,
            "instance_seed": ctx.instance_seed,
            "host": {
                "cpu_model": ctx.cpu_model,
                "cpu_signature": ctx.cpu_sig,
                "cpu_count": os.cpu_count(),
                "cpu_line_bytes": ctx.line_bytes,
                "platform": platform.platform(),
            },
            "mpi": {"ranks": ctx.size, "cartesian_grid": [2, 2], "shared_window": ctx.shm_info},
            "gpu": {
                "global_grid": [GLOBAL_NY, GLOBAL_NX],
                "local_padded_shape": [ctx.ny + 2, ctx.nx + 2],
                "steps": STENCIL_STEPS,
                "dx": ctx.dx,
                "kappa_max": kmax,
                "dt_max": dt_max,
                "dt": dt,
                "lambda": LAMBDA_DAMP,
                "energy_trace": trace_o,
                "energy_strictly_decreasing": monotonic,
                "gpu_serialized_ms": float(timings[0]),
                "gpu_overlapped_ms": float(timings[1]),
                "wall_serialized_ms": float(timings[2]),
                "wall_overlapped_ms": float(timings[3]),
                "parity_max_diff": parity,
                "parity_passed": bool(parity < 1e-12),
                "ptx_counts": ptx_counts,
                "ptx_audit_passed": bool(ptx_ok),
            },
            "shm": {
                "iterations_per_rank": SHM_ITERS,
                "repeats": SHM_REPEATS,
                "results": shm_results,
                "slowdown_stride1_vs_stride8": s1["best_ms"] / s8["best_ms"],
                "ratio_stride16_vs_stride8": s16["best_ms"] / s8["best_ms"],
                "false_sharing_observed": bool(s1["best_ms"] / s8["best_ms"] > 1.3),
                "llvm_audit": llvm,
            },
            "verification_passed": ok,
        }

        print("=" * 76)
        print(f"GITHUB_USERNAME  : {ctx.github_user}")
        print(f"GPU UUID         : {ctx.gpu_uuid}")
        print(f"CPU              : {ctx.cpu_model} ({os.cpu_count()} logical CPUs, line {ctx.line_bytes} B)")
        print(f"INSTANCE_SEED    : {ctx.instance_seed}")
        print(f"MPI              : {ctx.size} ranks, 2 x 2 Cartesian; node window {ctx.shm_info['slots']} x uint64, "
              f"64B-aligned={ctx.shm_info['view_64byte_aligned']}")
        print(f"dx={ctx.dx:.9e} kappa_max={kmax:.9f} dt_max={dt_max:.9e} dt={dt:.9e} ({CFL_FRACTION:.2f} x dt_max)")
        print("-" * 76)
        print("Allreduce energy trace (overlapped):")
        for t in trace_o:
            print(f"  step {t['step']:4d}  E = {t['energy']:.12e}  L2 residual = {t['l2_residual']:.6e}")
        print("Allreduce energy trace (serialized):")
        for t in trace_s:
            print(f"  step {t['step']:4d}  E = {t['energy']:.12e}  L2 residual = {t['l2_residual']:.6e}")
        print(f"Energy strictly decreasing : {monotonic}")
        print(f"gpu_serialized_ms (CUDA events) : {timings[0]:.3f}")
        print(f"gpu_overlapped_ms (CUDA events) : {timings[1]:.3f}")
        print(f"serialized / overlapped (wall)  : {timings[2]:.3f} / {timings[3]:.3f} ms  ({timings[2] / timings[3]:.4f}x)")
        print(f"parity max_diff                 : {parity:.3e}  (< 1e-12 required)")
        print("-" * 76)
        for k, v in ptx_counts.items():
            print(f"PTX {k:18s}: {v}")
        print(f"ptx_audit_passed : {ptx_ok}")
        print("-" * 76)
        print(f"Host shared-memory sweep ({SHM_ITERS:,} updates / rank, best of {SHM_REPEATS}):")
        for key in ("stride_1", "stride_8", "stride_16"):
            v = shm_results[key]
            print(f"  stride={v['stride_elements']:2d} ({v['stride_bytes']:3d} B): {v['best_ms']:10.3f} ms  "
                  f"{v['ns_per_update']:7.3f} ns/update  exact={v['exact_final_counters']}")
        print(f"slowdown stride1/stride8 : {s1['best_ms'] / s8['best_ms']:.3f}x")
        print(f"stride16/stride8         : {s16['best_ms'] / s8['best_ms']:.3f}x")
        print(f"LLVM load i64 / store i64: {llvm['llvm_load_i64_count']} / {llvm['llvm_store_i64_count']} "
              f"(python fallback: {llvm['llvm_python_object_fallback']})  llvm_ok={llvm['llvm_ok']}")
        print(f"verification_passed      : {ok}")
        print("=" * 76)

        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {MANIFEST_PATH}")

    ctx.win.Free()
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: hybrid_mpi_cuda_cache.py <github_username>")
    sys.exit(run_hybrid(sys.argv[1]))
