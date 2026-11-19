import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:  # Windows
    resource = None

# Replace with your exact Git handle before executing (or pass it as argv[1])
GIT_USERNAME = "230109011-lang"

NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)           # elements of 8 bytes -> 8 B, 32 B, 64 B, 128 B
LAYOUT_NAMES = {1: "A_contiguous", 4: "B_half_line", 8: "C_line_padded", 16: "D_prefetch_safe"}
REPEATS = 3                       # each layout is timed REPEATS times, best run is reported
GIL_DEMO_ITERS = 3_000_000        # pure-Python (GIL) demo for Question 5
MANIFEST_PATH = "cache_lab_manifest.json"
ANALYSIS_PATH = "ANALYSIS.md"


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    """Best-effort CPU model string."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def pick_cpus(n: int):
    """Pick n logical CPUs, preferring one hardware thread per physical core."""
    if not hasattr(os, "sched_getaffinity"):
        return None, False
    avail = sorted(os.sched_getaffinity(0))
    seen, primary = set(), []
    for c in avail:
        key = c
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                key = f.read().strip()
        except OSError:
            pass
        if key not in seen:
            seen.add(key)
            primary.append(c)
    order = primary + [c for c in avail if c not in primary]
    cpus = [order[i % len(order)] for i in range(n)]
    return cpus, len(primary) >= n


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{cpu_model}::{logical_cores}"

    # Deterministic 64-bit seed: SHA-256(git_user :: cpu_sig), first 16 hex digits.
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # Core affinity masks (Linux only; best effort elsewhere).
    cpus, distinct_physical = pick_cpus(NUM_THREADS)
    affinity_supported = cpus is not None

    return {
        "git_username": git_user.strip(),
        "cpu_model": cpu_model,
        "cpu_signature": cpu_sig,
        "instance_seed": instance_seed,
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "elements_per_line": line_size // 8,
        "affinity_supported": affinity_supported,
        "cpus": cpus,
        "distinct_physical_cores": bool(distinct_physical),
        "platform": platform.platform(),
    }


def aligned_zeros(n: int, align: int = 64) -> np.ndarray:
    """uint64 zero array whose data pointer is aligned to `align` bytes."""
    raw = np.zeros(n * 8 + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    arr = raw[offset: offset + n * 8].view(np.uint64)
    assert arr.ctypes.data % align == 0
    return arr


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    # `z` is 0 at run time (the counter never reaches 2**63) but it is read from memory,
    # so the compiler cannot know it. The store goes through index j = slot + z while the
    # next iteration loads buffer[slot]. LLVM can neither prove j != slot nor j == slot, so
    # it must re-load buffer[slot] and store to memory on every iteration: no hoisting of
    # the accumulator into a register and no closed-form `buffer[slot] += iters`.
    z = np.int64(buffer[slot] >> uint64(63))
    zu = uint64(z)
    j = slot + z
    x = uint64(seed)
    for _ in range(iters):
        x = x * uint64(6364136223846793005) + uint64(1442695040888963407)
        inc = uint64(1) + (x & zu)          # always 1 at run time; opaque to the compiler
        buffer[j] = buffer[slot] + inc      # read-modify-write directly on buffer memory
    return buffer[slot]


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    zu = buffer[slot] >> uint64(63)         # 0 at run time, opaque to the compiler
    x = uint64(seed)
    acc = uint64(0)
    for _ in range(iters):
        x = x * uint64(6364136223846793005) + uint64(1442695040888963407)
        acc += uint64(1) + (x & zu)         # accumulator lives in a register
    buffer[slot] = acc                      # single shared-memory write
    return acc


def _rusage_switches():
    if resource is None:
        return 0, 0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_nvcsw, ru.ru_nivcsw


def _run_once(kernel, buffer, stride, iters, ctx, seeds):
    barrier = threading.Barrier(NUM_THREADS + 1)
    cpus = ctx["cpus"]

    def target(t):
        if cpus is not None:
            try:
                os.sched_setaffinity(0, {cpus[t]})  # pid 0 = calling thread on Linux
            except OSError:
                pass
        barrier.wait()
        kernel(buffer, t * stride, iters, np.uint64(seeds[t]))

    threads = [threading.Thread(target=target, args=(t,)) for t in range(NUM_THREADS)]
    for th in threads:
        th.start()
    v0, i0 = _rusage_switches()
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    t1 = time.perf_counter_ns()
    v1, i1 = _rusage_switches()
    return t1 - t0, v1 - v0, i1 - i0


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    seeds = [(ctx["instance_seed"] + 0x9E3779B97F4A7C15 * (t + 1)) % (1 << 64) for t in range(NUM_THREADS)]

    # JIT warm-up so compilation is not timed
    warm = aligned_zeros(NUM_THREADS * 16 + 16)
    _run_once(worker_strided_accumulate, warm, 1, 2000, ctx, seeds)
    _run_once(worker_private_accumulate, warm, 1, 2000, ctx, seeds)

    results, finals = {}, {}
    for stride in STRIDES:
        times, nv, ni = [], [], []
        final = None
        for _ in range(REPEATS):
            buf = aligned_zeros(NUM_THREADS * stride + 16)
            ns, dv, di = _run_once(worker_strided_accumulate, buf, stride, iters, ctx, seeds)
            times.append(ns); nv.append(dv); ni.append(di)
            final = [int(buf[t * stride]) for t in range(NUM_THREADS)]
        best = min(times)
        results[LAYOUT_NAMES[stride]] = {
            "stride_elements": stride,
            "stride_bytes": stride * 8,
            "best_ns": best,
            "all_ns": times,
            "ns_per_update_per_thread": best / iters,
            "ns_per_update_aggregate": best / (iters * NUM_THREADS),
            "voluntary_ctx_switches": nv[times.index(best)],
            "involuntary_ctx_switches": ni[times.index(best)],
        }
        finals[LAYOUT_NAMES[stride]] = final

    # Task 4 baseline (thread-local register accumulation, adjacent slots, one store each)
    times, nv, ni = [], [], []
    final = None
    for _ in range(REPEATS):
        buf = aligned_zeros(NUM_THREADS + 16)
        ns, dv, di = _run_once(worker_private_accumulate, buf, 1, iters, ctx, seeds)
        times.append(ns); nv.append(dv); ni.append(di)
        final = [int(buf[t]) for t in range(NUM_THREADS)]
    best = min(times)
    results["private_register"] = {
        "stride_elements": 1,
        "stride_bytes": 8,
        "best_ns": best,
        "all_ns": times,
        "ns_per_update_per_thread": best / iters,
        "ns_per_update_aggregate": best / (iters * NUM_THREADS),
        "voluntary_ctx_switches": nv[times.index(best)],
        "involuntary_ctx_switches": ni[times.index(best)],
    }
    finals["private_register"] = final
    return {"results": results, "finals": finals}


# ==============================================================================
# Question 5 helper: same stride=1 experiment with plain Python bytecode (GIL held)
# ==============================================================================
def _py_worker(buffer, slot, iters):
    for _ in range(iters):
        buffer[slot] += 1


def gil_demo(ctx: dict, iters: int = GIL_DEMO_ITERS) -> dict:
    out = {}
    for n_threads in (1, NUM_THREADS):
        buf = aligned_zeros(NUM_THREADS + 16)
        threads = [threading.Thread(target=_py_worker, args=(buf, t, iters)) for t in range(n_threads)]
        t0 = time.perf_counter_ns()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        out[f"threads_{n_threads}_ns"] = time.perf_counter_ns() - t0
    out["iters_per_thread"] = iters
    out["slowdown_4_vs_1_threads"] = out[f"threads_{NUM_THREADS}_ns"] / out["threads_1_ns"]
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def audit_llvm_and_parity(bench: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    finals = bench["finals"]
    expected = [iters] * NUM_THREADS
    per_layout_ok = {k: (v == expected) for k, v in finals.items()}
    layouts = list(finals.values())
    bit_exact = all(v == layouts[0] for v in layouts) and all(per_layout_ok.values())

    ir = worker_strided_accumulate.inspect_llvm()
    full = "\n".join(ir.values()) if isinstance(ir, dict) else str(ir)
    # Keep only the native kernel body; Numba also emits a CPython wrapper (`@_ZN7cpython...`)
    # that legitimately calls PyArg_*/PyLong_* to unbox the arguments.
    parts = full.split("\ndefine ")
    native = [p for p in parts[1:] if not p.startswith(("ptr @_ZN7cpython", "ptr @\"_ZN7cpython"))
              and "@_ZN7cpython" not in p.split("(", 1)[0]]
    text = "\n".join(native) if native else full
    store_i64 = text.count("store i64")
    load_i64 = text.count("load i64")
    python_fallback = any(tok in text for tok in ("PyObject", "PyArg_Parse", "pyapi", "PyLong"))
    snippet = [ln.strip() for ln in text.splitlines()
               if ("store i64" in ln or "load i64" in ln) and "%" in ln][:8]
    llvm_ok = store_i64 > 0 and load_i64 > 0 and not python_fallback

    return {
        "expected_counter_per_thread": iters,
        "final_counters": finals,
        "per_layout_exact": per_layout_ok,
        "bit_exact_all_layouts": bool(bit_exact),
        "llvm_store_i64_count": store_i64,
        "llvm_load_i64_count": load_i64,
        "llvm_python_object_fallback": bool(python_fallback),
        "llvm_ok": bool(llvm_ok),
        "llvm_snippet": snippet,
    }


def build_manifest(ctx, bench, audit, gil) -> dict:
    r = bench["results"]
    s1, s4, s8, s16, priv = (r["A_contiguous"], r["B_half_line"], r["C_line_padded"],
                             r["D_prefetch_safe"], r["private_register"])
    slowdown_1_vs_8 = s1["best_ns"] / s8["best_ns"]
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "platform": ctx["platform"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "affinity_supported": ctx["affinity_supported"],
        "cpus_used": ctx["cpus"],
        "distinct_physical_cores": ctx["distinct_physical_cores"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats_per_layout": REPEATS,
        "results": r,
        "slowdown_stride1_vs_stride8": slowdown_1_vs_8,
        "slowdown_stride4_vs_stride8": s4["best_ns"] / s8["best_ns"],
        "ratio_stride16_vs_stride8": s16["best_ns"] / s8["best_ns"],
        "ratio_stride8_vs_private": s8["best_ns"] / priv["best_ns"],
        "ratio_stride1_vs_private": s1["best_ns"] / priv["best_ns"],
        "false_sharing_observed": bool(slowdown_1_vs_8 > 1.3),
        "audit": audit,
        "gil_demo": gil,
    }
    manifest["verification_passed"] = bool(audit["bit_exact_all_layouts"] and audit["llvm_ok"])
    return manifest


def write_analysis(m: dict, path: str = ANALYSIS_PATH) -> None:
    r = m["results"]
    a, b, c, d, p = (r["A_contiguous"], r["B_half_line"], r["C_line_padded"],
                     r["D_prefetch_safe"], r["private_register"])
    g = m["gil_demo"]
    au = m["audit"]
    nthr = m["num_threads"]
    t_n = g[f"threads_{nthr}_ns"]
    ratio = m["slowdown_stride1_vs_stride8"]
    if m["false_sharing_observed"]:
        effect = (f"stride=1 was {ratio:.2f}x slower than stride=8, which is the false-sharing "
                  f"penalty on this machine.")
    else:
        effect = (f"stride=1 was only {ratio:.2f}x relative to stride=8, so false sharing was weak "
                  f"on this machine (possible reasons: threads share an SMT core or L2, a VM, or "
                  f"affinity not applied).")
    lines = [
        "# ANALYSIS",
        "",
        f"Run: `{m['cpu_model']}`, {m['logical_cores']} logical cores, cache line {m['cache_line_size']} B, "
        f"{m['num_threads']} threads x {m['iterations_per_thread']:,} updates, best of {m['repeats_per_layout']} runs.",
        "",
        "## Question 1 (MESI Protocol State Transitions)",
        "",
        "With stride=1, buffer[0] and buffer[1] are in the same 64-byte line. Sequence:",
        "1. Core 0 stores buffer[0]: line is I (or E) in Core 0, so it issues an RFO; the line becomes **M** in Core 0 and **I** in every other cache.",
        "2. Core 1 stores buffer[1] at once: its copy is **I**, so it issues an RFO. Core 0 holds the line in M, so it must supply/write back the data and goes **M -> I**; Core 1 gets the line **I -> M**.",
        "3. Core 0 stores again: now its copy is I, so it sends an RFO and takes the line back (**M -> I** in Core 1, **I -> M** in Core 0). This repeats on every store (ping-pong). The state E appears only when one core reads a line nobody else has; S appears if a core only reads it.",
        "",
        "Why the RFO stalls the store buffer: x86 (TSO) must commit stores to L1 in program order, and a store can only commit when the line is in M or E. The store at the head of the store buffer waits for the RFO to finish (tens of ns across cores), so all younger stores queue behind it. When the store buffer is full the core stops issuing new stores and the pipeline stalls.",
        "",
        "## Question 2 (Quantitative Slowdown & Coherency Overhead)",
        "",
        "| Layout | stride | best time (ns) | ns / update / thread | ns / update (aggregate, 4 threads) |",
        "|---|---|---|---|---|",
        f"| A contiguous | 1 (8 B) | {a['best_ns']:,} | {a['ns_per_update_per_thread']:.3f} | {a['ns_per_update_aggregate']:.3f} |",
        f"| B half-line | 4 (32 B) | {b['best_ns']:,} | {b['ns_per_update_per_thread']:.3f} | {b['ns_per_update_aggregate']:.3f} |",
        f"| C line-padded | 8 (64 B) | {c['best_ns']:,} | {c['ns_per_update_per_thread']:.3f} | {c['ns_per_update_aggregate']:.3f} |",
        f"| D prefetch-safe | 16 (128 B) | {d['best_ns']:,} | {d['ns_per_update_per_thread']:.3f} | {d['ns_per_update_aggregate']:.3f} |",
        f"| private register | one store | {p['best_ns']:,} | {p['ns_per_update_per_thread']:.3f} | {p['ns_per_update_aggregate']:.3f} |",
        "",
        f"Slowdown of stride=1 relative to stride=8: {a['best_ns']:,} / {c['best_ns']:,} = **{ratio:.2f}x**. {effect} "
        f"Stride=4 vs stride=8: {m['slowdown_stride4_vs_stride8']:.2f}x. "
        f"stride=8 vs private register: {m['ratio_stride8_vs_private']:.2f}x (with no cache-line sharing at stride=8 this gap is mainly the L1 store-to-load forwarding latency of a memory read-modify-write every iteration, versus a register add, and not coherence). "
        f"Context switches (best run): stride=1 {a['voluntary_ctx_switches']} voluntary / {a['involuntary_ctx_switches']} involuntary; "
        f"stride=8 {c['voluntary_ctx_switches']} / {c['involuntary_ctx_switches']}.",
        "",
        "## Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)",
        "",
        f"Measured stride=8 (64 B spacing): {c['best_ns']:,} ns; stride=16 (128 B spacing): {d['best_ns']:,} ns "
        f"(ratio {m['ratio_stride16_vs_stride8']:.3f}). The adjacent-cache-line prefetcher treats two 64-byte lines "
        "(an even line and the following odd line, aligned to 128 B) as one 128-byte sector, and when one line of the pair is fetched it may also fetch its buddy. "
        "With stride=8 the slots are in lines 0,1,2,3: threads 0 and 1 are in one 128-byte pair, and threads 2 and 3 in another, so a prefetch of the buddy line can pull a line another core is writing and force extra invalidations. "
        "With stride=16 every thread has its own 128-byte sector, so this cannot happen. 64-byte padding can therefore still show a small secondary penalty when the neighbouring line in the same 128-byte pair is written by another core; on CPUs without this prefetcher (or with it disabled) the two cases are equal.",
        "",
        "## Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)",
        "",
        f"The LLVM IR of `worker_strided_accumulate` has {au['llvm_load_i64_count']} `load i64` and {au['llvm_store_i64_count']} `store i64` instructions and no Python-object calls (python fallback: {au['llvm_python_object_fallback']}), so it is native code. "
        "An optimizing compiler can hoist `buffer[slot]` into a register (loop-invariant memory promotion, LICM): it loads once before the loop, adds in the register, and stores once after the loop. That would remove all memory traffic and false sharing would disappear. "
        "To prevent it, the loop stores through `buffer[j]` with `j = slot + z`, where `z = buffer[slot] >> 63` is read from memory before the loop (it is 0 at run time, but unknown at compile time), and loads through `buffer[slot]` in the next iteration. (My first attempt used `z = iters >> 63`; LLVM proved it was 0 and collapsed the whole loop to one addition, so the zero had to come from memory.) LLVM cannot prove the two addresses are different or equal, so the load must be re-issued after each store, and each iteration is a real L1 load + store (a loop-carried dependency through memory). "
        "Example IR lines from this run: " + "; ".join(f"`{s}`" for s in au["llvm_snippet"][:4]) + ".",
        "",
        "## Question 5 (CPython GIL Masking vs. nogil=True Execution)",
        "",
        f"Pure-Python threads run bytecode only while holding the GIL, so only one thread executes at any time; they never store to the shared line simultaneously from different cores, and there is no cache-line ping-pong to measure. "
        f"In my GIL demo ({g['iters_per_thread']:,} increments per thread) 1 thread took {g['threads_1_ns']:,} ns and {nthr} threads took {t_n:,} ns "
        f"({g['slowdown_4_vs_1_threads']:.2f}x), but the work done is {m['num_threads']}x larger, so this time is GIL serialization plus lock hand-off/switching overhead (threads waiting and being woken), not coherency traffic. "
        "GIL contention is software lock contention in the interpreter (one thread runs, the others sleep), while cache-line contention is a hardware effect between cores that really run in parallel. With `nogil=True` the Numba kernels release the GIL, run on separate cores at the same time, and expose the hardware effect.",
        "",
        f"Manifest: `{MANIFEST_PATH}` (verification_passed = {m['verification_passed']}, bit-exact parity = {au['bit_exact_all_layouts']}).",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    user = sys.argv[1] if len(sys.argv) > 1 else GIT_USERNAME
    ctx = init_system_context(user)
    print(f"Git user          : {ctx['git_username']}")
    print(f"CPU               : {ctx['cpu_model']} ({ctx['logical_cores']} logical cores)")
    print(f"Cache line size   : {ctx['cache_line_size']} B ({ctx['elements_per_line']} x int64 per line)")
    print(f"INSTANCE_SEED     : {ctx['instance_seed']}")
    print(f"Affinity          : supported={ctx['affinity_supported']} cpus={ctx['cpus']} "
          f"distinct_physical_cores={ctx['distinct_physical_cores']}")
    print("Running stride sweep (this can take a while)...", flush=True)

    bench = run_stride_benchmark(ctx)
    audit = audit_llvm_and_parity(bench)
    gil = gil_demo(ctx)
    manifest = build_manifest(ctx, bench, audit, gil)

    print("-" * 78)
    print(f"{'layout':18s}{'stride':>8s}{'best ns':>16s}{'ns/upd/thread':>16s}{'vcsw':>8s}{'ivcsw':>8s}")
    for name, v in manifest["results"].items():
        print(f"{name:18s}{v['stride_elements']:>8d}{v['best_ns']:>16,d}{v['ns_per_update_per_thread']:>16.3f}"
              f"{v['voluntary_ctx_switches']:>8d}{v['involuntary_ctx_switches']:>8d}")
    print(f"slowdown stride1 / stride8 : {manifest['slowdown_stride1_vs_stride8']:.3f}x")
    print(f"stride16 / stride8         : {manifest['ratio_stride16_vs_stride8']:.3f}x")
    print(f"stride8 / private          : {manifest['ratio_stride8_vs_private']:.3f}x")
    print(f"bit-exact parity           : {audit['bit_exact_all_layouts']}")
    print(f"LLVM load i64 / store i64  : {audit['llvm_load_i64_count']} / {audit['llvm_store_i64_count']}  "
          f"(python fallback: {audit['llvm_python_object_fallback']})")
    print(f"GIL demo 1 vs {NUM_THREADS} threads    : {gil['threads_1_ns']:,} ns vs {gil[f'threads_{NUM_THREADS}_ns']:,} ns")
    print(f"verification_passed        : {manifest['verification_passed']}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_analysis(manifest)
    print(f"Wrote {MANIFEST_PATH} and {ANALYSIS_PATH}")
    return 0 if manifest["verification_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
