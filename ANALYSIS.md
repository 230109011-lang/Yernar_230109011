# ANALYSIS

Run: `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`, 16 logical cores, cache line 64 B, 4 threads x 60,000,000 updates, best of 3 runs.

## Question 1 (MESI Protocol State Transitions)

With stride=1, buffer[0] and buffer[1] are in the same 64-byte line. Sequence:
1. Core 0 stores buffer[0]: line is I (or E) in Core 0, so it issues an RFO; the line becomes **M** in Core 0 and **I** in every other cache.
2. Core 1 stores buffer[1] at once: its copy is **I**, so it issues an RFO. Core 0 holds the line in M, so it must supply/write back the data and goes **M -> I**; Core 1 gets the line **I -> M**.
3. Core 0 stores again: now its copy is I, so it sends an RFO and takes the line back (**M -> I** in Core 1, **I -> M** in Core 0). This repeats on every store (ping-pong). The state E appears only when one core reads a line nobody else has; S appears if a core only reads it.

Why the RFO stalls the store buffer: x86 (TSO) must commit stores to L1 in program order, and a store can only commit when the line is in M or E. The store at the head of the store buffer waits for the RFO to finish (tens of ns across cores), so all younger stores queue behind it. When the store buffer is full the core stops issuing new stores and the pipeline stalls.

## Question 2 (Quantitative Slowdown & Coherency Overhead)

| Layout | stride | best time (ns) | ns / update / thread | ns / update (aggregate, 4 threads) |
|---|---|---|---|---|
| A contiguous | 1 (8 B) | 1,417,571,000 | 23.626 | 5.907 |
| B half-line | 4 (32 B) | 624,243,100 | 10.404 | 2.601 |
| C line-padded | 8 (64 B) | 53,409,100 | 0.890 | 0.223 |
| D prefetch-safe | 16 (128 B) | 55,671,200 | 0.928 | 0.232 |
| private register | one store | 53,676,900 | 0.895 | 0.224 |

Slowdown of stride=1 relative to stride=8: 1,417,571,000 / 53,409,100 = **26.54x**. stride=1 was 26.54x slower than stride=8, which is the false-sharing penalty on this machine. Stride=4 vs stride=8: 11.69x. stride=8 vs private register: 1.00x (with no cache-line sharing at stride=8 this gap is mainly the L1 store-to-load forwarding latency of a memory read-modify-write every iteration, versus a register add, and not coherence). Context switches (best run): stride=1 0 voluntary / 0 involuntary; stride=8 0 / 0.

## Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)

Measured stride=8 (64 B spacing): 53,409,100 ns; stride=16 (128 B spacing): 55,671,200 ns (ratio 1.042). The adjacent-cache-line prefetcher treats two 64-byte lines (an even line and the following odd line, aligned to 128 B) as one 128-byte sector, and when one line of the pair is fetched it may also fetch its buddy. With stride=8 the slots are in lines 0,1,2,3: threads 0 and 1 are in one 128-byte pair, and threads 2 and 3 in another, so a prefetch of the buddy line can pull a line another core is writing and force extra invalidations. With stride=16 every thread has its own 128-byte sector, so this cannot happen. 64-byte padding can therefore still show a small secondary penalty when the neighbouring line in the same 128-byte pair is written by another core; on CPUs without this prefetcher (or with it disabled) the two cases are equal.

## Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)

The LLVM IR of `worker_strided_accumulate` has 12 `load i64` and 11 `store i64` instructions and no Python-object calls (python fallback: False), so it is native code. An optimizing compiler can hoist `buffer[slot]` into a register (loop-invariant memory promotion, LICM): it loads once before the loop, adds in the register, and stores once after the loop. That would remove all memory traffic and false sharing would disappear. To prevent it, the loop stores through `buffer[j]` with `j = slot + z`, where `z = buffer[slot] >> 63` is read from memory before the loop (it is 0 at run time, but unknown at compile time), and loads through `buffer[slot]` in the next iteration. (My first attempt used `z = iters >> 63`; LLVM proved it was 0 and collapsed the whole loop to one addition, so the zero had to come from memory.) LLVM cannot prove the two addresses are different or equal, so the load must be re-issued after each store, and each iteration is a real L1 load + store (a loop-carried dependency through memory). Example IR lines from this run: `%.53 = load i64, ptr %.52, align 8`; `%.223.prol = load i64, ptr %sunkaddr2, align 8`; `store i64 %.224.prol, ptr %sunkaddr4, align 8`; `%.223 = load i64, ptr %sunkaddr6, align 8`.

## Question 5 (CPython GIL Masking vs. nogil=True Execution)

Pure-Python threads run bytecode only while holding the GIL, so only one thread executes at any time; they never store to the shared line simultaneously from different cores, and there is no cache-line ping-pong to measure. In my GIL demo (3,000,000 increments per thread) 1 thread took 441,595,200 ns and 4 threads took 1,765,597,600 ns (4.00x), but the work done is 4x larger, so this time is GIL serialization plus lock hand-off/switching overhead (threads waiting and being woken), not coherency traffic. GIL contention is software lock contention in the interpreter (one thread runs, the others sleep), while cache-line contention is a hardware effect between cores that really run in parallel. With `nogil=True` the Numba kernels release the GIL, run on separate cores at the same time, and expose the hardware effect.

Manifest: `cache_lab_manifest.json` (verification_passed = True, bit-exact parity = True).
