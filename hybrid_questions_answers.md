### Analytical & Forensic Engineering Responses

**Question 1 (GPU Coalescing vs. CPU Cache-Line Granularity)**

*Question:* Contrast how a 64-byte host CPU cache line handles strided access against how an NVIDIA T4 GPU handles strided column reads (u[1:257, 1], 2,064-byte row stride) in 32-byte L2 memory sectors. Quantify the wasted PCIe/DRAM transfer overhead avoided by pack_halos_kernel.

**Answer:** The array has a row pitch of 258 doubles = 2,064 bytes, so the cells u[1:257,1] of one column are 2,064 bytes apart. A CPU 64-byte line holds 8 doubles, so a column walk touches a new line for every element (256 lines = 16,384 B moved for 2,048 B useful, 12.5% used), but a row walk uses all 8 words of each line. On the T4 a warp reads in 32-byte sectors (4 per 128-byte L1 line); 32 threads reading 32 rows of one column touch 32 different sectors = 1,024 B for 256 useful B (25%). For a whole column: 256 sectors x 32 B = 8,192 B of DRAM/L2 traffic for 2,048 B of data, i.e. 6,144 B wasted per column. If a strided column were sent by one DMA copy over PCIe, the bounding box would be (255 x 258 + 1) x 8 = 526,328 B instead of 2,048 B, i.e. 524,280 B wasted per column (1,048,560 B for both left/right columns). pack_halos_kernel gathers the columns into 2,048-byte contiguous buffers on the GPU, so this overhead is avoided.

---

**Question 2 (Shared-Memory Tile Synchronization & PTX Audit)**

*Question:* Report your exact counts of st.shared.f64, ld.shared.f64, and bar.sync from tiled_stencil_kernel.inspect_asm(). Explain what Read-After-Write hazard occurs across the 8 warps of a 16 x 16 threadblock if cuda.syncthreads() is removed.

**Answer:** tiled_stencil_kernel.inspect_asm() gives st.shared.f64 = 5, ld.shared.f64 = 5, bar.sync = 1. The 5 stores are the center cell and the 4 apron cells, the 5 loads are center/up/down/left/right, and the single bar.sync is cuda.syncthreads(). A 16x16 block has 8 warps that do not run in lockstep. Without syncthreads there is a read-after-write hazard: a warp can read a tile cell (for example the row below or the apron) before the other warp that owns it has written it, so it reads stale or uninitialized shared memory and the stencil gives wrong, nondeterministic values.

---

**Question 3 (Dual-Stream Compute-Communication Overlap on Shared GPU Silicon)**

*Question:* Compare your measured gpu_serialized_ms and gpu_overlapped_ms values. Explain why four MPI processes sharing a single physical GPU context serialize PCIe DMA queues differently than four dedicated GPUs connected via GPUDirect RDMA.

**Answer:** Measured gpu_serialized_ms = 1288.760 and gpu_overlapped_ms = 2813.189 (CUDA events, slowest rank, 300 steps); wall time was 1289.665 vs 2813.765 ms. So in my run the overlap was not faster: it was 2.18x slower (serialized/overlapped = 0.458x, about 1524 ms lost). The 4 MPI processes each own a CUDA context on the same T4, so the GPU time-slices between contexts: kernels from different processes do not run at the same time, and all D2H/H2D copies share the same PCIe link and copy engines. This means there is no free GPU capacity for stream_interior to use while MPI is waiting. The halos are tiny (2,048 B per side), so there is very little communication to hide, but overlapped mode adds cost: 5 stencil launches per step instead of 1, two event record/wait pairs and more context switches between the processes; the 2-core Colab CPU running 4 ranks also competes for the host launch work. (These are the likely reasons; I did not profile them.) With four dedicated GPUs and GPUDirect RDMA each rank has its own SMs and copy engines and the NIC reads GPU memory directly, so the transfer runs in parallel with the interior stencil and overlap can really pay off.

---

**Question 4 (MPI-3 Shared-Memory MESI RFO Ping-Pong)**

*Question:* In your MPI.Win.Allocate_shared benchmark, report your measured latencies for stride = 1 (8 B), stride = 8 (64 B), and stride = 16 (128 B), along with the false-sharing slowdown ratio. Trace the exact MESI state transitions (M, E, S, I) when Rank 0 writes to shm_view[0] while Rank 1 writes to shm_view[1].

**Answer:** Measured over 40,000,000 updates per rank (slowest rank, best of 3): stride=1 (8 B) = 449.070 ms (11.227 ns/update), stride=8 (64 B) = 241.949 ms (6.049 ns/update), stride=16 (128 B) = 248.549 ms (6.214 ns/update). Slowdown ratio stride=1 / stride=8 = 1.856x. False sharing is visible: stride=1 is 1.86x slower than stride=8. MESI trace for shm_view[0] (Rank 0) and shm_view[1] (Rank 1), which are in the same 64-byte line: (1) Rank 0 writes: its copy is I (or E), it sends an RFO, the line becomes M in Rank 0 and I in every other core. (2) Rank 1 writes shm_view[1]: its copy is I, so it sends an RFO; Rank 0 holds the line in M, writes it back / forwards it and goes M -> I, and Rank 1 goes I -> M. (3) Rank 0 writes again: now Rank 0 is I and Rank 1 is M, so another RFO moves the line back (Rank 1 M -> I, Rank 0 I -> M). This repeats on every store (ping-pong); E or S appear only if a core just reads a line. At stride=8 the two counters are in different lines, so each stays M in its own core and no RFO crosses cores.

---

**Question 5 (128-Byte Spatial Prefetcher Pairing & Padding Trade-Offs)**

*Question:* Compare your measured timings between stride = 8 (64-byte padding) and stride = 16 (128-byte padding). Explain how hardware adjacent-cache-line prefetchers pair even/odd 64-byte lines into 128-byte sectors and why excessive padding can induce L1/L2 capacity misses.

**Answer:** Measured stride=8 (64 B padding) = 241.949 ms and stride=16 (128 B padding) = 248.549 ms, ratio stride16/stride8 = 1.027. The L2 adjacent-cache-line (spatial) prefetcher treats an even 64-byte line and the following odd line as one 128-byte sector and fetches the buddy line together with the requested one. With stride=8, counters of ranks 0 and 1 sit in the two lines of one 128-byte pair (and ranks 2 and 3 in the next pair), so a prefetch of the buddy line can pull a line another core is writing and cause extra invalidations; stride=16 puts every rank in its own 128-byte sector and removes that. Excessive padding has a cost too: each counter uses 128 B for 8 useful bytes (6.25% utilization), so the same data touches more lines, takes more L1/L2 sets and TLB entries, and with a lot of counters it causes capacity misses and wastes cache bandwidth; with only 4 counters this cost is negligible.

---

**Question 6 (CFL Stability & Distributed Energy Dissipation)**

*Question:* Using your manifest's kappa_max and dx = 1/513, compute the 2D explicit stability limit dt_max = dx^2 / (4 * kappa_max), verify that your simulation's dt satisfies the Courant condition, and cite your printed allreduce global energy values across steps 0, 100, 200, and 300.

**Answer:** dx = 1/513 = 1.949317739e-03 and kappa_max = 0.239999322, so dt_max = dx^2 / (4 kappa_max) = 3.958177474e-06. The simulation used dt = 3.166541979e-06 = 0.80 x dt_max, and dt < dt_max, so the Courant (CFL) condition is satisfied. The Allreduce global energy at steps 0, 100, 200, 300 is 8.155742716082e-02, 8.153753976889e-02, 8.152433089991e-02, 8.151202786649e-02. It decreases at every report (cubic damping -lambda u^3 and zero Dirichlet edges remove energy), with no growth, so the scheme is stable and dissipative. Serialized and overlapped fields agree with max_diff = 0.000e+00 (< 1e-12).

---
