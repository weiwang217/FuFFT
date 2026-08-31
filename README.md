# fuFFT — fused FFT convolution kernels for CUDA

CUDA kernels that run a cyclic convolution — forward FFT, pointwise multiply,
inverse FFT — **in a single kernel launch**, instead of the three launches and
seven HBM passes the cuFFT host API needs. One kernel, three passes over
memory, and the difference shows up directly in throughput.

## Performance (measured on an NVIDIA A10, sm_86)

| benchmark | speedup |
|---|---|
| fp32 **vs cuFFT host API** (ExecC2C → multiply → ExecC2C), N = 512…4096 | **2.27–2.36× faster** |
| **int16 block-floating-point mode** vs that same cuFFT fp32 chain | **4.3–4.5× faster** |
| int16-BFP vs cuFFT's own fp16 chain (`cufftXt`, half in/out) | **2.3× faster, 35–55× more accurate** |
| int16-BFP vs cuFFTDx's fused fp32 kernel | **1.9× faster** |
| int16-BFP vs cuFFTDx **fp16** (both 4 B/pt) | **speed parity (1.00×), 45–80× more accurate** |
| radix-16 sixteen-values-per-lane kernel vs our radix-8 baseline | **1.25–1.43× faster** |
| large N = 32768 (four-step decomposition) vs cuFFT | **1.50× faster** |

Accuracy: rel L2 ≈ 1e-06 against cuFFT at every fp32 size; the int16-BFP path
holds ≈ 4e-05 with input quantisation included.

Backends: every row above holds on all three (`-DFUFFT_BACKEND=0/1/2`).
The inline-PTX and C++ builds tie on every kernel — their times differ by
less than the benchmark's own ±4% run-to-run jitter, with no consistent
winner and bit-identical rel L2. ptxas already emits the same SASS the
hand-written PTX asks for (`make sass` shows this side by side).

Against NVIDIA's device-side cuFFTDx, the fp32 kernels measure 1–1.05× —
on par with the vendor's fused library (`make run-dx` reproduces this).

Why the fusion wins: the cuFFT host chain moves every point over HBM seven
times (2 + 3 + 2); the fused kernel moves it three times (read signal, read
filter spectrum, write result). Both sides run at the bandwidth wall, so the
pass count is the whole story — the measured speedups match the pass-count
model to within 0.002×.

The 16-bit mode stacks a second ≈2× on top: int16 with a shared per-block
exponent halves the bytes per point, and at the bandwidth wall bytes are time.
The math inside the kernel stays fp32, and block floating point keeps 15
significant bits where fp16 storage keeps 11 — that is the whole trade.
Against cuFFTDx's fp16 mode, the like-for-like 4-byte contender, the BFP
kernel runs at speed parity — 1.00× at N ≤ 4096 — while fp16's error
compounds through the stages: rel L2 1.7e-03 at N = 512 growing to 3.3e-03
at N = 4096, where block floating point holds ≈ 4e-05 flat. Same traffic,
45–80× less error.
cuFFT's own half mode (`cufftXt`, CUDA_C_16F end to end) is still a 3-launch,
7-pass chain, so the fused kernel beats it ≈2.3× on top of the accuracy gap.
`make run-gpu` prints the cuFFT rows; `make run-dx` adds the cuFFTDx ones.

## What's inside

- **In-place DIF/DIT with digit reversal never materialised.** The forward
  transform leaves the spectrum digit-reversed, the filter spectrum is
  permuted once on the host, and the inverse returns natural order — no
  reorder pass ever runs on the GPU. Shared memory is half of a ping-pong
  layout, which triples resident blocks on sm_86.
- **Radix 16 at sixteen values per lane** (`V=16`): one fewer pass at almost
  every size with half the lanes — the source of the 1.25–1.43× kernel-level
  gain and the headline numbers.
- **Conflict-free shared memory** via the bijection `i ^ ((i>>3) & 15)`:
  every gather, scatter and linear access is 1-way, with zero padding cost.
- **Four-step decomposition** for N beyond one block: outer flat passes plus
  a fused middle kernel, radix 8 or 16 per level.
- **Twiddle recurrence**: `W^j` computed once, `W^2j…W^7j` by multiplication —
  ~2.5× cheaper than per-twiddle `__sincosf` on the MUFU pipe.
- **Composable pipeline**: the mid-operation (multiply / conjugate-multiply /
  nothing) is a policy struct, so a new fusion costs a struct, not a kernel.
  2D convolution (rows × cols fused), two-for-one real-signal packing, int16
  block-floating-point storage, and split forward/apply entry points for the
  one-signal-many-filters case are all built this way.
- **Three arithmetic backends** (`-DFUFFT_BACKEND=0/1/2`): naive C++,
  algebraically specialised C++, and inline PTX — same butterfly graph, same
  memory layout, so the value of hand-written assembly is measurable rather
  than assumed. On the shipping bandwidth-saturated kernels the three tie,
  which is itself the finding: the results do not depend on assembly.

## Verified before it ever ran on a GPU

- `verify_kernel.cpp` — a line-by-line CPU transcription of the device code
  (slot/lane decomposition, ping-pong parity, base offsets, every variant
  including 2D, real-packing and the full four-step space): **78 checks**
  against fp64 references, several of them bit-identity.
- `audit_ptx.py` — static validation of all inline PTX: opcodes, operand
  arity and range, and the `"=&f"` early-clobber hazard.
- `audit_smem.py` — proves the swizzle is a bijection and no access exceeds
  1-way.
- `count_instructions.py` / `ncu_compare.py` — analytic instruction counts,
  checked against Nsight Compute on hardware.

## Building and running

```bash
make run-cpu                       # host-only: references + 78-check suite + audits
make counts                        # analytic instruction counts (the `make ncu` reference)

make run-gpu SM=86                 # all three backends vs cuFFT fp32 + fp16  (SM=89 Ada, 90 Hopper, 120 Blackwell)
make run-dx  SM=86 CUFFTDX_DIR=... # adds cuFFTDx fp32 + fp16 (NVIDIA MathDx, or `pip install nvidia-mathdx`)
make sass    SM=86                 # per-backend SASS opcode histograms + register counts
make occ     SM=86                 # which resource caps occupancy
make ncu     SM=86                 # instruction counts vs cuFFT under Nsight Compute
```

The GPU targets need CUDA ≥ 12; the host suite needs only g++ and python3.
`make run-dx` uses C++17 for that target only.

## Limitations

- N must factor as lead·8^k or lead·16^k (lead ∈ {1,2,4,8}); 512–4096 in one
  block, larger via the four-step path.
- fp32 (and int16 block-floating-point storage) only.
- Single-GPU.

## License

Apache-2.0 — see [LICENSE](LICENSE).
