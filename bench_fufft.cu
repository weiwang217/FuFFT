// ---------------------------------------------------------------------------
// bench_fufft.cu — cyclic convolution: fuFFT vs cuFFT (and optionally cuFFTDx)
//
// For each size N this benchmark computes  IDFT(DFT(a) · b̂)  three ways:
//
//   cuFFT chain    ExecC2C -> pointwise multiply -> ExecC2C   (7 HBM passes)
//   fuFFT          one fused kernel                           (3 HBM passes)
//   cuFFTDx        one fused kernel, NVIDIA's device library  (3 HBM passes)
//                  (built only when CUFFTDX_DIR is set: `make run-dx`)
//
// Three fuFFT variants are timed:
//   in-place r8    radix-8 baseline, 8 values per lane
//   V=16           radix-16, 16 values per lane — the headline kernel
//   int16-BFP      V=16 with int16+scale storage in HBM (half the bytes)
//
// Every variant is checked against the cuFFT chain (rel L2) before timing.
// Run with `iters = 0` for profile mode: exactly one launch per kernel, so
// Nsight Compute sees each kernel once (`make ncu`).
//
//   ./bench_<backend> [iters=200]
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cufft.h>
#include <cufftXt.h>          // half-precision plans (CUDA_C_16F)
#include <cuda_fp16.h>

#include "fufft.cuh"

#define CK(x) do { cudaError_t e_ = (x); if(e_ != cudaSuccess) { \
   fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
   exit(1); } } while(0)

#define CF(x) do { cufftResult r_ = (x); if(r_ != CUFFT_SUCCESS) { \
   fprintf(stderr, "cufft %s:%d code %d\n", __FILE__, __LINE__, (int)r_); \
   exit(1); } } while(0)

struct Timer {
   cudaEvent_t a, b;
   Timer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
   ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
   void  start() { cudaEventRecord(a); }
   float stop()  { cudaEventRecord(b); cudaEventSynchronize(b);
                   float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static double rel_l2(const std::vector<float2>& x, const std::vector<float2>& y)
{
   double num = 0.0, den = 0.0;
   for(size_t i = 0; i < x.size(); i++) {
      double dx = (double)x[i].x - (double)y[i].x;
      double dy = (double)x[i].y - (double)y[i].y;
      num += dx * dx + dy * dy;
      den += (double)y[i].x * y[i].x + (double)y[i].y * y[i].y;
   }
   return sqrt(num / den);
}

// ---------------------------------------------------------------------------
// Filter permutation.  The forward DIF transform leaves the spectrum in
// bit-group-reversed order, so the filter spectrum is permuted ONCE on the
// host and the kernels never reorder anything.  The groups are the radix
// widths in pass order ([lead, W, W, ...], W = 3 for radix 8, 4 for radix 16),
// taken MSB-first and emitted in reverse.  A plain digit reversal agrees only
// when every group has the same width — i.e. only when N is a pure power of
// the radix.  (verify_kernel.cpp checks this against fp64 on the host.)
// ---------------------------------------------------------------------------
static int mixedrev_host(int idx, int LEAD, int NRAD, int W)
{
   int w[16], p = 0, m = 0;
   if(LEAD > 1) { int lw = 0; while((1 << lw) < LEAD) lw++; w[p++] = lw; m += lw; }
   for(int i = 0; i < NRAD; i++) { w[p++] = W; m += W; }
   int d[16], pos = m;
   for(int i = 0; i < p; i++) { pos -= w[i]; d[i] = (idx >> pos) & ((1 << w[i]) - 1); }
   int out = 0, op = m;
   for(int i = p - 1; i >= 0; i--) { op -= w[i]; out |= d[i] << op; }
   return out;
}

// float2 <-> cuFFT's CUDA_C_16F layout (one point = __half2{re, im})
__global__ void pack_h2(const float2* __restrict__ a, __half2* __restrict__ o, size_t total)
{
   const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(k < total) o[k] = __float22half2_rn(a[k]);
}
__global__ void unpack_h2(const __half2* __restrict__ a, float2* __restrict__ o, size_t total)
{
   const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(k < total) o[k] = __half22float2(a[k]);
}
// pointwise_mul on half2 data; fp32 internally -- the kernel is bandwidth
// bound, so the upconvert is free and slightly kinder to the incumbent
__global__ void pointwise_mul_h2(__half2* __restrict__ x, const __half2* __restrict__ w, size_t total)
{
   const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(k >= total) return;
   const float2 a = __half22float2(x[k]), b = __half22float2(w[k]);
   x[k] = __floats2half2_rn(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// int16 block-floating-point: one scale per transform, values quantised to
// int16.  Mirrors the device-side quantisation in fufft_pipeline_v16_bfp.
static void bfp_quant_host(const float2* x, int N, short2* q, float* scale)
{
   float mx = 1e-30f;
   for(int i = 0; i < N; i++)
      mx = std::max(mx, std::max(std::fabs(x[i].x), std::fabs(x[i].y)));
   const float inv = 32767.0f / mx;
   *scale = mx / 32767.0f;
   for(int i = 0; i < N; i++)
      q[i] = make_short2((short)lrintf(std::min(std::max(x[i].x * inv, -32767.f), 32767.f)),
                         (short)lrintf(std::min(std::max(x[i].y * inv, -32767.f), 32767.f)));
}

#include "cufftdx_conv.cuh"     // uses Timer / CK / rel_l2; inert without FUFFT_HAVE_CUFFTDX

// ---------------------------------------------------------------------------
// One size, single-block kernels.  N = lead·8^k = lead'·16^k' with the leads
// read off the bit width at compile time; CPB = transforms per block (keeps
// small N busy).  batch × N is sized to 64 MB per array in main().
// ---------------------------------------------------------------------------
template<int N, int CPB>
static void run_conv(int batch, int iters)
{
   // factorisations: N = L8·8^NR8 = L16·16^NR16
   constexpr int MBITS = (N ==  512 ?  9 : N == 1024 ? 10 : N == 2048 ? 11
                        : N == 4096 ? 12 : 0);
   static_assert(MBITS != 0, "add this N to the MBITS table");
   constexpr int NR8  = MBITS / 3, L8  = 1 << (MBITS % 3);
   constexpr int NR16 = MBITS / 4, L16 = 1 << (MBITS % 4);

   const bool   profile = (iters == 0);
   const size_t n     = (size_t)N * batch;
   const size_t bytes = n * sizeof(float2);
   const int    T     = N / 8;              // lanes per transform at radix 8
   const int    tpb   = CPB * T;
   const int    tpb16 = tpb / 2;            // V=16 halves the lane count
   const int    grid  = batch / CPB;
   const size_t smem  = (size_t)CPB * N * sizeof(float2);

   printf("\n=========== N = %-5d batch = %-6d (%zu MB/array, %d thr/blk, %d blocks) ===========\n",
          N, batch, bytes >> 20, tpb, grid);

   // opt-in shared memory (needed above 48 KB per block; harmless below)
   CK(cudaFuncSetAttribute((fufft_pipeline_ip<N, L8, NR8, CPB, MidMultiply>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
   CK(cudaFuncSetAttribute((fufft_pipeline_v16<N, L16, NR16, CPB>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
   CK(cudaFuncSetAttribute((fufft_pipeline_v16_bfp<N, L16, NR16, CPB>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

   // ---- inputs: signal a, filter b (1/N folded into b so nobody pays a
   //      normalisation kernel), plus b permuted for each radix chain -------
   std::mt19937 rng(N + batch);
   std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
   std::vector<float2> h_a(n), h_b(n), h_b8(n), h_b16(n);
   const float invN = 1.0f / (float)N;
   for(size_t i = 0; i < n; i++) {
      h_a[i] = make_float2(dist(rng), dist(rng));
      h_b[i] = make_float2(dist(rng) * invN, dist(rng) * invN);
   }
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++) {
         h_b8 [(size_t)t*N+i] = h_b[(size_t)t*N + mixedrev_host(i, L8,  NR8,  3)];
         h_b16[(size_t)t*N+i] = h_b[(size_t)t*N + mixedrev_host(i, L16, NR16, 4)];
      }

   // int16-BFP copies of the signal and of the V=16-permuted filter
   std::vector<short2> h_aq(n), h_bq(n);
   std::vector<float>  h_as(batch), h_bs(batch);
   for(int t = 0; t < batch; t++) {
      bfp_quant_host(&h_a  [(size_t)t * N], N, &h_aq[(size_t)t * N], &h_as[t]);
      bfp_quant_host(&h_b16[(size_t)t * N], N, &h_bq[(size_t)t * N], &h_bs[t]);
   }

   float2 *d_a, *d_b, *d_b8, *d_b16, *d_out, *d_ref, *d_tmp;
   for(float2** q : {&d_a, &d_b, &d_b8, &d_b16, &d_out, &d_ref, &d_tmp})
      CK(cudaMalloc(q, bytes));
   CK(cudaMemcpy(d_a,   h_a.data(),   bytes, cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_b,   h_b.data(),   bytes, cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_b8,  h_b8.data(),  bytes, cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_b16, h_b16.data(), bytes, cudaMemcpyHostToDevice));

   short2 *d_aq, *d_oq, *d_bq;
   float  *d_as, *d_os, *d_bs;
   CK(cudaMalloc(&d_aq, n * sizeof(short2)));
   CK(cudaMalloc(&d_oq, n * sizeof(short2)));
   CK(cudaMalloc(&d_bq, n * sizeof(short2)));
   CK(cudaMalloc(&d_as, batch * sizeof(float)));
   CK(cudaMalloc(&d_os, batch * sizeof(float)));
   CK(cudaMalloc(&d_bs, batch * sizeof(float)));
   CK(cudaMemcpy(d_aq, h_aq.data(), n * sizeof(short2), cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_bq, h_bq.data(), n * sizeof(short2), cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_as, h_as.data(), batch * sizeof(float), cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_bs, h_bs.data(), batch * sizeof(float), cudaMemcpyHostToDevice));

   // ---- the cuFFT chain: baseline and accuracy reference ------------------
   cufftHandle plan;
   CF(cufftPlan1d(&plan, N, CUFFT_C2C, batch));
   const int ptpb = 256, pblk = (int)((n + ptpb - 1) / ptpb);
   auto cufft_chain = [&] {
      CF(cufftExecC2C(plan, d_a, d_tmp, CUFFT_FORWARD));
      pointwise_mul<<<pblk, ptpb>>>(d_tmp, d_b, n);
      CF(cufftExecC2C(plan, d_tmp, d_ref, CUFFT_INVERSE));
   };
   cufft_chain();
   CK(cudaDeviceSynchronize());
   std::vector<float2> h_ref(n), h_out(n);
   CK(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));

   // ---- the contenders, as launch lambdas ---------------------------------
   auto run_ip  = [&] { fufft_pipeline_ip<N, L8, NR8, CPB><<<grid, tpb, smem>>>(
                           d_a, d_out, MidMultiply{d_b8}, nullptr); };
   auto run_v16 = [&] { fufft_pipeline_v16<N, L16, NR16, CPB><<<grid, tpb16, smem>>>(
                           d_a, d_out, MidMultiply{d_b16}, nullptr); };
   auto run_bfp = [&] { fufft_pipeline_v16_bfp<N, L16, NR16, CPB><<<grid, tpb16, smem>>>(
                           d_aq, d_as, d_bq, d_bs, d_oq, d_os, nullptr); };

   // ---- accuracy first: a fast wrong kernel is worth nothing --------------
   auto check = [&](const char* name) {
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      double e = rel_l2(h_out, h_ref);
      printf("      rel L2 %-14s %10.3e\n", name, e);
   };
   run_ip();  check("in-place r8");
   run_v16(); check("V=16");
   run_bfp();
   {  // dequantise the BFP output for its accuracy check
      CK(cudaDeviceSynchronize());
      std::vector<short2> oq(n); std::vector<float> os(batch);
      CK(cudaMemcpy(oq.data(), d_oq, n * sizeof(short2), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(os.data(), d_os, batch * sizeof(float), cudaMemcpyDeviceToHost));
      for(int t = 0; t < batch; t++)
         for(int i = 0; i < N; i++) {
            short2 v = oq[(size_t)t * N + i];
            h_out[(size_t)t * N + i] = make_float2(v.x * os[t], v.y * os[t]);
         }
      printf("      rel L2 %-14s %10.3e   (includes input quantisation)\n",
             "int16-BFP", rel_l2(h_out, h_ref));
   }

   if(profile) {   // one launch each already happened; that is what ncu wants
      cufftDestroy(plan);
      for(float2* q : {d_a, d_b, d_b8, d_b16, d_out, d_ref, d_tmp}) cudaFree(q);
      for(void* q : {(void*)d_aq, (void*)d_oq, (void*)d_bq,
                     (void*)d_as, (void*)d_os, (void*)d_bs}) cudaFree(q);
      return;
   }

   // ---- cuFFT's own 16-bit mode: half plan via cufftXt --------------------
   // Skipped gracefully where the library refuses half (exotic sizes/arch).
   __half2 *d_ah2, *d_bh2, *d_oh2, *d_th2;
   for(__half2** q : {&d_ah2, &d_bh2, &d_oh2, &d_th2}) CK(cudaMalloc(q, n * sizeof(__half2)));
   pack_h2<<<pblk, ptpb>>>(d_a, d_ah2, n);
   pack_h2<<<pblk, ptpb>>>(d_b, d_bh2, n);
   cufftHandle plan16;
   CF(cufftCreate(&plan16));
   long long nll = N; size_t ws16 = 0;
   const bool have16 = cufftXtMakePlanMany(plan16, 1, &nll, nullptr, 1, N, CUDA_C_16F,
                                           nullptr, 1, N, CUDA_C_16F, batch, &ws16,
                                           CUDA_C_16F) == CUFFT_SUCCESS;
   auto cufft16_chain = [&] {
      CF(cufftXtExec(plan16, d_ah2, d_th2, CUFFT_FORWARD));
      pointwise_mul_h2<<<pblk, ptpb>>>(d_th2, d_bh2, n);
      CF(cufftXtExec(plan16, d_th2, d_oh2, CUFFT_INVERSE));
   };
   double err_cufft16 = 0.0;
   if(have16) {
      cufft16_chain();
      unpack_h2<<<pblk, ptpb>>>(d_oh2, d_out, n);
      CK(cudaGetLastError());
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
      err_cufft16 = rel_l2(h_out, h_ref);
   }

   // ---- timings: warm 3, time `iters` -------------------------------------
   Timer tm;
   auto bench = [&](auto fn) {
      for(int i = 0; i < 3; i++) fn();
      CK(cudaDeviceSynchronize());
      tm.start();
      for(int i = 0; i < iters; i++) fn();
      return tm.stop() / iters;
   };
   const float ms_cufft = bench(cufft_chain);
   const float ms_ip    = bench(run_ip);
   const float ms_v16   = bench(run_v16);
   const float ms_bfp   = bench(run_bfp);
   const float ms_cufft16 = have16 ? bench(cufft16_chain) : 0.0f;

#ifdef FUFFT_HAVE_CUFFTDX
   // cuFFTDx multiplies against a NATURAL-order spectrum (d_b, not d_b16!) —
   // handing it a permuted filter produces rel L2 ≈ 1.4 with normal timings.
   double err_dx = 0.0;
   const float ms_dx = dx_bench<N, 8, 1, FUFFT_DX_ARCH>(d_a, d_b, d_out, batch,
                                                        iters, tm, &err_dx, h_ref);
   // fp16 pairs two batches per element, so it gets the plain natural-order
   // spectrum too; d_out is free again as the unpack scratch.
   // fp16 rules: FFTsPerBlock must be even (2 logical FFTs = 1 pair) and the
   // half pipeline caps at 256 threads, so N = 4096 needs 16 elements/thread.
   double err_dx16 = 0.0;
   const float ms_dx16 = dx16_bench<N, (N >= 4096 ? 16 : 8), 2, FUFFT_DX_ARCH>(
                            d_a, d_b, d_out, batch, iters, tm, &err_dx16, h_ref);
#endif

   // ---- report ------------------------------------------------------------
   // GB/s uses the fused traffic (3 passes = read a + read b̂ + write out);
   // BFP moves the same points at 4 bytes each instead of 8.
   const double gb  = 3.0 * bytes / 1e9;
   const double gbq = 3.0 * n * sizeof(short2) / 1e9;
   printf("      %-22s %10s %10s %8s\n", "", "ms", "GB/s", "passes");
   printf("      %-22s %10.4f %10.1f %8d\n", "cuFFT 3-kernel chain",
          ms_cufft, 7.0 / 3.0 * gb / (ms_cufft * 1e-3), 7);
   if(have16)
      printf("      %-22s %10.4f %10.1f %8d   (half, 4 B/pt, rel L2 %.1e)\n",
             "cuFFT fp16 chain", ms_cufft16,
             7.0 / 3.0 * gbq / (ms_cufft16 * 1e-3), 7, err_cufft16);
   printf("      %-22s %10.4f %10.1f %8d\n", "fuFFT in-place r8",
          ms_ip,  gb / (ms_ip  * 1e-3), 3);
   printf("      %-22s %10.4f %10.1f %8d\n", "fuFFT V=16",
          ms_v16, gb / (ms_v16 * 1e-3), 3);
   printf("      %-22s %10.4f %10.1f %8d   (int16+scale, 4 B/pt)\n", "fuFFT int16-BFP",
          ms_bfp, gbq / (ms_bfp * 1e-3), 3);
#ifdef FUFFT_HAVE_CUFFTDX
   printf("      %-22s %10.4f %10.1f %8d   (rel L2 %.1e)\n", "cuFFTDx fused",
          ms_dx, gb / (ms_dx * 1e-3), 3, err_dx);
   printf("      %-22s %10.4f %10.1f %8d   (fp16 pairs, 4 B/pt, rel L2 %.1e)\n",
          "cuFFTDx fp16 fused", ms_dx16, gbq / (ms_dx16 * 1e-3), 3, err_dx16);
#endif
   printf("      speedup vs cuFFT       %9.2fx   (best of ours)\n",
          ms_cufft / std::min(ms_v16, ms_ip));
   if(have16)
      printf("      16-bit vs cuFFT fp16   %9.2fx   (int16-BFP vs cuFFT half chain)\n",
             ms_cufft16 / ms_bfp);
#ifdef FUFFT_HAVE_CUFFTDX
   printf("      speedup vs cuFFTDx     %9.2fx  fp32   %.2fx  int16-BFP\n",
          ms_dx / ms_v16, ms_dx / ms_bfp);
   printf("      16-bit vs 16-bit       %9.2fx   (int16-BFP vs cuFFTDx fp16)\n",
          ms_dx16 / ms_bfp);
#endif

   cufftDestroy(plan);
   cufftDestroy(plan16);
   for(float2* q : {d_a, d_b, d_b8, d_b16, d_out, d_ref, d_tmp}) cudaFree(q);
   for(void* q : {(void*)d_aq, (void*)d_oq, (void*)d_bq,
                  (void*)d_as, (void*)d_os, (void*)d_bs}) cudaFree(q);
   for(void* q : {(void*)d_ah2, (void*)d_bh2, (void*)d_oh2, (void*)d_th2}) cudaFree(q);
}

// ---------------------------------------------------------------------------
// Large N: four-step decomposition.  LOUT outer levels of radix ROUT run as
// flat passes over HBM; what remains factors into independent transforms of
// size MID, handled by the unchanged fused V=16 kernel with chunks posing as
// batch items.  Total passes = 4·LOUT + 3.
//
// MidLaunch picks the narrow or wide V=16 kernel at compile time: a middle
// above 4096 needs more than 256 lanes, which the standard launch bound
// forbids (and the wide variant must ask for ONE resident block — three of
// 64 KB shared each is impossible, and asking anyway makes ptxas spill).
// ---------------------------------------------------------------------------
template<int MID, int MLEAD, int MNRV, bool WIDE = (MID / 16 > 256)>
struct MidLaunch {
   static void set(size_t smem) {
      CK(cudaFuncSetAttribute((fufft_pipeline_v16<MID, MLEAD, MNRV, 1>),
                              cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
   }
   static void go(int grid, size_t smem, const float2* a, float2* out, const float2* b) {
      fufft_pipeline_v16<MID, MLEAD, MNRV, 1><<<grid, MID / 16, smem>>>(
         a, out, MidMultiply{b}, nullptr);
   }
};
template<int MID, int MLEAD, int MNRV>
struct MidLaunch<MID, MLEAD, MNRV, true> {
   static void set(size_t smem) {
      CK(cudaFuncSetAttribute((fufft_pipeline_v16_wide<MID, MLEAD, MNRV, 1>),
                              cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
   }
   static void go(int grid, size_t smem, const float2* a, float2* out, const float2* b) {
      fufft_pipeline_v16_wide<MID, MLEAD, MNRV, 1><<<grid, MID / 16, smem>>>(
         a, out, MidMultiply{b}, nullptr);
   }
};

template<int N, int LOUT, int ROUT, int MID, int MLEAD, int MNRV>
static void run_large(int batch, int iters)
{
   const size_t n     = (size_t)N * batch;
   const size_t bytes = n * sizeof(float2);
   printf("\n=========== LARGE N = %-8d LOUT=%d batch = %-5d (%zu MB/array) ===========\n",
          N, LOUT, batch, bytes >> 20);

   MidLaunch<MID, MLEAD, MNRV>::set(MID * 8);

   std::mt19937 rng(N + batch);
   std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
   std::vector<float2> h_a(n), h_b(n), h_bdr(n);
   const float invN = 1.0f / (float)N;
   for(size_t i = 0; i < n; i++) {
      h_a[i] = make_float2(dist(rng), dist(rng));
      h_b[i] = make_float2(dist(rng) * invN, dist(rng) * invN);
   }

   // Filter permutation for the CHAINED transform: bit-group reversal over
   // [ROUT]×LOUT ++ the middle's radices — same rule as mixedrev_host, with
   // the outer levels prepended.  verify_kernel.cpp checks the whole
   // decomposition against a monolithic transform on the host.
   int wid[24], np = 0, mb = 0;
   {
      auto push = [&](int R){ int lw = 0; while((1 << lw) < R) lw++; wid[np++] = lw; mb += lw; };
      for(int l = 0; l < LOUT; l++) push(ROUT);
      if(MLEAD > 1) push(MLEAD);
      for(int l = 0; l < MNRV; l++) push(16);
   }
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++) {
         int dg[24], pos = mb;
         for(int k = 0; k < np; k++) { pos -= wid[k]; dg[k] = (i >> pos) & ((1 << wid[k]) - 1); }
         int r = 0, op = mb;
         for(int k = np - 1; k >= 0; k--) { op -= wid[k]; r |= dg[k] << op; }
         h_bdr[(size_t)t * N + i] = h_b[(size_t)t * N + r];
      }

   float2 *d_a, *d_w, *d_bn, *d_bdr, *d_ref, *d_tmp;
   for(float2** q : {&d_a, &d_w, &d_bn, &d_bdr, &d_ref, &d_tmp}) CK(cudaMalloc(q, bytes));
   CK(cudaMemcpy(d_a,   h_a.data(),   bytes, cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_bn,  h_b.data(),   bytes, cudaMemcpyHostToDevice));
   CK(cudaMemcpy(d_bdr, h_bdr.data(), bytes, cudaMemcpyHostToDevice));

   cufftHandle plan;
   CF(cufftPlan1d(&plan, N, CUFFT_C2C, batch));
   const int    ptpb = 256, pblk = (int)((n + ptpb - 1) / ptpb);
   const size_t ng_r = n / ROUT;
   const int    otpb = 256, oblk = (int)((n / 8 + otpb - 1) / otpb);
   const int    mid_grid = (int)(n / MID);

   auto ours = [&](float2* buf) {
      int M = N / ROUT;
      for(int l = 0; l < LOUT; l++, M /= ROUT)
         fufft_outer_pass_r<ROUT, -1, false><<<oblk, otpb>>>(buf, M, ng_r);
      MidLaunch<MID, MLEAD, MNRV>::go(mid_grid, MID * 8, buf, buf, d_bdr);
      M = MID;
      for(int l = 0; l < LOUT; l++, M *= ROUT)
         fufft_outer_pass_r<ROUT, +1, true><<<oblk, otpb>>>(buf, M, ng_r);
   };

   // correctness vs the cuFFT chain
   CF(cufftExecC2C(plan, d_a, d_tmp, CUFFT_FORWARD));
   pointwise_mul<<<pblk, ptpb>>>(d_tmp, d_bn, n);
   CF(cufftExecC2C(plan, d_tmp, d_ref, CUFFT_INVERSE));
   CK(cudaMemcpy(d_w, d_a, bytes, cudaMemcpyDeviceToDevice));
   ours(d_w);
   CK(cudaGetLastError());
   CK(cudaDeviceSynchronize());
   std::vector<float2> h_ours(n), h_ref(n);
   CK(cudaMemcpy(h_ours.data(), d_w, bytes, cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(h_ref.data(),  d_ref, bytes, cudaMemcpyDeviceToHost));
   const double err = rel_l2(h_ours, h_ref);

   Timer tm;
   auto bench = [&](auto fn) {
      for(int i = 0; i < 3; i++) fn();
      CK(cudaDeviceSynchronize());
      tm.start();
      for(int i = 0; i < iters; i++) fn();
      return tm.stop() / std::max(iters, 1);
   };
   const float ms_cufft = bench([&] {
      CF(cufftExecC2C(plan, d_a, d_tmp, CUFFT_FORWARD));
      pointwise_mul<<<pblk, ptpb>>>(d_tmp, d_bn, n);
      CF(cufftExecC2C(plan, d_tmp, d_ref, CUFFT_INVERSE));
   });
   const float ms_ours = bench([&] { ours(d_w); });   // in-place reuse: timing only

   const double gb = (4.0 * LOUT + 3.0) * bytes / 1e9;
   printf("  config : outer radix %d x%d, middle %d (V=16, lead %d, %d x r16)\n",
          ROUT, LOUT, MID, MLEAD, MNRV);
   printf("  cuFFT chain              %9.4f ms\n", ms_cufft);
   printf("  fuFFT four-step          %9.4f ms   %6.1f GB/s eff (%d passes)\n",
          ms_ours, gb / (ms_ours * 1e-3), 4 * LOUT + 3);
   printf("  speedup vs cuFFT         %9.2fx\n", ms_cufft / ms_ours);
   printf("  rel L2 vs cuFFT chain    %9.3e\n", err);

   cufftDestroy(plan);
   for(float2* q : {d_a, d_w, d_bn, d_bdr, d_ref, d_tmp}) cudaFree(q);
}

int main(int argc, char** argv)
{
   int iters = (argc > 1) ? atoi(argv[1]) : 200;   // 0 = profile mode (for ncu)

   cudaDeviceProp p;
   CK(cudaGetDeviceProperties(&p, 0));
   printf("device : %s   sm_%d%d   %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
   // CUDA 13 removed memoryClockRate/memoryBusWidth from cudaDeviceProp;
   // cudaDeviceGetAttribute still carries them on every version.
   int mclk_khz = 0, mbus_bits = 0;
   cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, 0);
   cudaDeviceGetAttribute(&mbus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
   printf("peak BW: %.0f GB/s     smem/block %zu KB (opt-in %zu KB)\n",
          2.0 * mclk_khz * (mbus_bits / 8) / 1.0e6,
          p.sharedMemPerBlock >> 10, p.sharedMemPerBlockOptin >> 10);
   printf("backend: %s\n", fufft_backend_name());
   printf("iters  : %d%s\n", iters, iters ? "" : "  (profile mode: one launch per kernel)");

   // batch chosen so every size moves 64 MB per array; CPB keeps blocks full
   run_conv< 512, 8>(16384, iters);
   run_conv<1024, 4>( 8192, iters);
   run_conv<2048, 2>( 4096, iters);
   run_conv<4096, 1>( 2048, iters);

   printf("\n\n################ large N: four-step, fused middle ################\n");
   //         N      LOUT ROUT  MID  lead nrv        outer factor
   run_large<32768,   1,  16,  2048,  8,  2>(256, iters);   // 16
   run_large<262144,  2,  16,  1024,  4,  2>(32,  iters);   // 256 = 16^2

   printf("\nSupported N: lead*8^k / lead*16^k (lead 1,2,4,8), one block up to 4096;\n"
          "larger N via the four-step path.\n");
   return 0;
}
