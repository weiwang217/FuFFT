#pragma once
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Backend selection.  All three compute exactly the same butterfly graph over
// the same shared-memory layout; only the way the arithmetic is expressed
// differs.  That isolates the question "does hand-written assembly help?".
//
//   0  CPP_NAIVE : every twiddle goes through the generic complex multiply
//   1  CPP_OPT   : x(+-i) is a swap+negate (zero flops); x w^1 and x w^3 use
//                  the (c, +-c) structure -> 2 mul + 2 add instead of 4 mul +
//                  2 add
//   2  PTX       : the same specialisations, hand written as inline PTX
// ---------------------------------------------------------------------------
#define FUFFT_CPP_NAIVE 0
#define FUFFT_CPP_OPT   1
#define FUFFT_PTX       2

#ifndef FUFFT_BACKEND
#define FUFFT_BACKEND FUFFT_CPP_OPT
#endif

#ifndef FUFFT_FAST_TRIG
#define FUFFT_FAST_TRIG 1
#endif

// Each pass needs W^(j*k) for k=1..7.  Computing all seven with __sincosf costs
// ~1.5x the entire butterfly arithmetic, because MUFU runs at roughly 1/8 the
// FMA rate.  With the recurrence we pay one __sincosf for W^j and reach the
// rest by repeated multiplication: ~2.5x cheaper on the twiddle path, at the
// cost of error growing to about 6*eps on W^7 instead of 1*eps.
//
// This also matters for the backend comparison: the three backends differ only
// in butterfly arithmetic, so leaving the transcendentals dominant would dilute
// whatever difference exists between them.
#ifndef FUFFT_TWIDDLE_RECURRENCE
#define FUFFT_TWIDDLE_RECURRENCE 1
#endif

#define FUFFT_R2 0.70710678118654752440f

// Barrier scoped to what actually needs synchronising.  A block holds CPB
// independent transforms; the block-wide __syncthreads chains them together
// for no reason.  When one transform fits in a single warp (T <= 32, and the
// slot boundaries are warp-aligned because T divides 32 or is a multiple of
// it), __syncwarp is enough and the slots decouple completely -- each warp
// flows through its passes without waiting for its neighbours.  T is a
// template constant, so the branch is resolved at compile time.
template<int T_>
__device__ __forceinline__ void fufft_bar()
{
   if(T_ <= 32) __syncwarp();
   else         __syncthreads();
}

// Occupancy is capped by whichever of shared memory, threads and registers
// binds first, and for these kernels it is easy for registers to bind before
// shared memory does -- in which case halving the shared buffer buys nothing.
// __launch_bounds__ tells ptxas the target so it caps registers deliberately
// instead of us finding out from a profiler.  FUFFT_MIN_BLOCKS is the number of
// resident blocks per SM to aim for; ptxas will spill rather than exceed it, so
// check the `Used N registers` line and any spill counts in the build log
// before trusting it.
//
// Kernels holding this much shared memory need explicit launch bounds, or
// ptxas plans registers for an occupancy the shared budget can never deliver.
#ifndef FUFFT_MIN_BLOCKS
#define FUFFT_MIN_BLOCKS 3
#endif
#ifndef FUFFT_MAX_THREADS
#define FUFFT_MAX_THREADS 512
#endif
#if FUFFT_MIN_BLOCKS > 0
#define FUFFT_BOUNDS   __launch_bounds__(FUFFT_MAX_THREADS, FUFFT_MIN_BLOCKS)
// V=16 halves the lane count, so its blocks run FUFFT_MAX_THREADS/2 threads.
// Reusing FUFFT_BOUNDS would tell ptxas to budget registers for 512-thread
// blocks -- 65536/(512*3) = 42 regs -- and V=16 carries 32 data registers
// before any addressing, so that cap would force spills and sabotage the A/B.
// At 256 threads the same three-block target allows 85.
#define FUFFT_BOUNDS16 __launch_bounds__(FUFFT_MAX_THREADS / 2, FUFFT_MIN_BLOCKS)
#else
#define FUFFT_BOUNDS
#define FUFFT_BOUNDS16
#endif

// ---------------------------------------------------------------------------
// In-kernel phase timer, the mechanism from asm_helper.cu's TIMER_MARK.
//
// Nsight Compute reports per-kernel metrics; this reports the split *inside*
// one kernel -- how many cycles go to the global load, each butterfly pass, the
// pointwise multiply and the store.  That is what turns "memory-bound by 4-7x"
// from a model into a measurement.
//
// Design, following the original:
//   * only thread 0 reads %clock64, and only after a barrier, so every thread
//     has actually reached the mark
//   * the phase id goes in the top 8 bits and the cycle delta in the low 24,
//     so one u32 carries both.  24 bits is 16.7M cycles, ~7 ms at 2.4 GHz
//   * marks are cumulative from START; a phase duration is the difference
//     between consecutive marks
//   * %smid is recorded so you can see which SM a block landed on
//   * compiled out entirely when off
//
// Caveat, and it is the same one the original has: the marks insert barriers
// that the untimed kernel does not have, so timing perturbs what it measures.
// Read the phase *proportions*, not the absolute total, and confirm the total
// against an untimed run.
// ---------------------------------------------------------------------------
#ifndef FUFFT_TIMING
#define FUFFT_TIMING 0
#endif

#define FUFFT_MAX_MARKS    30
#define FUFFT_TIMER_BLOCKS 64          // how many blocks report
#define FUFFT_TIMER_STRIDE (FUFFT_MAX_MARKS + 2)

#define K_LOAD  1u
#define K_FWD   2u
#define K_MULT  3u
#define K_INV   4u
#define K_STORE 5u

#if FUFFT_TIMING

// File scope, as timer.cu does it: the marks live in device functions called
// from the kernel, and those cannot see a __shared__ declared inside it.  A
// file-scope __shared__ still gives each block its own copy.
__shared__ unsigned int       s_marks[FUFFT_MAX_MARKS];
__shared__ unsigned int       s_nmark;
__shared__ unsigned int       s_smid;
__shared__ unsigned long long s_t0;

#define FUFFT_TIMER_DECL()      do {} while(0)

#define FUFFT_TIMER_START()                                                    \
   do {                                                                       \
      __syncthreads();                                                        \
      if(threadIdx.x == 0) {                                                  \
         s_nmark = 0;                                                         \
         asm volatile("mov.u32 %0, %%smid;" : "=r"(s_smid));                  \
         s_t0 = clock64();                                                    \
      }                                                                       \
      __syncthreads();                                                        \
   } while(0)

#define FUFFT_TIMER_MARK(key)                                                  \
   do {                                                                       \
      __syncthreads();                                                        \
      if(threadIdx.x == 0) {                                                  \
         unsigned long long d = clock64() - s_t0;                             \
         unsigned int c = (d > 0x00FFFFFFull) ? 0x00FFFFFFu                   \
                                              : (unsigned int)d;              \
         if(s_nmark < FUFFT_MAX_MARKS)                                         \
            s_marks[s_nmark++] = c | ((key) << 24);                           \
      }                                                                       \
   } while(0)

#define FUFFT_TIMER_DUMP(out)                                                  \
   do {                                                                       \
      __syncthreads();                                                        \
      if(threadIdx.x == 0 && (out) != 0 && blockIdx.x < FUFFT_TIMER_BLOCKS) {  \
         unsigned int* p_ = (out) + blockIdx.x * FUFFT_TIMER_STRIDE;           \
         p_[0] = s_smid;                                                      \
         p_[1] = s_nmark;                                                     \
         for(unsigned int i_ = 0; i_ < s_nmark; i_++) p_[2 + i_] = s_marks[i_]; \
      }                                                                       \
   } while(0)

#else
#define FUFFT_TIMER_DECL()      do {} while(0)
#define FUFFT_TIMER_START()     do {} while(0)
#define FUFFT_TIMER_MARK(key)   do {} while(0)
// consumes the pointer so the kernels keep one signature either way and the
// parameter does not read as unused when timing is compiled out
#define FUFFT_TIMER_DUMP(out)   do { (void)(out); } while(0)
#endif

// ---------------------------------------------------------------------------
// Shared-memory swizzle.  Without it the stage-0 scatter writes at a stride of
// 8 float2 (64 bytes), so a 16-lane LDS/STS phase lands on only 2 distinct
// banks -> 8-way conflict.  Stage 1 is 2-way.  This permutation is bijective
// (bits 4..6 are untouched, which lets bit 3 and then bits 0..2 be recovered)
// and takes every gather and scatter in every stage to 1-way.  Costs no extra
// shared memory, unlike padding.
//
// Correctness only requires that it be applied to *every* shared access; the
// CPU transcription in verify_kernel.cpp applies the same permutation, so a
// missed site shows up as an fp64 mismatch there.
#ifndef FUFFT_SWIZZLE
#define FUFFT_SWIZZLE 1
#endif
#if FUFFT_SWIZZLE
// Bijective, and 1-way on every access pattern in this file: the radix-8
// in-place gather/scatter, the Stockham pair, the linear load/store loops, and
// the radix-16 lane-pair passes.
//
// The first term folds bits 3..6 into bits 0..3 and handles a stride-8 scatter.
// The other two fold bits 7 and 11 into bit 3, which is what separates a
// radix-16 lane pair: its two lanes are 8M apart, and 8M runs over
// {8, 128, 2048} across the radix-16 stride sequence.  Bits 4+ are untouched,
// so bit 3 and then bits 0..2 are recoverable -- a lower-triangular XOR, hence
// a permutation and not a data-corrupting alias.
#define SWZ(i) ((i) ^ (((i) >> 3) & 15) ^ (((i) >> 4) & 8) ^ (((i) >> 8) & 8))
#else
#define SWZ(i) (i)
#endif

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ float2 cadd(float2 a, float2 b)
{
#if FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   asm("{\n\t"
       "add.f32 %0, %2, %4;\n\t"
       "add.f32 %1, %3, %5;\n\t"
       "}" : "=&f"(r.x), "=&f"(r.y)
           : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
   return r;
#else
   return make_float2(a.x + b.x, a.y + b.y);
#endif
}

__device__ __forceinline__ float2 csub(float2 a, float2 b)
{
#if FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   asm("{\n\t"
       "sub.f32 %0, %2, %4;\n\t"
       "sub.f32 %1, %3, %5;\n\t"
       "}" : "=&f"(r.x), "=&f"(r.y)
           : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
   return r;
#else
   return make_float2(a.x - b.x, a.y - b.y);
#endif
}

// full complex multiply, used for the per-stage twiddles
__device__ __forceinline__ float2 cmul(float2 a, float2 b)
{
#if FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   asm("{\n\t"
       ".reg .f32 t0, t1;\n\t"
       "mul.f32    t0, %3, %5;\n\t"      // ay*by
       "mul.f32    t1, %3, %4;\n\t"      // ay*bx
       "neg.f32    t0, t0;\n\t"
       "fma.rn.f32 %0, %2, %4, t0;\n\t"  // ax*bx - ay*by
       "fma.rn.f32 %1, %2, %5, t1;\n\t"  // ax*by + ay*bx
       "}" : "=&f"(r.x), "=&f"(r.y)
           : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
   return r;
#else
   return make_float2(fmaf(a.x, b.x, -a.y * b.y),
                      fmaf(a.x, b.y,  a.y * b.x));
#endif
}

// multiply by w^2 = (0, SIGN) -- i.e. by +-i.
// (x + yi)(0 + si) = -s*y + s*x*i .  No arithmetic, only sign flips.
template<int SIGN>
__device__ __forceinline__ float2 cmul_i(float2 a)
{
#if FUFFT_BACKEND == FUFFT_CPP_NAIVE
   return cmul(a, make_float2(0.0f, (float)SIGN));
#elif FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   if(SIGN < 0) {
      asm("{\n\t"
          "mov.f32 %0, %3;\n\t"
          "neg.f32 %1, %2;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y));
   } else {
      asm("{\n\t"
          "neg.f32 %0, %3;\n\t"
          "mov.f32 %1, %2;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y));
   }
   return r;
#else
   return (SIGN < 0) ? make_float2( a.y, -a.x)
                     : make_float2(-a.y,  a.x);
#endif
}

// multiply by w^1 = r2*(1, SIGN)
//   SIGN=-1 : ( r2*(x+y),  r2*(y-x) )
//   SIGN=+1 : ( r2*(x-y),  r2*(x+y) )
template<int SIGN>
__device__ __forceinline__ float2 cmul_w1(float2 a)
{
#if FUFFT_BACKEND == FUFFT_CPP_NAIVE
   return cmul(a, make_float2(FUFFT_R2, (float)SIGN * FUFFT_R2));
#elif FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   const float k = FUFFT_R2;
   if(SIGN < 0) {
      asm("{\n\t"
          ".reg .f32 s0, s1;\n\t"
          "add.f32 s0, %2, %3;\n\t"
          "sub.f32 s1, %3, %2;\n\t"
          "mul.f32 %0, s0, %4;\n\t"
          "mul.f32 %1, s1, %4;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y), "f"(k));
   } else {
      asm("{\n\t"
          ".reg .f32 s0, s1;\n\t"
          "sub.f32 s0, %2, %3;\n\t"
          "add.f32 s1, %2, %3;\n\t"
          "mul.f32 %0, s0, %4;\n\t"
          "mul.f32 %1, s1, %4;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y), "f"(k));
   }
   return r;
#else
   return (SIGN < 0) ? make_float2(FUFFT_R2 * (a.x + a.y), FUFFT_R2 * (a.y - a.x))
                     : make_float2(FUFFT_R2 * (a.x - a.y), FUFFT_R2 * (a.x + a.y));
#endif
}

// multiply by w^3 = r2*(-1, SIGN)
//   SIGN=-1 : ( r2*(y-x),  r2*(-x-y) )
//   SIGN=+1 : ( r2*(-x-y), r2*(x-y)  )
template<int SIGN>
__device__ __forceinline__ float2 cmul_w3(float2 a)
{
#if FUFFT_BACKEND == FUFFT_CPP_NAIVE
   return cmul(a, make_float2(-FUFFT_R2, (float)SIGN * FUFFT_R2));
#elif FUFFT_BACKEND == FUFFT_PTX
   float2 r;
   const float k = FUFFT_R2;
   if(SIGN < 0) {
      asm("{\n\t"
          ".reg .f32 s0, s1;\n\t"
          "sub.f32 s0, %3, %2;\n\t"
          "add.f32 s1, %2, %3;\n\t"
          "neg.f32 s1, s1;\n\t"
          "mul.f32 %0, s0, %4;\n\t"
          "mul.f32 %1, s1, %4;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y), "f"(k));
   } else {
      asm("{\n\t"
          ".reg .f32 s0, s1;\n\t"
          "add.f32 s0, %2, %3;\n\t"
          "neg.f32 s0, s0;\n\t"
          "sub.f32 s1, %2, %3;\n\t"
          "mul.f32 %0, s0, %4;\n\t"
          "mul.f32 %1, s1, %4;\n\t"
          "}" : "=&f"(r.x), "=&f"(r.y) : "f"(a.x), "f"(a.y), "f"(k));
   }
   return r;
#else
   return (SIGN < 0) ? make_float2(FUFFT_R2 * (a.y - a.x), FUFFT_R2 * (-a.x - a.y))
                     : make_float2(FUFFT_R2 * (-a.x - a.y), FUFFT_R2 * (a.x - a.y));
#endif
}

__device__ __forceinline__ float2 twiddle(float ang)
{
   float s, c;
#if FUFFT_FAST_TRIG
   __sincosf(ang, &s, &c);
#else
   sincosf(ang, &s, &c);
#endif
   return make_float2(c, s);
}

// ---------------------------------------------------------------------------
// Radix-8 DFT, three radix-2 stages, output in natural order.
// Mirrors FFT8() in asm_fft.cu: MULTIPLY_1_8 / _1_4 / _3_8 are w^1, w^2, w^3.
// ---------------------------------------------------------------------------
template<int SIGN>
__device__ __forceinline__ void dft8(const float2 u[8], float2 v[8])
{
   float2 a0 = cadd(u[0], u[4]);
   float2 a1 = cadd(u[1], u[5]);
   float2 a2 = cadd(u[2], u[6]);
   float2 a3 = cadd(u[3], u[7]);
   float2 a4 = csub(u[0], u[4]);
   float2 a5 = cmul_w1<SIGN>(csub(u[1], u[5]));
   float2 a6 = cmul_i <SIGN>(csub(u[2], u[6]));
   float2 a7 = cmul_w3<SIGN>(csub(u[3], u[7]));

   float2 b0 = cadd(a0, a2);
   float2 b1 = cadd(a1, a3);
   float2 b2 = csub(a0, a2);
   float2 b3 = cmul_i<SIGN>(csub(a1, a3));
   float2 b4 = cadd(a4, a6);
   float2 b5 = cadd(a5, a7);
   float2 b6 = csub(a4, a6);
   float2 b7 = cmul_i<SIGN>(csub(a5, a7));

   v[0] = cadd(b0, b1);
   v[4] = csub(b0, b1);
   v[2] = cadd(b2, b3);
   v[6] = csub(b2, b3);
   v[1] = cadd(b4, b5);
   v[5] = csub(b4, b5);
   v[3] = cadd(b6, b7);
   v[7] = csub(b6, b7);
}

template<int SIGN>
__device__ __forceinline__ void dft2(const float2 u[2], float2 v[2])
{
   v[0] = cadd(u[0], u[1]);
   v[1] = csub(u[0], u[1]);
}

template<int SIGN>
__device__ __forceinline__ void dft4(const float2 u[4], float2 v[4])
{
   float2 a0 = cadd(u[0], u[2]);
   float2 a1 = cadd(u[1], u[3]);
   float2 a2 = csub(u[0], u[2]);
   float2 a3 = cmul_i<SIGN>(csub(u[1], u[3]));   // x(+-i) is free, as in dft8
   v[0] = cadd(a0, a1);
   v[2] = csub(a0, a1);
   v[1] = cadd(a2, a3);
   v[3] = csub(a2, a3);
}

template<int R, int SIGN> struct Butterfly;
template<int SIGN> struct Butterfly<2, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft2<SIGN>(u, v); } };
template<int SIGN> struct Butterfly<4, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft4<SIGN>(u, v); } };
template<int SIGN> struct Butterfly<8, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft8<SIGN>(u, v); } };

// twiddles for a radix-R pass; same recurrence, R-1 of them instead of 7
template<int R, int SIGN>
__device__ __forceinline__ void apply_twiddles_r(float2* x, float step, int j)
{
#if FUFFT_TWIDDLE_RECURRENCE
   const float2 w1 = twiddle(step * (float)j);
   float2 w = w1;
   x[1] = cmul(x[1], w);
#pragma unroll
   for(int k = 2; k < R; k++) { w = cmul(w, w1); x[k] = cmul(x[k], w); }
#else
#pragma unroll
   for(int k = 1; k < R; k++) x[k] = cmul(x[k], twiddle(step * (float)(j * k)));
#endif
}

// ---------------------------------------------------------------------------
// A radix-R pass with the lane count fixed at T = N/8.  Each lane still holds
// eight values, so it runs G = 8/R independent butterflies and owns the groups
// { lane, lane+T, ..., lane+(G-1)T }.  Shared memory, register pressure and the
// block shape are identical to the radix-8 case; only the butterfly and the
// index arithmetic change.  That is what lets N be any power of two rather than
// only a power of eight.
// ---------------------------------------------------------------------------
template<int R, int SIGN>
__device__ __forceinline__ void fufft_dif_pass_r(float2* A, int M, int lane, int T)
{
   const int G = 8 / R;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
#pragma unroll
   for(int s = 0; s < G; s++) {
      const int gidx = lane + s * T;
      const int j    = gidx % M;
      const int b    = gidx / M;
      const int base = b * R * M + j;

      float2 u[R], v[R];
#pragma unroll
      for(int k = 0; k < R; k++) u[k] = A[SWZ(base + k * M)];
      Butterfly<R, SIGN>::go(u, v);
      if(M > 1) apply_twiddles_r<R, SIGN>(v, step, j);
#pragma unroll
      for(int k = 0; k < R; k++) A[SWZ(base + k * M)] = v[k];
   }
}

template<int R, int SIGN>
__device__ __forceinline__ void fufft_dit_pass_r(float2* A, int M, int lane, int T)
{
   const int G = 8 / R;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
#pragma unroll
   for(int s = 0; s < G; s++) {
      const int gidx = lane + s * T;
      const int j    = gidx % M;
      const int b    = gidx / M;
      const int base = b * R * M + j;

      float2 u[R], v[R];
#pragma unroll
      for(int k = 0; k < R; k++) u[k] = A[SWZ(base + k * M)];
      if(M > 1) apply_twiddles_r<R, SIGN>(u, step, j);
      Butterfly<R, SIGN>::go(u, v);
#pragma unroll
      for(int k = 0; k < R; k++) A[SWZ(base + k * M)] = v[k];
   }
}

// ---------------------------------------------------------------------------
// radix 16 across a lane pair.
//
// radix 16 = one radix-2 stage feeding two radix-8s.  That stage only ever
// pairs u[i] with u[i+8], so lanes 2t and 2t+1 hold the two halves and the
// pairing is one __shfl_xor_sync -- registers and lane count stay exactly as
// they are for radix 8, which is what makes this affordable at all.
//
// The output is then redistributed so lane h holds X[8h..8h+7] rather than the
// interleaved X[2m+h].  That is not cosmetic: it makes the scatter's pair
// offset 8M, the same as the gather's, so one swizzle covers both.  With the
// interleaved order the scatter offset is M and the two requirements are
// contradictory.
//
// W16^(2k) = W8^k, so half the internal twiddles are the radix-8 set and only
// c = cos(pi/8) and s = sin(pi/8) are new.
// ---------------------------------------------------------------------------
#define FUFFT_C16 0.92387953251128673848f
#define FUFFT_S16 0.38268343236508977173f

__device__ __forceinline__ float2 shfl_pair(float2 v)
{
   float2 r;
   r.x = __shfl_xor_sync(0xffffffffu, v.x, 1);
   r.y = __shfl_xor_sync(0xffffffffu, v.y, 1);
   return r;
}

// x * W16^K, with SIGN the transform direction.
template<int K, int SIGN>
__device__ __forceinline__ float2 mul_w16(float2 x)
{
   const float c = FUFFT_C16, sn = FUFFT_S16, s = (float)SIGN;
   switch(K) {
      case 0:  return x;
      case 1:  return cmul(x, make_float2( c,   s * sn));
      case 2:  return cmul_w1<SIGN>(x);                     // = W8^1, cheap form
      case 3:  return cmul(x, make_float2( sn,  s * c ));
      case 4:  return cmul_i<SIGN>(x);                      // = W8^2 = -+i, free
      case 5:  return cmul(x, make_float2(-sn,  s * c ));
      case 6:  return cmul_w3<SIGN>(x);                     // = W8^3, cheap form
      default: return cmul(x, make_float2(-c,   s * sn));
   }
}

#define FUFFT_R2STAGE(K)                                                        \
   { float2 p = shfl_pair(u[K]);                                               \
     float2 sum = cadd(u[K], p);                                               \
     float2 dif = mul_w16<K, SIGN>(csub(p, u[K]));                             \
     u[K] = (h == 0) ? sum : dif; }

template<int SIGN, bool DIT>
__device__ __forceinline__ void fufft_pass_r16(float2* A, int M, int lane)
{
   const int h    = lane & 1;
   const int t    = lane >> 1;
   const int j    = t % M;
   const int b    = t / M;
   const int base = b * 16 * M + j;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(16 * M);

   float2 u[8], v[8], o[8];
#pragma unroll
   for(int k = 0; k < 8; k++) u[k] = A[SWZ(base + (h * 8 + k) * M)];

   // Lane h holds X[8h+m] on the way out, so the stage twiddle is
   // W^(j*(8h+m)) = w1^(8h) * w1^m: both lanes run the same recurrence and
   // lane 1 starts eight steps along.  w1^8 by squaring costs less than a
   // second __sincosf.
   float2 w1 = make_float2(1.0f, 0.0f), w8 = w1;
   if(M > 1) {
      w1 = twiddle(step * (float)j);
      float2 w2 = cmul(w1, w1), w4 = cmul(w2, w2);
      w8 = cmul(w4, w4);
   }
   const float2 wstart = (h == 0) ? make_float2(1.0f, 0.0f) : w8;

   if(DIT) {                                   // twiddle first
      float2 w = wstart;
      o[0] = cmul(u[0], w);
#pragma unroll
      for(int m = 1; m < 8; m++) { w = cmul(w, w1); o[m] = cmul(u[m], w); }
#pragma unroll
      for(int k = 0; k < 8; k++) u[k] = o[k];
   }

   FUFFT_R2STAGE(0) FUFFT_R2STAGE(1) FUFFT_R2STAGE(2) FUFFT_R2STAGE(3)
   FUFFT_R2STAGE(4) FUFFT_R2STAGE(5) FUFFT_R2STAGE(6) FUFFT_R2STAGE(7)

   dft8<SIGN>(u, v);

   // redistribute: lane 0 keeps v[0..3] and takes the partner's v[0..3];
   // lane 1 takes the partner's v[4..7] and keeps its own.  Four shuffles.
#pragma unroll
   for(int q = 0; q < 4; q++) {
      float2 snd = (h == 0) ? v[4 + q] : v[q];
      float2 rcv = shfl_pair(snd);
      o[2 * q]     = (h == 0) ? v[q] : rcv;
      o[2 * q + 1] = (h == 0) ? rcv  : v[4 + q];
   }

   if(!DIT && M > 1) {                         // twiddle after the butterfly
      float2 w = wstart;
      o[0] = cmul(o[0], w);
#pragma unroll
      for(int m = 1; m < 8; m++) { w = cmul(w, w1); o[m] = cmul(o[m], w); }
   }

#pragma unroll
   for(int m = 0; m < 8; m++) A[SWZ(base + (h * 8 + m) * M)] = o[m];
}

// ---------------------------------------------------------------------------
// Composable pipeline: rather than hand-copying a kernel per fusion, the
// middle operation is a policy and the buffer strategy is a device function.
// A new variant costs a struct.
//
//   MidNothing      forward then inverse, nothing between (a round trip)
//   MidMultiply     convolution
//   MidConjMultiply correlation
// ---------------------------------------------------------------------------
struct MidNothing {
   __device__ __forceinline__
   void operator()(float2*, int, int, int, size_t) const {}
};

struct MidMultiply {
   const float2* __restrict__ b;
   __device__ __forceinline__
   void operator()(float2* A, int lane, int T, int N, size_t base) const
   {
      for(int i = lane; i < N; i += T) A[SWZ(i)] = cmul(A[SWZ(i)], b[base + i]);
   }
};

struct MidConjMultiply {
   const float2* __restrict__ b;
   __device__ __forceinline__
   void operator()(float2* A, int lane, int T, int N, size_t base) const
   {
      for(int i = lane; i < N; i += T) {
         float2 x = A[SWZ(i)], y = b[base + i];
         A[SWZ(i)] = make_float2(fmaf(x.x, y.x,  x.y * y.y),
                                 fmaf(x.y, y.x, -x.x * y.y));
      }
   }
};

// ---------------------------------------------------------------------------
// radix 16 with sixteen values in one lane (V=16, T = N/16).
//
// The alternative to the lane-pair route.  It costs 32 data registers instead
// of 16 and halves the lane count, but it has no shuffles and no lane
// divergence, and that turns out to matter more: normalised per point per pass
// it is 19 weighted instructions against radix-8's 16 and the lane pair's
// 31.25.  Over a whole N=4096 convolution that is -11% ALU where the lane pair
// is +46%.
//
// It needs its own swizzle.  At M == 1 a lane reads sixteen consecutive slots,
// so lanes are 16 apart and every one lands on the same bank; folding bits 7..10
// into 0..3 spreads them.  The shared layout is private to a kernel -- data
// enters and leaves in natural order -- so a per-kernel swizzle is free.
// ---------------------------------------------------------------------------
#if FUFFT_SWIZZLE
#define SWZ16(i) ((i) ^ (((i) >> 3) & 15) ^ (((i) >> 7) & 15))
#else
#define SWZ16(i) (i)
#endif

template<int SIGN>
__device__ __forceinline__ void dft16(const float2 u[16], float2 v[16])
{
   float2 a[8], b[8], va[8], vb[8];
   a[0] = cadd(u[0], u[ 8]);  b[0] =                    csub(u[0], u[ 8]);
   a[1] = cadd(u[1], u[ 9]);  b[1] = mul_w16<1, SIGN>(csub(u[1], u[ 9]));
   a[2] = cadd(u[2], u[10]);  b[2] = mul_w16<2, SIGN>(csub(u[2], u[10]));
   a[3] = cadd(u[3], u[11]);  b[3] = mul_w16<3, SIGN>(csub(u[3], u[11]));
   a[4] = cadd(u[4], u[12]);  b[4] = mul_w16<4, SIGN>(csub(u[4], u[12]));
   a[5] = cadd(u[5], u[13]);  b[5] = mul_w16<5, SIGN>(csub(u[5], u[13]));
   a[6] = cadd(u[6], u[14]);  b[6] = mul_w16<6, SIGN>(csub(u[6], u[14]));
   a[7] = cadd(u[7], u[15]);  b[7] = mul_w16<7, SIGN>(csub(u[7], u[15]));
   dft8<SIGN>(a, va);
   dft8<SIGN>(b, vb);
#pragma unroll
   for(int k = 0; k < 8; k++) { v[2 * k] = va[k]; v[2 * k + 1] = vb[k]; }
}

template<int R, int SIGN> struct Butterfly16;
template<int SIGN> struct Butterfly16<2, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft2<SIGN>(u, v); } };
template<int SIGN> struct Butterfly16<4, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft4<SIGN>(u, v); } };
template<int SIGN> struct Butterfly16<8, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft8<SIGN>(u, v); } };
template<int SIGN> struct Butterfly16<16, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v) { dft16<SIGN>(u, v); } };

// Stage twiddles for a radix-16 pass, factorised by output parity.
// v[2m] needs W^(2jm) = (W^2)^(jm) and v[2m+1] needs W^j * (W^2)^(jm), so one
// squaring gives two independent 7-step chains instead of one 15-step chain:
// same 29 cmuls, half the dependency depth (ILP), and rounding error from at
// most 8 accumulated steps instead of 15 -- this is what the 4.4e-07 vs
// 3.3e-07 accuracy gap at N=4096 was.
template<int SIGN>
__device__ __forceinline__ void apply_twiddles_16(float2 x[16], float step, int j)
{
   const float2 w1 = twiddle(step * (float)j);
   const float2 w2 = cmul(w1, w1);
   x[1] = cmul(x[1], w1);
   float2 ze = w2, zo = cmul(w2, w1);
#pragma unroll
   for(int m = 1; m < 8; m++) {
      x[2 * m]     = cmul(x[2 * m],     ze);
      x[2 * m + 1] = cmul(x[2 * m + 1], zo);
      if(m < 7) { ze = cmul(ze, w2); zo = cmul(zo, w2); }
   }
}

// V = 16: a radix-R pass runs 16/R butterflies on one lane's sixteen values.
template<int R, int SIGN, bool DIT>
__device__ __forceinline__ void fufft_pass_v16(float2* A, int M, int lane, int T)
{
   const int G = 16 / R;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
#pragma unroll
   for(int g = 0; g < G; g++) {
      const int gidx = lane + g * T;
      const int j    = gidx % M;
      const int b    = gidx / M;
      const int base = b * R * M + j;

      float2 u[R], v[R];
#pragma unroll
      for(int k = 0; k < R; k++) u[k] = A[SWZ16(base + k * M)];
      if(DIT && M > 1) {
         if(R == 16) apply_twiddles_16<SIGN>(u, step, j);
         else        apply_twiddles_r<R, SIGN>(u, step, j);
      }
      Butterfly16<R, SIGN>::go(u, v);
      if(!DIT && M > 1) {
         if(R == 16) apply_twiddles_16<SIGN>(v, step, j);
         else        apply_twiddles_r<R, SIGN>(v, step, j);
      }
#pragma unroll
      for(int k = 0; k < R; k++) A[SWZ16(base + k * M)] = v[k];
   }
}

template<int R, int SIGN, bool DIT> struct LeadPass16 {
   __device__ __forceinline__ static void go(float2* A, int M, int lane, int T)
      { fufft_pass_v16<R, SIGN, DIT>(A, M, lane, T); } };
template<int SIGN, bool DIT> struct LeadPass16<1, SIGN, DIT> {
   __device__ __forceinline__ static void go(float2*, int, int, int) {} };

// N = LEAD * 16^NRV, LEAD in {1,2,4,8}.
// Takes MidMultiply concretely, not a Mid policy: the policy structs index the
// buffer with SWZ, but this kernel's shared layout uses SWZ16.  Calling
// mid(...) here would compile and silently corrupt data, so the multiply is
// inlined with the right swizzle instead.
template<int N, int LEAD, int NRV>
__device__ __forceinline__ void fufft_body_v16(float2* A, const float2* __restrict__ a,
                                              float2* __restrict__ out,
                                              const MidMultiply& mid, int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ16(i)] = a[base + i];
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass16<LEAD, -1, false>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NRV; s++) {
      M /= 16;
      fufft_pass_v16<16, -1, false>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   for(int i = lane; i < N; i += T) A[SWZ16(i)] = cmul(A[SWZ16(i)], mid.b[base + i]);
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 1;
#pragma unroll
   for(int s = 0; s < NRV; s++) {
      fufft_pass_v16<16, +1, true>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 16;
   }
   LeadPass16<LEAD, +1, true>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_INV); }

   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ16(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

// N = LEAD * 8^NR8 with LEAD in {1,2,4}.  LEAD == 1 means a pure power of eight
// and the leading pass disappears entirely.
template<int R, int SIGN> struct LeadPass {
   __device__ __forceinline__ static void dif(float2* A, int M, int lane, int T)
      { fufft_dif_pass_r<R, SIGN>(A, M, lane, T); }
   __device__ __forceinline__ static void dit(float2* A, int M, int lane, int T)
      { fufft_dit_pass_r<R, SIGN>(A, M, lane, T); }
};
template<int SIGN> struct LeadPass<1, SIGN> {
   __device__ __forceinline__ static void dif(float2*, int, int, int) {}
   __device__ __forceinline__ static void dit(float2*, int, int, int) {}
};

// ---------------------------------------------------------------------------
// One Stockham autosort pass, radix 8.
// ---------------------------------------------------------------------------
template<int N, int SIGN>
__device__ __forceinline__ void fufft_pass(const float2* __restrict__ A,
                                          float2* __restrict__ B,
                                          int m, int lane)
{
   const int mh = m / 8;
   const int j  = lane % mh;
   const int i  = lane / mh;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)m;

   float2 u[8], v[8];
#pragma unroll
   for(int k = 0; k < 8; k++)
      u[k] = A[SWZ(i * mh + j + k * (N / 8))];

   // At mh == 1 every lane has j == 0 and every twiddle is W^0 = 1, exactly as
   // in the M == 1 DIF/DIT passes.  Guarded here too so the Stockham and
   // in-place paths are compared on equal terms.
   if(mh > 1) {
#if FUFFT_TWIDDLE_RECURRENCE
   const float2 w1 = twiddle(step * (float)j);
   float2 w = w1;
   u[1] = cmul(u[1], w);
#pragma unroll
   for(int k = 2; k < 8; k++) {
      w    = cmul(w, w1);
      u[k] = cmul(u[k], w);
   }
#else
#pragma unroll
   for(int k = 1; k < 8; k++)
      u[k] = cmul(u[k], twiddle(step * (float)(j * k)));
#endif
   }

   dft8<SIGN>(u, v);

#pragma unroll
   for(int k = 0; k < 8; k++)
      B[SWZ(i * m + j + k * mh)] = v[k];
}

template<int N, int LOG8N, int SIGN>
__device__ __forceinline__ float2* fufft_stages(float2* p, float2* q, int lane)
{
   int m = 1;
#pragma unroll
   for(int s = 0; s < LOG8N; s++) {
      m *= 8;
      fufft_pass<N, SIGN>(p, q, m, lane);
      __syncthreads();
      float2* t = p; p = q; q = t;
   }
   return p;
}

// ---------------------------------------------------------------------------
// Mode A: batched forward C2C.  CPB independent transforms per block so small
// N still fills a block.
// ---------------------------------------------------------------------------
template<int N, int LOG8N, int CPB>
__global__ FUFFT_BOUNDS void fufft_c2c(const float2* __restrict__ in,
                         float2* __restrict__ out,
                         unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;

   float2* A = smem + (size_t)slot * 2 * N;
   float2* B = A + N;
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;

   FUFFT_TIMER_START();
   for(int i = lane; i < N; i += T) A[SWZ(i)] = in[base + i];
   __syncthreads();
   FUFFT_TIMER_MARK(K_LOAD);

   float2* p = fufft_stages<N, LOG8N, -1>(A, B, lane);
   FUFFT_TIMER_MARK(K_FWD);

   for(int i = lane; i < N; i += T) out[base + i] = p[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

// ---------------------------------------------------------------------------
// In-place radix-8 Cooley-Tukey.  One buffer of N instead of Stockham's 2N,
// which takes shared memory from 64 KB to 32 KB per block and occupancy from
// 2 to 3 blocks/SM.
//
//   DIF  natural order in, digit-reversed (base 8) out.  butterfly, then twiddle.
//        stride M runs N/8, N/64, ... 1
//   DIT  digit-reversed in, natural order out.  twiddle, then butterfly.
//        stride M runs 1, 8, ... N/8
//
// Each lane owns the group (b, j) = (lane/M, lane%M) and touches only
// { b*8M + j + k*M : k=0..7 }.  Those sets are disjoint across lanes, so the
// pass is race-free in place and needs a barrier only between passes.
//
// This is used for the convolution only.  A plain forward transform would have
// to unscramble on the way out, and A[SWZ(digitrev(i))] is an 8- to 16-way bank
// conflict even with the swizzle -- so fufft_c2c stays on Stockham.  A
// convolution never needs the natural order in between: the filter spectrum is
// permuted once on the host instead.
// ---------------------------------------------------------------------------
// one transform group, in place, single N buffer.
// N = LEAD * 8^NR8.  DIF applies the leading radix first (largest stride) then
// the radix-8 passes; DIT runs them in reverse, so the pass at M == 1 is always
// radix 8 as long as NR8 >= 1.
template<int N, int LEAD, int NR8, class Mid>
__device__ __forceinline__ void fufft_body_inplace(float2* A, const float2* __restrict__ a,
                                                  float2* __restrict__ out,
                                                  const Mid& mid, int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass<LEAD, -1>::dif(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      M /= 8;
      fufft_dif_pass_r<8, -1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   mid(A, lane, T, N, base);
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 1;
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      fufft_dit_pass_r<8, +1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 8;
   }
   LeadPass<LEAD, +1>::dit(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_INV); }

   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

// Register-resident fwd/inv boundary: multiply the eight per-lane values in
// registers between the forward and inverse transforms, never spilling the
// spectrum to shared.
//
// It works here because the last DIF pass and the first DIT pass both run at
// M == 1, where lane L owns exactly {8L .. 8L+7} in both.  So the scatter that
// ends the forward transform and the gather that starts the inverse are the
// same eight slots, and both can be skipped.  Both also have j == 0, so neither
// has a twiddle.
//
// bperm must be laid out so the filter read stays coalesced:
//     bperm[lane + t*T] = bspec_digitrev[8*lane + t]
// Lane L then reads bperm[base + L + t*T], consecutive lanes hit consecutive
// addresses, 100% of every fetched sector is used.  Reading
// bspec_digitrev[8L+t] directly would be a 64-byte stride and 25% efficiency.
// The permutation folds into the digit-reversal the host already does.
//
// saves 2 LDS + 2 STS per point and two barriers.
template<int N, int LEAD, int NR8>
__device__ __forceinline__ void fufft_body_fused(float2* A, const float2* __restrict__ a,
                                                const float2* __restrict__ bperm,
                                                float2* __restrict__ out,
                                                int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass<LEAD, -1>::dif(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NR8 - 1; s++) {
      M /= 8;
      fufft_dif_pass_r<8, -1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   {                                          // the M == 1 boundary, radix 8
      const int b0 = lane * 8;
      float2 u[8], v[8], w[8];
#pragma unroll
      for(int k = 0; k < 8; k++) u[k] = A[SWZ(b0 + k)];
      dft8<-1>(u, v);
#pragma unroll
      for(int k = 0; k < 8; k++) v[k] = cmul(v[k], bperm[base + lane + k * T]);
      dft8<+1>(v, w);
#pragma unroll
      for(int k = 0; k < 8; k++) A[SWZ(b0 + k)] = w[k];
   }
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 8;
#pragma unroll
   for(int s = 1; s < NR8; s++) {
      fufft_dit_pass_r<8, +1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 8;
   }
   LeadPass<LEAD, +1>::dit(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_INV); }

   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

// radix-16 body.  N = LEAD * 16^NR16 with LEAD in {1,2,4,8}: the leading pass
// uses the ordinary radix-LEAD path (lane count is unchanged, so it composes),
// then NR16 lane-pair radix-16 passes.  One fewer pass than the radix-8 route
// at every size except pure powers of eight.
template<int N, int LEAD, int NR16, class Mid>
__device__ __forceinline__ void fufft_body_r16(float2* A, const float2* __restrict__ a,
                                              float2* __restrict__ out,
                                              const Mid& mid, int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass<LEAD, -1>::dif(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NR16; s++) {
      M /= 16;
      fufft_pass_r16<-1, false>(A, M, lane);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   mid(A, lane, T, N, base);
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 1;
#pragma unroll
   for(int s = 0; s < NR16; s++) {
      fufft_pass_r16<+1, true>(A, M, lane);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 16;
   }
   LeadPass<LEAD, +1>::dit(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 8>(); FUFFT_TIMER_MARK(K_INV); }

   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

// ---------------------------------------------------------------------------
// Large N, four-step shape (step -> step -> fused middle -> istep -> istep).
// An in-place radix-8 DIF peels its M >= 4096 passes first; after
// them every remaining butterfly group lies inside one contiguous 4096-block,
// so each block is an independent standard N=4096 transform.  The fused middle
// is therefore literally fufft_pipeline_ip<4096> with chunks posing as batch
// items -- unchanged -- and only these thin outer passes are new.  The whole
// chain executes the same butterflies in the same order as a monolithic
// in-place transform of N, so it verifies bit-identically against one.
//
// An outer pass works on global memory directly: one thread owns one butterfly
// group (8 taps at stride M), registers only.  Consecutive threads take
// consecutive j, so every tap is coalesced.  Groups are disjoint, so writing
// in place is race-free.  Traffic: one read + one write of the array per
// level, 4*LOUT + 3 passes for a whole convolution.
// ---------------------------------------------------------------------------
template<int R, int SIGN> struct OuterDFT {
   __device__ __forceinline__ static void go(const float2* u, float2* v)
      { dft8<SIGN>(u, v); } };
template<int SIGN> struct OuterDFT<16, SIGN> {
   __device__ __forceinline__ static void go(const float2* u, float2* v)
      { dft16<SIGN>(u, v); } };

// Radix-R outer pass, R in {8,16}.  The four-step originally hardcoded radix 8
// here AND used the radix-8 in-place kernel as its middle, so the V=16 kernel
// that ties cuFFTDx was never on the large-N path at all.  Radix 16 matters
// because it changes the LEVEL COUNT: with an 8192-point middle, a 2M transform
// has outer factor 256 = 16^2, clearing in two levels instead of three --
// 15 passes down to 11, which is the whole large-N deficit.
// Digit order for the chained transform is bit-group reversal over the radix
// widths in pass order (verify_kernel.cpp: chain_rev); the four-step is checked
// against a monolithic transform built from the same radix sequence.
template<int R, int SIGN, bool DIT>
__global__ FUFFT_BOUNDS void fufft_outer_pass_r(float2* __restrict__ A,
                                              int M, size_t ngroups_total)
{
   const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(g >= ngroups_total) return;

   // groups are numbered within a transform of size 8*M*(groups/transform);
   // the batch dimension is folded into g, and because a transform holds a
   // whole number of groups the j/b split below stays correct across it.
   const size_t j    = g % (size_t)M;
   const size_t b    = g / (size_t)M;
   const size_t base = b * (size_t)R * (size_t)M + j;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);

   float2 u[R], v[R];
#pragma unroll
   for(int k = 0; k < R; k++) u[k] = A[base + (size_t)k * M];
   if(DIT) {
      if(M > 1) apply_twiddles_r<R, SIGN>(u, step, (int)j);
      OuterDFT<R, SIGN>::go(u, v);
   } else {
      OuterDFT<R, SIGN>::go(u, v);
      if(M > 1) apply_twiddles_r<R, SIGN>(v, step, (int)j);
   }
#pragma unroll
   for(int k = 0; k < R; k++) A[base + (size_t)k * M] = v[k];
}

// keeps the original radix-8 entry point working unchanged
template<int SIGN, bool DIT>
__global__ FUFFT_BOUNDS void fufft_outer_pass(float2* __restrict__ A,
                                            int M, size_t ngroups_total)
{
   const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(g >= ngroups_total) return;
   const size_t j    = g % (size_t)M;
   const size_t b    = g / (size_t)M;
   const size_t base = b * 8u * (size_t)M + j;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(8 * M);
   float2 u[8], v[8];
#pragma unroll
   for(int k = 0; k < 8; k++) u[k] = A[base + (size_t)k * M];
   if(DIT) {
      if(M > 1) apply_twiddles_r<8, SIGN>(u, step, (int)j);
      dft8<SIGN>(u, v);
   } else {
      dft8<SIGN>(u, v);
      if(M > 1) apply_twiddles_r<8, SIGN>(v, step, (int)j);
   }
#pragma unroll
   for(int k = 0; k < 8; k++) A[base + (size_t)k * M] = v[k];
}

// ---------------------------------------------------------------------------
// 2D convolution, fully fused for tiles with H*W <= 4096 (H, W powers of 8).
//
// A 2D DFT is DFT_H (x) DFT_W with NO twiddles between the dimensions, so the
// row levels are the existing chunk-local passes (fufft_dif_pass_r already
// keeps every group inside one W-chunk when 8M <= W) and only a column pass is
// new: same radix-8 butterfly, taps at stride W*m, twiddle period 8m indexed
// by the row position within the column.  The bank audit shows the shared
// swizzle stays 1-way for both patterns at every (H, W) split of 4096.
//
// The whole conv -- row FFT, column FFT, pointwise, column IFFT, row IFFT --
// is one kernel and 3 HBM passes.  cuFFT's 2D plan runs row and column
// kernels separately, so its chain is 7-11 passes; this is the best ratio in
// the project, and it lands on the FNO tile size (64x64).
//
// The filter is prescrambled per dimension: slot (r, c) holds spectrum
// (rev8_H(r), rev8_W(c)).
// ---------------------------------------------------------------------------
template<int W, int SIGN, bool DIT>
__device__ __forceinline__ void fufft_col_pass(float2* A, int m, int lane)
{
   const int c    = lane % W;
   const int g2   = lane / W;
   const int j2   = g2 % m;
   const int b2   = g2 / m;
   const int base = c + W * (b2 * 8 * m + j2);
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(8 * m);

   float2 u[8], v[8];
#pragma unroll
   for(int k = 0; k < 8; k++) u[k] = A[SWZ(base + k * W * m)];
   if(DIT) {
      if(m > 1) apply_twiddles_r<8, SIGN>(u, step, j2);
      dft8<SIGN>(u, v);
   } else {
      dft8<SIGN>(u, v);
      if(m > 1) apply_twiddles_r<8, SIGN>(v, step, j2);
   }
#pragma unroll
   for(int k = 0; k < 8; k++) A[SWZ(base + k * W * m)] = v[k];
}

template<int H, int W, int CPB>
__global__ FUFFT_BOUNDS void fufft_conv2d(const float2* __restrict__ a,
                                        float2* __restrict__ out,
                                        MidMultiply mid,
                                        unsigned int* __restrict__ timing)
{
   constexpr int NP = H * W;
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = NP / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * NP;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * NP;

   for(int i = lane; i < NP; i += T) A[SWZ(i)] = a[base + i];
   fufft_bar<NP / 8>();
   FUFFT_TIMER_MARK(K_LOAD);

   for(int M = W / 8; M >= 1; M /= 8) {           // row forward
      fufft_dif_pass_r<8, -1>(A, M, lane, T);
      fufft_bar<NP / 8>();
   }
   for(int m = H / 8; m >= 1; m /= 8) {           // column forward
      fufft_col_pass<W, -1, false>(A, m, lane);
      fufft_bar<NP / 8>();
   }
   FUFFT_TIMER_MARK(K_FWD);

   for(int i = lane; i < NP; i += T)
      A[SWZ(i)] = cmul(A[SWZ(i)], mid.b[base + i]);
   fufft_bar<NP / 8>();
   FUFFT_TIMER_MARK(K_MULT);

   for(int m = 1; m <= H / 8; m *= 8) {           // column inverse
      fufft_col_pass<W, +1, true>(A, m, lane);
      fufft_bar<NP / 8>();
   }
   for(int M = 1; M <= W / 8; M *= 8) {           // row inverse
      fufft_dit_pass_r<8, +1>(A, M, lane, T);
      fufft_bar<NP / 8>();
   }
   FUFFT_TIMER_MARK(K_INV);

   for(int i = lane; i < NP; i += T) out[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

// ---------------------------------------------------------------------------
// Two-for-one real convolution.  Convolution with a REAL filter commutes with
// complex packing -- (x + iy) (*) h = (x (*) h) + i (y (*) h) by linearity --
// so two real signals ride one complex transform and the kernel arithmetic is
// exactly fufft_body_inplace: only the load packs re/im from two real arrays
// and the store unpacks them.  Per real conv the HBM traffic halves (12N bytes
// against 24N for treating each signal as complex), which is the whole point:
// the pass count already sits at its floor, so the remaining lever is bytes.
//
// The filter must be real in the time domain (its spectrum is used as-is; no
// conjugate-symmetric unpacking is ever needed).  A complex filter silently
// mixes the two results -- documented, and the verifier pins the identity.
// ---------------------------------------------------------------------------
template<int N, int LEAD, int NR8, int CPB>
__global__ FUFFT_BOUNDS void fufft_conv_r2(const float* __restrict__ xr,
                                         const float* __restrict__ yr,
                                         const float2* __restrict__ bspec_dr,
                                         float* __restrict__ ur,
                                         float* __restrict__ vr,
                                         unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;

   for(int i = lane; i < N; i += T)
      A[SWZ(i)] = make_float2(xr[base + i], yr[base + i]);
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass<LEAD, -1>::dif(A, M, lane, T);
   if(LEAD > 1) fufft_bar<N / 8>();
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      M /= 8;
      fufft_dif_pass_r<8, -1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   for(int i = lane; i < N; i += T)
      A[SWZ(i)] = cmul(A[SWZ(i)], bspec_dr[base + i]);
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 1;
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      fufft_dit_pass_r<8, +1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 8;
   }
   LeadPass<LEAD, +1>::dit(A, M, lane, T);
   if(LEAD > 1) fufft_bar<N / 8>();

   for(int i = lane; i < N; i += T) {
      const float2 w = A[SWZ(i)];
      ur[base + i] = w.x;
      vr[base + i] = w.y;
   }
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

// Split entry points for the one-signal-many-filters shape: precompute the
// recurring operand's spectrum once, then each later convolution runs only
// multiply + inverse.  This beats recomputing the forward transform every
// time: the spectrum is 8N bytes (32KB at N=4096), it stays in L2, and the
// per-filter cost falls to an effective two HBM passes -- below the 2K+2 a
// cuFFT chain needs.
//
// fwd:   natural order in -> scrambled (digit-reversed) spectrum out.
// apply: scrambled spectrum + scrambled filter -> pointwise -> natural out.
// fwd then apply is bit-identical to the fused conv: the extra global round
// trip stores exact values.
template<int N, int LEAD, int NR8, int CPB>
__global__ FUFFT_BOUNDS void fufft_pipeline_fwd(const float2* __restrict__ a,
                                              float2* __restrict__ spec,
                                              unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_LOAD);
   int M = N / LEAD;
   LeadPass<LEAD, -1>::dif(A, M, lane, T);
   if(LEAD > 1) fufft_bar<N / 8>();
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      M /= 8;
      fufft_dif_pass_r<8, -1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_FWD);
   }
   for(int i = lane; i < N; i += T) spec[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LEAD, int NR8, int CPB>
__global__ FUFFT_BOUNDS void fufft_pipeline_apply(const float2* __restrict__ spec,
                                                const float2* __restrict__ bspec_dr,
                                                float2* __restrict__ out,
                                                unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   for(int i = lane; i < N; i += T)
      A[SWZ(i)] = cmul(spec[base + i], bspec_dr[base + i]);
   fufft_bar<N / 8>();
   FUFFT_TIMER_MARK(K_MULT);
   int M = 1;
#pragma unroll
   for(int s = 0; s < NR8; s++) {
      fufft_dit_pass_r<8, +1>(A, M, lane, T);
      fufft_bar<N / 8>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 8;
   }
   LeadPass<LEAD, +1>::dit(A, M, lane, T);
   if(LEAD > 1) fufft_bar<N / 8>();
   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

// one transform group, Stockham ping-pong, 2N buffer
template<int N, int LOG8N, class Mid>
__device__ __forceinline__ void fufft_body_stockham(float2* A, float2* B,
                                                   const float2* __restrict__ a,
                                                   float2* __restrict__ out,
                                                   const Mid& mid, int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
   __syncthreads();
   FUFFT_TIMER_MARK(K_LOAD);

   float2* p = fufft_stages<N, LOG8N, -1>(A, B, lane);
   float2* q = (p == A) ? B : A;
   FUFFT_TIMER_MARK(K_FWD);

   mid(p, lane, T, N, base);
   __syncthreads();
   FUFFT_TIMER_MARK(K_MULT);

   p = fufft_stages<N, LOG8N, +1>(p, q, lane);
   FUFFT_TIMER_MARK(K_INV);

   for(int i = lane; i < N; i += T) out[base + i] = p[SWZ(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

// V=16 with the register-resident boundary, the same trick fufft_body_fused
// plays for radix 8: at M == 1 a lane owns the sixteen consecutive slots
// {16L..16L+15} in both the last DIF pass and the first DIT pass, so the
// scatter, the separate pointwise pass and the gather between them collapse
// into register arithmetic.  Shared accesses drop from 8 LDS + 8 STS per point
// to 5 + 5, and three barriers disappear.
//
// bperm is the filter laid out so lane L reads position 16L+k at [L + k*T]:
// consecutive lanes hit consecutive addresses, fully coalesced.
//
// Register note: x[16] + y[16] alone are 64 registers and the dft16 internals
// briefly add more; FUFFT_BOUNDS16 caps at 85, so watch the build log's spill
// check before trusting timings.
template<int N, int LEAD, int NRV>
__device__ __forceinline__ void fufft_body_v16f(float2* A, const float2* __restrict__ a,
                                               const float2* __restrict__ bperm,
                                               float2* __restrict__ out,
                                               int lane, int T, size_t base)
{
   for(int i = lane; i < N; i += T) A[SWZ16(i)] = a[base + i];
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass16<LEAD, -1, false>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NRV - 1; s++) {
      M /= 16;
      fufft_pass_v16<16, -1, false>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   {                                          // M == 1 boundary in registers
      const int b0 = lane * 16;
      float2 x[16], y[16];
#pragma unroll
      for(int k = 0; k < 16; k++) x[k] = A[SWZ16(b0 + k)];
      dft16<-1>(x, y);
#pragma unroll
      for(int k = 0; k < 16; k++) y[k] = cmul(y[k], bperm[base + lane + k * T]);
      dft16<+1>(y, x);
#pragma unroll
      for(int k = 0; k < 16; k++) A[SWZ16(b0 + k)] = x[k];
   }
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 16;
#pragma unroll
   for(int s = 1; s < NRV; s++) {
      fufft_pass_v16<16, +1, true>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 16;
   }
   LeadPass16<LEAD, +1, true>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_INV); }

   for(int i = lane; i < N; i += T) out[base + i] = A[SWZ16(i)];
   FUFFT_TIMER_MARK(K_STORE);
}

template<int N, int LEAD, int NRV, int CPB>
__global__ FUFFT_BOUNDS16 void fufft_pipeline_v16f(const float2* __restrict__ a,
                                                 const float2* __restrict__ bperm,
                                                 float2* __restrict__ out,
                                                 unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 16;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_v16f<N, LEAD, NRV>(A, a, bperm, out, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LEAD, int NRV, int CPB>
__global__ FUFFT_BOUNDS16 void fufft_pipeline_v16(const float2* __restrict__ a,
                                              float2* __restrict__ out,
                                              MidMultiply mid, unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 16;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_v16<N, LEAD, NRV>(A, a, out, mid, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}


// ---------------------------------------------------------------------------
// int16 block-floating-point convolution -- the bytes lever.
//
// The A10 measurements closed the fp32 question: our V=16 ties cuFFTDx at the
// bandwidth wall (99.3% of measured ceiling), and the backend A/B showed no
// code change moves a wall-bound kernel.  The only way past the wall is fewer
// bytes.  This kernel stores the HBM tensors as int16 pairs (4 B/point) with
// one fp32 scale per transform: half the traffic of any fp32 kernel,
// cuFFTDx included.
//
// Numerics: quantisation lives ONLY at the HBM boundary; shared memory and
// every butterfly stay fp32, so accuracy is strictly better than the
// host-verified full-int16 pipeline bound of 13.3-14.0 effective bits
// cuFFTDx's comparable move is fp16 storage at the same
// 4 B/point but ~11 bits.  The output scale is computed per transform with an
// in-shared abs-max reduction -- no extra HBM pass.
//
// Registered expectations (falsifiable on the next GPU session):
//   - vs fp32 V=16: 1.4-2.0x.  2.0x is the traffic ratio; the dequant/quant
//     ALU sits below the wall, so the backend result predicts the gap between
//     1.4 and 2.0 is compiler scheduling -- the PTX backend may matter here.
//   - accuracy: >= 13 effective bits end to end.
// ---------------------------------------------------------------------------

// BFP I/O primitives.  The A10 run measures the BFP kernel at 96-97% of the
// byte ceiling (N=512-2048; 93% at N=4096) -- the last big bite was replacing
// per-thread shared atomicMax with a warp tree in the output-scale reduction.
// What remains at 4096 is the quant rendezvous itself: one extra smem pass
// plus a barrier that the fp32 kernels don't need.
//
// bfp_quant2: PTX float->s16 conversion saturates by ISA definition, so one
// cvt.rni.s16.f32 replaces the clamp/clamp/round/truncate chain per component.
// Numerically identical to the C++ path everywhere reachable: divergence would
// need |v|*inv >= 32767.5, and inv is built from the block abs-max so
// |v|*inv <= 32767*(1+2eps).  The C++ fallback keeps all backends building.
__device__ __forceinline__ unsigned int bfp_quant2(float2 v, float inv)
{
#if FUFFT_BACKEND == 2
   unsigned int r;
   asm("{\n\t"
       ".reg .f32 fx, fy;\n\t"
       ".reg .s16 sx, sy;\n\t"
       "mul.f32 fx, %1, %3;\n\t"
       "mul.f32 fy, %2, %3;\n\t"
       "cvt.rni.s16.f32 sx, fx;\n\t"
       "cvt.rni.s16.f32 sy, fy;\n\t"
       "mov.b32 %0, {sx, sy};\n\t"
       "}" : "=r"(r) : "f"(v.x), "f"(v.y), "f"(inv));
   return r;
#else
   const short qx = (short)__float2int_rn(fminf(fmaxf(v.x * inv, -32767.0f), 32767.0f));
   const short qy = (short)__float2int_rn(fminf(fmaxf(v.y * inv, -32767.0f), 32767.0f));
   return ((unsigned int)(unsigned short)qy << 16) | (unsigned short)qx;
#endif
}

__device__ __forceinline__ float2 bfp_dequant2(unsigned int w, float scale)
{
#if FUFFT_BACKEND == 2
   float2 v;
   asm("{\n\t"
       ".reg .s16 sx, sy;\n\t"
       ".reg .f32 fx, fy;\n\t"
       "mov.b32 {sx, sy}, %2;\n\t"
       "cvt.rn.f32.s16 fx, sx;\n\t"
       "cvt.rn.f32.s16 fy, sy;\n\t"
       "mul.f32 %0, fx, %3;\n\t"
       "mul.f32 %1, fy, %3;\n\t"
       "}" : "=&f"(v.x), "=&f"(v.y) : "r"(w), "f"(scale));
   return v;
#else
   const short qx = (short)(w & 0xffff);
   const short qy = (short)(w >> 16);
   return make_float2(qx * scale, qy * scale);
#endif
}

template<int N, int LEAD, int NRV, int CPB>
__global__ FUFFT_BOUNDS16 void fufft_pipeline_v16_bfp(
      const short2* __restrict__ aq, const float* __restrict__ a_scale,
      const short2* __restrict__ bq, const float* __restrict__ b_scale,
      short2* __restrict__ outq, float* __restrict__ out_scale,
      unsigned int* __restrict__ timing)
{
   static_assert(CPB <= 16, "abs-max scratch sized for CPB <= 16");
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   __shared__ int s_absmax[16];        // fp32 bits; int atomicMax is monotonic for x >= 0
   const int T    = N / 16;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t tix  = (size_t)blockIdx.x * CPB + slot;
   const size_t base = tix * N;

   // load + dequantise: 4 B/point off HBM, fp32 into shared.  Both scales
   // load here: sb's 4 B ride the same latency as the bulk loads instead of
   // stalling the multiply loop behind a barrier later.
   const float sa = a_scale[tix];
   const float sb = b_scale[tix];
   const unsigned int* aw = reinterpret_cast<const unsigned int*>(aq);
   for(int i = lane; i < N; i += T)
      A[SWZ16(i)] = bfp_dequant2(aw[base + i], sa);
   if(lane == 0) s_absmax[slot] = 0;
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_LOAD);

   int M = N / LEAD;
   LeadPass16<LEAD, -1, false>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_FWD); }
#pragma unroll
   for(int s = 0; s < NRV; s++) {
      M /= 16;
      fufft_pass_v16<16, -1, false>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_FWD);
   }

   const unsigned int* bw = reinterpret_cast<const unsigned int*>(bq);
   for(int i = lane; i < N; i += T)
      A[SWZ16(i)] = cmul(A[SWZ16(i)], bfp_dequant2(bw[base + i], sb));
   fufft_bar<N / 16>();
   FUFFT_TIMER_MARK(K_MULT);

   M = 1;
#pragma unroll
   for(int s = 0; s < NRV; s++) {
      fufft_pass_v16<16, +1, true>(A, M, lane, T);
      fufft_bar<N / 16>();
      FUFFT_TIMER_MARK(K_INV);
      M *= 16;
   }
   LeadPass16<LEAD, +1, true>::go(A, M, lane, T);
   if(LEAD > 1) { fufft_bar<N / 16>(); FUFFT_TIMER_MARK(K_INV); }

   // per-transform block scale, in shared -- no extra HBM pass
   // order-independent reduction: walk the slot linearly as float4
   // (2 points per trip, no swizzle math, conflict-free consecutive banks)
   float m = 0.0f;
   const float4* A4 = reinterpret_cast<const float4*>(A);
   for(int v4 = lane; v4 < N / 2; v4 += T) {
      const float4 v = A4[v4];
      m = fmaxf(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                      fmaxf(fabsf(v.z), fabsf(v.w))), m);
   }
   // warp tree first, then ONE shared atomic per warp: T threads aiming
   // atomicMax at the same address serialise (256 deep at N = 4096, which is
   // exactly where the byte-ceiling gap was widest).  T is a multiple of 32,
   // so a warp never straddles slots and the full mask is safe.
#pragma unroll
   for(int off = 16; off; off >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
   if((lane & 31) == 0) atomicMax(&s_absmax[slot], __float_as_int(m));
   fufft_bar<N / 16>();
   const float mx  = fmaxf(__int_as_float(s_absmax[slot]), 1e-30f);
   const float inv = 32767.0f / mx;
   unsigned int* ow = reinterpret_cast<unsigned int*>(outq);
   for(int i = lane; i < N; i += T)
      ow[base + i] = bfp_quant2(A[SWZ16(i)], inv);
   if(lane == 0) out_scale[tix] = mx / 32767.0f;
   FUFFT_TIMER_MARK(K_STORE);
   FUFFT_TIMER_DUMP(timing);
}

// V=16 with the FULL 512-thread bound.  FUFFT_BOUNDS16 is MAX_THREADS/2 = 256,
// which is right when V=16 halves the lane count of a <=4096 transform -- but a
// middle of 8192 needs T = 8192/16 = 512 lanes and the 256 bound makes the
// launch invalid.  The four-step's large-N path uses this one.
// min-blocks 1, not FUFFT_MIN_BLOCKS: a 512-thread kernel holding a 64 KB
// middle can never have 3 resident blocks (192 KB > any SM), and asking for 3
// makes ptxas throttle registers and spill -- measured 20.6 ms vs 15.3 for the
// kernel it was meant to beat.
#if defined(FUFFT_MAX_THREADS)
#define FUFFT_BOUNDS16_WIDE __launch_bounds__(FUFFT_MAX_THREADS, 1)
#else
#define FUFFT_BOUNDS16_WIDE
#endif

template<int N, int LEAD, int NRV, int CPB>
__global__ FUFFT_BOUNDS16_WIDE void fufft_pipeline_v16_wide(const float2* __restrict__ a,
                                                   float2* __restrict__ out,
                                                   MidMultiply mid,
                                                   unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 16;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_v16<N, LEAD, NRV>(A, a, out, mid, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LEAD, int NR16, int CPB, class Mid>
__global__ FUFFT_BOUNDS void fufft_pipeline_r16(const float2* __restrict__ a,
                                              float2* __restrict__ out,
                                              Mid mid, unsigned int* __restrict__ timing)
{
   // fufft_pass_r16 shuffles with the full warp mask, so every warp must be
   // complete and lane parity must equal thread parity.
   static_assert((N / 8) * CPB % 32 == 0, "r16 needs whole warps");
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_r16<N, LEAD, NR16, Mid>(A, a, out, mid, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LEAD, int NR8, int CPB, class Mid>
__global__ FUFFT_BOUNDS void fufft_pipeline_ip(const float2* __restrict__ a, float2* __restrict__ out,
                                 Mid mid, unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_inplace<N, LEAD, NR8, Mid>(A, a, out, mid, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LEAD, int NR8, int CPB>
__global__ FUFFT_BOUNDS void fufft_pipeline_fused(const float2* __restrict__ a,
                                    const float2* __restrict__ bperm,
                                    float2* __restrict__ out,
                                    unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_fused<N, LEAD, NR8>(A, a, bperm, out, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

template<int N, int LOG8N, int CPB, class Mid>
__global__ FUFFT_BOUNDS void fufft_pipeline_st(const float2* __restrict__ a, float2* __restrict__ out,
                                 Mid mid, unsigned int* __restrict__ timing)
{
   FUFFT_TIMER_DECL();
   extern __shared__ float2 smem[];
   const int T    = N / 8;
   const int slot = threadIdx.x / T;
   const int lane = threadIdx.x % T;
   float2* A = smem + (size_t)slot * 2 * N;
   float2* B = A + N;

   FUFFT_TIMER_START();
   const size_t base = ((size_t)blockIdx.x * CPB + slot) * N;
   fufft_body_stockham<N, LOG8N, Mid>(A, B, a, out, mid, lane, T, base);
   FUFFT_TIMER_DUMP(timing);
}

__global__ void pointwise_mul(float2* __restrict__ spec,
                              const float2* __restrict__ bspec,
                              size_t n)
{
   size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(i < n) {
      float2 a = spec[i], b = bspec[i];
      spec[i] = make_float2(fmaf(a.x, b.x, -a.y * b.y),
                            fmaf(a.x, b.y,  a.y * b.x));
   }
}

inline const char* fufft_backend_name(void)
{
#if   FUFFT_BACKEND == FUFFT_CPP_NAIVE
   return "cpp-naive (generic cmul for every twiddle)";
#elif FUFFT_BACKEND == FUFFT_CPP_OPT
   return "cpp-opt   (xi free, x w^1/w^3 as 2mul+2add)";
#elif FUFFT_BACKEND == FUFFT_PTX
   return "ptx       (same specialisations, inline asm)";
#else
   return "unknown";
#endif
}
