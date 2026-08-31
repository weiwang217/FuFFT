// CPU transcription of fufft.cuh.  Checks (a) the index arithmetic, ping-pong
// parity and shared-memory slot layout, and (b) that the CPP_NAIVE and CPP_OPT
// backends agree bit-for-bit-ish, i.e. that the algebraic specialisations of
// x(+-i), x w^1 and x w^3 are correct.  The PTX backend cannot run here, but it
// is a transcription of CPP_OPT and is checked against it by inspection plus
// the on-device accuracy report in bench_fufft.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <random>

typedef std::complex<float>  cf;
typedef std::complex<double> cd;
static const double PI = 3.14159265358979323846;
static const float  R2 = 0.70710678118654752440f;

// must match FUFFT_SWIZZLE / SWZ in fufft.cuh
#define SWZ(i) ((i) ^ (((i) >> 3) & 15) ^ (((i) >> 4) & 8) ^ (((i) >> 8) & 8))

enum { CPP_NAIVE = 0, CPP_OPT = 1 };

// must match FUFFT_TWIDDLE_RECURRENCE in fufft.cuh
#ifndef TWIDDLE_RECURRENCE
#define TWIDDLE_RECURRENCE 1
#endif

// ---- primitives ----------------------------------------------------------

static cf dev_twiddle(float ang) { return cf(cosf(ang), sinf(ang)); }

static cf dev_cmul(cf a, cf b)
{
   return cf(fmaf(a.real(), b.real(), -a.imag() * b.imag()),
             fmaf(a.real(), b.imag(),  a.imag() * b.real()));
}

static cf dev_cmul_i(cf a, int SIGN, int backend)
{
   if(backend == CPP_NAIVE) return dev_cmul(a, cf(0.0f, (float)SIGN));
   return (SIGN < 0) ? cf( a.imag(), -a.real())
                     : cf(-a.imag(),  a.real());
}

static cf dev_cmul_w1(cf a, int SIGN, int backend)
{
   if(backend == CPP_NAIVE) return dev_cmul(a, cf(R2, (float)SIGN * R2));
   const float x = a.real(), y = a.imag();
   return (SIGN < 0) ? cf(R2 * (x + y), R2 * (y - x))
                     : cf(R2 * (x - y), R2 * (x + y));
}

static cf dev_cmul_w3(cf a, int SIGN, int backend)
{
   if(backend == CPP_NAIVE) return dev_cmul(a, cf(-R2, (float)SIGN * R2));
   const float x = a.real(), y = a.imag();
   return (SIGN < 0) ? cf(R2 * (y - x), R2 * (-x - y))
                     : cf(R2 * (-x - y), R2 * (x - y));
}

static void dev_dft8(const cf u[8], cf v[8], int SIGN, int bk)
{
   cf a0 = u[0] + u[4];
   cf a1 = u[1] + u[5];
   cf a2 = u[2] + u[6];
   cf a3 = u[3] + u[7];
   cf a4 = u[0] - u[4];
   cf a5 = dev_cmul_w1(u[1] - u[5], SIGN, bk);
   cf a6 = dev_cmul_i (u[2] - u[6], SIGN, bk);
   cf a7 = dev_cmul_w3(u[3] - u[7], SIGN, bk);

   cf b0 = a0 + a2;
   cf b1 = a1 + a3;
   cf b2 = a0 - a2;
   cf b3 = dev_cmul_i(a1 - a3, SIGN, bk);
   cf b4 = a4 + a6;
   cf b5 = a5 + a7;
   cf b6 = a4 - a6;
   cf b7 = dev_cmul_i(a5 - a7, SIGN, bk);

   v[0] = b0 + b1;  v[4] = b0 - b1;
   v[2] = b2 + b3;  v[6] = b2 - b3;
   v[1] = b4 + b5;  v[5] = b4 - b5;
   v[3] = b6 + b7;  v[7] = b6 - b7;
}

static void dev_pass(int N, const cf* A, cf* B, int m, int SIGN, int bk, int lane)
{
   const int mh = m / 8;
   const int j  = lane % mh;
   const int i  = lane / mh;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)m;

   cf u[8], v[8];
   for(int k = 0; k < 8; k++) u[k] = A[SWZ(i * mh + j + k * (N / 8))];
   if(mh > 1) {
#if TWIDDLE_RECURRENCE
   const cf w1 = dev_twiddle(step * (float)j);
   cf w = w1;
   u[1] = dev_cmul(u[1], w);
   for(int k = 2; k < 8; k++) { w = dev_cmul(w, w1); u[k] = dev_cmul(u[k], w); }
#else
   for(int k = 1; k < 8; k++) u[k] = dev_cmul(u[k], dev_twiddle(step * (float)(j * k)));
#endif
   }
   dev_dft8(u, v, SIGN, bk);
   for(int k = 0; k < 8; k++) B[SWZ(i * m + j + k * mh)] = v[k];
}

static cf* dev_stages(int N, int LOG8N, cf* p, cf* q, int SIGN, int bk, int T)
{
   int m = 1;
   for(int s = 0; s < LOG8N; s++) {
      m *= 8;
      for(int lane = 0; lane < T; lane++) dev_pass(N, p, q, m, SIGN, bk, lane);
      cf* t = p; p = q; q = t;
   }
   return p;
}

// ---- in-place DIF/DIT transcription (fufft_conv_ip) -------------------------

// N = LEAD * 8^NR8, LEAD in {1,2,4}
// N = LEAD * 16^NR16, LEAD in {1,2,4,8}; 0 passes means unsupported
static void factorise16(int N, int& LEAD, int& NR16)
{
   int m = 0; while((1 << m) < N) m++;
   NR16 = m / 4;
   LEAD = 1 << (m % 4);
}

static void factorise(int N, int& LEAD, int& NR8)
{
   int m = 0; while((1 << m) < N) m++;
   NR8 = m / 3;
   const int r = m % 3;
   LEAD = (r == 0) ? 1 : (r == 1 ? 2 : 4);
}

// The order a DIF leaves behind.  Every radix is a power of two, so this is a
// reversal of bit *groups*: R[0] is the top group because the first pass splits
// by the most significant digit.  With LEAD == 1 it degenerates to base-8
// digit reversal.
static int mixedrev(int idx, int LEAD, int NR8)
{
   int w[16], p = 0, m = 0;
   if(LEAD > 1) { int lw = 0; while((1 << lw) < LEAD) lw++; w[p++] = lw; m += lw; }
   for(int i = 0; i < NR8; i++) { w[p++] = 3; m += 3; }

   int d[16], pos = m;
   for(int i = 0; i < p; i++) { pos -= w[i]; d[i] = (idx >> pos) & ((1 << w[i]) - 1); }
   int out = 0, outpos = m;
   for(int i = p - 1; i >= 0; i--) { outpos -= w[i]; out |= d[i] << outpos; }
   return out;
}

static void dev_dft2(const cf u[2], cf v[2])
{
   v[0] = u[0] + u[1];
   v[1] = u[0] - u[1];
}

static void dev_dft4(const cf u[4], cf v[4], int SIGN, int bk)
{
   cf a0 = u[0] + u[2], a1 = u[1] + u[3];
   cf a2 = u[0] - u[2];
   cf a3 = dev_cmul_i(u[1] - u[3], SIGN, bk);
   v[0] = a0 + a1;  v[2] = a0 - a1;
   v[1] = a2 + a3;  v[3] = a2 - a3;
}

static void dev_dftR(int R, const cf* u, cf* v, int SIGN, int bk)
{
   if(R == 2) dev_dft2(u, v);
   else if(R == 4) dev_dft4(u, v, SIGN, bk);
   else dev_dft8(u, v, SIGN, bk);
}

static void dev_apply_tw_r(int R, cf* x, float step, int j, int bk)
{
   (void)bk;
#if TWIDDLE_RECURRENCE
   const cf w1 = dev_twiddle(step * (float)j);
   cf w = w1;
   x[1] = dev_cmul(x[1], w);
   for(int k = 2; k < R; k++) { w = dev_cmul(w, w1); x[k] = dev_cmul(x[k], w); }
#else
   for(int k = 1; k < R; k++) x[k] = dev_cmul(x[k], dev_twiddle(step * (float)(j * k)));
#endif
}

// radix-R pass, lane count fixed at T = N/8, G = 8/R groups per lane
static void dev_pass_r(cf* A, int R, int M, int SIGN, int bk, int T, bool dit)
{
   const int G = 8 / R;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
   for(int lane = 0; lane < T; lane++)
      for(int s = 0; s < G; s++) {
         const int gidx = lane + s * T;
         const int j = gidx % M, b = gidx / M, base = b * R * M + j;
         cf u[8], v[8];
         for(int k = 0; k < R; k++) u[k] = A[SWZ(base + k * M)];
         if(dit && M > 1) dev_apply_tw_r(R, u, step, j, bk);
         dev_dftR(R, u, v, SIGN, bk);
         if(!dit && M > 1) dev_apply_tw_r(R, v, step, j, bk);
         for(int k = 0; k < R; k++) A[SWZ(base + k * M)] = v[k];
      }
}

static void kernel_pipeline_ip(int N, int LEAD, int NR8, int CPB, int bk,
                               const std::vector<cf>& a, const std::vector<cf>& bspec_dr,
                               std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(a.size(), cf(0, 0));

   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];

         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_r(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NR8; s++) { M /= 8; dev_pass_r(A, 8, M, -1, bk, T, false); }

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ(i)] = dev_cmul(A[SWZ(i)], bspec_dr[base + i]);

         M = 1;
         for(int s = 0; s < NR8; s++) { dev_pass_r(A, 8, M, +1, bk, T, true); M *= 8; }
         if(LEAD > 1) dev_pass_r(A, LEAD, M, +1, bk, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
      }
}

// Register-resident fwd/inv boundary.  bperm[lane + t*T] = bspec_dr[8*lane + t].
static void kernel_pipeline_fused(int N, int LEAD, int NR8, int CPB,
                                  const std::vector<cf>& a, const std::vector<cf>& bperm,
                                  std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(a.size(), cf(0, 0));

   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];

         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_r(A, LEAD, M, -1, CPP_OPT, T, false);
         for(int s = 0; s < NR8 - 1; s++) { M /= 8; dev_pass_r(A, 8, M, -1, CPP_OPT, T, false); }

         for(int lane = 0; lane < T; lane++) {      // the M == 1 boundary
            const int b0 = lane * 8;
            cf u[8], v[8], w[8];
            for(int k = 0; k < 8; k++) u[k] = A[SWZ(b0 + k)];
            dev_dft8(u, v, -1, CPP_OPT);
            for(int k = 0; k < 8; k++) v[k] = dev_cmul(v[k], bperm[base + lane + k * T]);
            dev_dft8(v, w, +1, CPP_OPT);
            for(int k = 0; k < 8; k++) A[SWZ(b0 + k)] = w[k];
         }

         M = 8;
         for(int s = 1; s < NR8; s++) { dev_pass_r(A, 8, M, +1, CPP_OPT, T, true); M *= 8; }
         if(LEAD > 1) dev_pass_r(A, LEAD, M, +1, CPP_OPT, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
      }
}

// ---- large-N four-step transcription --------------------------------------

static int ilog8(int n);
static double rel_l2_cf(const std::vector<cf>& x, const std::vector<cf>& y);
static void ref_dft(const std::vector<cd>& in, std::vector<cd>& out, int sign);

// flat pass on the whole array, no shared-memory swizzle
static void dev_dft16(const cf u[16], cf v[16], int SIGN, int bk);
static void dev_dftR(int R, const cf* u, cf* v, int SIGN, int bk);

// A middle of size MID under base radix R runs LEAD first (if > 1) then k
// passes of R, where MID = LEAD * R^k -- mirroring LeadPass/LeadPass16 in
// fufft.cuh.  Assuming MID is a pure power of R divides to zero for e.g.
// MID=2048, R=16 (= 8 * 16^2); that bug cost an FPE before this existed.
static std::vector<int> mid_radices(int MID, int R)
{
   int k = 0, m = MID;
   while(m % R == 0) { m /= R; k++; }
   std::vector<int> v;
   if(m > 1) v.push_back(m);
   for(int i = 0; i < k; i++) v.push_back(R);
   return v;
}
static void kernel_pipeline_v16(int N, int LEAD, int NRV, int CPB, int bk,
                                const std::vector<cf>& a, const std::vector<cf>& bspec,
                                std::vector<cf>& out, int batch);
static int mixedrev16(int idx, int LEAD, int NR16);

// radix-R flat pass over the whole array (R in {8,16}) -- the outer levels of
// the four-step.  R=16 halves the level count when the outer factor is a power
// of 16, which is the whole point of the radix-16 outer level.
static void dev_pass_flat_r(cf* A, int R, int M, int SIGN, int bk,
                            size_t ngroups, bool dit)
{
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
   for(size_t g = 0; g < ngroups; g++) {
      const size_t j = g % (size_t)M, b = g / (size_t)M;
      const size_t base = b * (size_t)R * (size_t)M + j;
      cf u[16], v[16];
      for(int k = 0; k < R; k++) u[k] = A[base + (size_t)k * M];
      if(dit) {
         if(M > 1) dev_apply_tw_r(R, u, step, (int)j, bk);
         if(R == 16) dev_dft16(u, v, SIGN, bk); else dev_dftR(R, u, v, SIGN, bk);
      } else {
         if(R == 16) dev_dft16(u, v, SIGN, bk); else dev_dftR(R, u, v, SIGN, bk);
         if(M > 1) dev_apply_tw_r(R, v, step, (int)j, bk);
      }
      for(int k = 0; k < R; k++) A[base + (size_t)k * M] = v[k];
   }
}

static void dev_pass_flat(cf* A, int M, int SIGN, int bk, size_t ngroups, bool dit)
{
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(8 * M);
   for(size_t g = 0; g < ngroups; g++) {
      const size_t j = g % (size_t)M, b = g / (size_t)M;
      const size_t base = b * 8u * (size_t)M + j;
      cf u[8], v[8];
      for(int k = 0; k < 8; k++) u[k] = A[base + (size_t)k * M];
      if(dit) {
         if(M > 1) dev_apply_tw_r(8, u, step, (int)j, bk);
         dev_dft8(u, v, SIGN, bk);
      } else {
         dev_dft8(u, v, SIGN, bk);
         if(M > 1) dev_apply_tw_r(8, v, step, (int)j, bk);
      }
      for(int k = 0; k < 8; k++) A[base + (size_t)k * M] = v[k];
   }
}

// ground truth for bit-identity: the monolithic in-place conv on the flat array
static void conv_monolithic(std::vector<cf>& A, const std::vector<cf>& b_dr,
                            int N, int batch, int bk)
{
   const size_t ng = (size_t)batch * (N / 8);
   for(int M = N / 8; M >= 1; M /= 8) dev_pass_flat(A.data(), M, -1, bk, ng, false);
   for(size_t i = 0; i < A.size(); i++) A[i] = dev_cmul(A[i], b_dr[i]);
   for(int M = 1; M <= N / 8; M *= 8) dev_pass_flat(A.data(), M, +1, bk, ng, true);
}

// General four-step: outer radix R_OUT down to a middle of size MID, run by
// kernel_pipeline_ip (radix-8) or kernel_pipeline_v16 (radix-16).
// The filter must arrive in whatever digit order the chosen middle expects;
// bfilter_order() below derives it, and test_large_cfg asserts the whole chain
// is bit-identical to a monolithic transform built from the same radices.
static void conv_foursteps_cfg(std::vector<cf>& A, const std::vector<cf>& b_dr,
                               int N, int batch, int bk,
                               int R_OUT, int MID, bool v16_middle)
{
   const size_t ng = (size_t)batch * (N / R_OUT);
   for(int M = N / R_OUT; M >= MID; M /= R_OUT)
      dev_pass_flat_r(A.data(), R_OUT, M, -1, bk, ng, false);

   std::vector<cf> out;
   const int chunks = batch * (N / MID);
   if(v16_middle) {
      int L16 = MID, NRV = 0; while(L16 % 16 == 0) { L16 /= 16; NRV++; }
      kernel_pipeline_v16(MID, L16, NRV, 1, bk, A, b_dr, out, chunks);
   } else {
      int L8 = MID, NR8 = 0; while(L8 % 8 == 0) { L8 /= 8; NR8++; }
      kernel_pipeline_ip(MID, L8, NR8, 1, bk, A, b_dr, out, chunks);
   }
   A = out;

   for(int M = MID; M <= N / R_OUT; M *= R_OUT)
      dev_pass_flat_r(A.data(), R_OUT, M, +1, bk, ng, true);
}

// monolithic equivalent: the same radix sequence applied in place over all N
static void conv_monolithic_cfg(std::vector<cf>& A, const std::vector<cf>& b_dr,
                                int N, int batch, int bk,
                                int R_OUT, int MID, bool v16_middle)
{
   const int R_MID = v16_middle ? 16 : 8;
   std::vector<int> radices;
   for(int M = N / R_OUT; M >= MID; M /= R_OUT) radices.push_back(R_OUT);
   { const std::vector<int> mr = mid_radices(MID, R_MID);
     radices.insert(radices.end(), mr.begin(), mr.end()); }
   // forward: DIF, decreasing M
   int M = N;
   for(size_t k = 0; k < radices.size(); k++) {
      M /= radices[k];
      dev_pass_flat_r(A.data(), radices[k], M, -1, bk,
                      (size_t)batch * (N / radices[k]), false);
   }
   for(size_t i = 0; i < A.size(); i++) A[i] = dev_cmul(A[i], b_dr[i]);
   for(size_t k = radices.size(); k-- > 0; ) {
      dev_pass_flat_r(A.data(), radices[k], M, +1, bk,
                      (size_t)batch * (N / radices[k]), true);
      M *= radices[k];
   }
}

// the four-step chain: outer flat passes around the existing 4096 middle
static void conv_foursteps(std::vector<cf>& A, const std::vector<cf>& b_dr,
                           int N, int batch, int bk)
{
   const size_t ng = (size_t)batch * (N / 8);
   for(int M = N / 8; M >= 4096; M /= 8) dev_pass_flat(A.data(), M, -1, bk, ng, false);
   std::vector<cf> out;
   kernel_pipeline_ip(4096, 1, 4, 1, bk, A, b_dr, out, batch * (N / 4096));
   A = out;
   for(int M = 4096; M <= N / 8; M *= 8) dev_pass_flat(A.data(), M, +1, bk, ng, true);
}

// independent fp64 reference: iterative radix-2 Cooley-Tukey
static void fft2_f64(std::vector<cd>& x, int sign)
{
   const int n = (int)x.size();
   for(int i = 1, j = 0; i < n; i++) {
      int bit = n >> 1;
      for(; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if(i < j) std::swap(x[i], x[j]);
   }
   for(int len = 2; len <= n; len <<= 1) {
      const double ang = sign * 2.0 * PI / (double)len;
      const cd wl(cos(ang), sin(ang));
      for(int i = 0; i < n; i += len) {
         cd w(1.0, 0.0);
         for(int k = 0; k < len / 2; k++) {
            cd a0 = x[i + k], a1 = x[i + k + len / 2] * w;
            x[i + k] = a0 + a1;
            x[i + k + len / 2] = a0 - a1;
            w *= wl;
         }
      }
   }
}

// Digit order the chained transform leaves its spectrum in: the radices in
// pass order, reversed.  Derived, not guessed -- test_large_cfg asserts it.
static int chain_rev(int idx, const std::vector<int>& radices)
{
   // BIT-GROUP reversal, matching mixedrev/mixedrev16 exactly: groups are the
   // radix widths in PASS order, extracted MSB-first, emitted in reverse pass
   // order with each group's bits intact.  (A plain LSB-first digit reversal
   // agrees only when every radix has the same width; it diverges the moment
   // a lead radix appears, which is what produced the 1.41 mismatches.)
   int w[32], p = 0, m = 0;
   for(size_t k = 0; k < radices.size(); k++) {
      int lw = 0; while((1 << lw) < radices[k]) lw++;
      w[p++] = lw; m += lw;
   }
   int d[32], pos = m;
   for(int i = 0; i < p; i++) { pos -= w[i]; d[i] = (idx >> pos) & ((1 << w[i]) - 1); }
   int out = 0, op = m;
   for(int i = p - 1; i >= 0; i--) { op -= w[i]; out |= d[i] << op; }
   return out;
}

static bool test_large_cfg(int N, int batch, int R_OUT, int MID, bool v16_mid)
{
   const int R_MID = v16_mid ? 16 : 8;
   std::vector<int> radices;
   for(int M = N / R_OUT; M >= MID; M /= R_OUT) radices.push_back(R_OUT);
   { const std::vector<int> mr = mid_radices(MID, R_MID);
     radices.insert(radices.end(), mr.begin(), mr.end()); }

   std::mt19937 rng(N ^ (R_OUT * 7919) ^ (MID * 104729));
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);
   const size_t n = (size_t)N * batch;
   const float invN = 1.0f / (float)N;

   std::vector<cf> a(n), b(n), b_dr(n);
   for(size_t i = 0; i < n; i++) {
      a[i] = cf(d(rng), d(rng));
      b[i] = cf(d(rng) * invN, d(rng) * invN);
   }
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++)
         b_dr[(size_t)t * N + i] = b[(size_t)t * N + chain_rev(i, radices)];

   std::vector<cf> F = a, M0 = a;
   conv_foursteps_cfg(F, b_dr, N, batch, CPP_OPT, R_OUT, MID, v16_mid);
   conv_monolithic_cfg(M0, b_dr, N, batch, CPP_OPT, R_OUT, MID, v16_mid);
   const double dBit = rel_l2_cf(F, M0);

   std::vector<cd> ad(N);
   for(int i = 0; i < N; i++) ad[i] = cd(a[i].real(), a[i].imag());
   fft2_f64(ad, -1);
   for(int i = 0; i < N; i++) ad[i] *= cd(b[i].real(), b[i].imag());
   fft2_f64(ad, +1);
   double num = 0, den = 0;
   for(int i = 0; i < N; i++) {
      cd v(F[i].real(), F[i].imag());
      num += std::norm(v - ad[i]); den += std::norm(ad[i]);
   }
   const double dRef = sqrt(num / den);
   const int lout = (int)(radices.size() - mid_radices(MID, R_MID).size());
   const int passes = 4 * lout + 3;
   // the radix-8 middle shares the flat model's arithmetic order exactly; the
   // V=16 middle reorders it (swizzled shared layout), so require bit-identity
   // only where it is meaningful and tight agreement otherwise.  fp64 is the
   // ground truth in both cases.
   const double bitTol = v16_mid ? 2e-6 : 0.0;
   const bool ok = (dBit <= bitTol) && (dRef < 5e-6);
   printf("   N=%-8d outer r%-2d mid %-5d %-8s | 4step-vs-mono %.0e | vs fp64 %.2e | %s\n",
          N, R_OUT, MID, v16_mid ? "V=16" : "radix8", dBit, dRef,
          ok ? "PASS" : "FAIL");
   printf("            LOUT=%d -> %d passes\n", lout, passes);
   return ok;
}

static bool test_large(int N, int batch)
{
   std::mt19937 rng(N ^ 0x5eed);
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);
   const size_t n = (size_t)N * batch;
   const float invN = 1.0f / (float)N;
   const int S = ilog8(N);

   std::vector<cf> a(n), b(n), b_dr(n);
   for(size_t i = 0; i < n; i++) {
      a[i] = cf(d(rng), d(rng));
      b[i] = cf(d(rng) * invN, d(rng) * invN);
   }
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++) {
         int r = 0, q = i;
         for(int s2 = 0; s2 < S; s2++) { r = r * 8 + (q & 7); q >>= 3; }
         b_dr[(size_t)t * N + i] = b[(size_t)t * N + r];
      }

   std::vector<cf> M1 = a, F1 = a;
   conv_monolithic(M1, b_dr, N, batch, CPP_OPT);
   conv_foursteps(F1, b_dr, N, batch, CPP_OPT);
   const double dBit = rel_l2_cf(F1, M1);          // must be exactly 0

   // fp64 reference on the first batch item only (radix-2, independent path)
   std::vector<cd> ad(N);
   for(int i = 0; i < N; i++) ad[i] = cd(a[i].real(), a[i].imag());
   fft2_f64(ad, -1);
   for(int i = 0; i < N; i++)
      ad[i] *= cd(b[i].real(), b[i].imag());
   fft2_f64(ad, +1);
   double num = 0, den = 0;
   for(int i = 0; i < N; i++) {
      cd v(F1[i].real(), F1[i].imag());
      num += std::norm(v - ad[i]);
      den += std::norm(ad[i]);
   }
   const double dRef = sqrt(num / den);

   const bool ok = (dBit == 0.0) && (dRef < 5e-6);
   printf("   N=%-8d LOUT=%d bt=%d | foursteps-vs-monolithic %.0e | vs fp64 fft %.2e | %s\n",
          N, S - 4, batch, dBit, dRef, ok ? "PASS" : "FAIL");
   return ok;
}

// ---- two-for-one real conv transcription ----------------------------------

static void kernel_conv_r2(int N, int LEAD, int NR8, int CPB, int bk,
                           const std::vector<float>& xr, const std::vector<float>& yr,
                           const std::vector<cf>& b_dr,
                           std::vector<float>& ur, std::vector<float>& vr, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   ur.assign(xr.size(), 0.0f);
   vr.assign(xr.size(), 0.0f);
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ(i)] = cf(xr[base + i], yr[base + i]);

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_r(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NR8; s++) { M /= 8; dev_pass_r(A, 8, M, -1, bk, T, false); }
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ(i)] = dev_cmul(A[SWZ(i)], b_dr[base + i]);
         M = 1;
         for(int s = 0; s < NR8; s++) { dev_pass_r(A, 8, M, +1, bk, T, true); M *= 8; }
         if(LEAD > 1) dev_pass_r(A, LEAD, M, +1, bk, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) {
               ur[base + i] = A[SWZ(i)].real();
               vr[base + i] = A[SWZ(i)].imag();
            }
      }
}

static bool test_r2(int N, int CPB, int batch)
{
   int LEAD, NR8; factorise(N, LEAD, NR8);
   std::mt19937 rng(N * 77 + 5);
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);
   const size_t n = (size_t)N * batch;
   const int S = ilog8(N);
   (void)S;

   std::vector<float> xr(n), yr(n);
   std::vector<cd> h(N);                     // one REAL filter shared by the batch
   for(size_t i = 0; i < n; i++) { xr[i] = d(rng); yr[i] = d(rng); }
   for(int i = 0; i < N; i++) h[i] = cd((double)d(rng), 0.0);

   // filter spectrum in fp64, scaled by 1/N, scrambled to the kernel's order
   std::vector<cd> H;
   ref_dft(h, H, -1);
   std::vector<cf> b_dr(n);
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++) {
         const cd v = H[mixedrev(i, LEAD, NR8)] / (double)N;
         b_dr[(size_t)t * N + i] = cf((float)v.real(), (float)v.imag());
      }

   std::vector<float> ur, vr;
   kernel_conv_r2(N, LEAD, NR8, CPB, CPP_OPT, xr, yr, b_dr, ur, vr, batch);

   // bit-identity: the complex kernel on pre-packed input must match exactly
   std::vector<cf> z(n), w;
   for(size_t i = 0; i < n; i++) z[i] = cf(xr[i], yr[i]);
   kernel_pipeline_ip(N, LEAD, NR8, CPB, CPP_OPT, z, b_dr, w, batch);
   double dBit = 0.0;
   for(size_t i = 0; i < n; i++)
      dBit = std::max(dBit, (double)std::abs(w[i].real() - ur[i])
                            + std::abs(w[i].imag() - vr[i]));

   // fp64 reference: two separate real convolutions
   double worst = 0.0;
   for(int t = 0; t < batch && t < 2; t++) {
      const size_t off = (size_t)t * N;
      for(int which = 0; which < 2; which++) {
         std::vector<cd> sig(N), SG, prod(N), res;
         for(int i = 0; i < N; i++)
            sig[i] = cd((double)(which ? yr[off + i] : xr[off + i]), 0.0);
         ref_dft(sig, SG, -1);
         for(int i = 0; i < N; i++) prod[i] = SG[i] * H[i] / (double)N;
         ref_dft(prod, res, +1);
         double num = 0, den = 0;
         for(int i = 0; i < N; i++) {
            const double got = which ? (double)vr[off + i] : (double)ur[off + i];
            num += (got - res[i].real()) * (got - res[i].real());
            den += res[i].real() * res[i].real();
         }
         worst = std::max(worst, sqrt(num / den));
      }
   }
   const bool ok = (dBit == 0.0) && (worst < 3e-6);
   printf("   r2 N=%-5d CPB=%d bt=%-3d | packed-vs-complex %.0e | u,v vs fp64 %.2e | %s\n",
          N, CPB, batch, dBit, worst, ok ? "PASS" : "FAIL");
   return ok;
}

// ---- 2D fused conv transcription ------------------------------------------

static void dev_col_pass(cf* A, int W, int m, int SIGN, int bk, int T, bool dit)
{
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(8 * m);
   for(int lane = 0; lane < T; lane++) {
      const int c = lane % W, g2 = lane / W, j2 = g2 % m, b2 = g2 / m;
      const int base = c + W * (b2 * 8 * m + j2);
      cf u[8], v[8];
      for(int k = 0; k < 8; k++) u[k] = A[SWZ(base + k * W * m)];
      if(dit) {
         if(m > 1) dev_apply_tw_r(8, u, step, j2, bk);
         dev_dft8(u, v, SIGN, bk);
      } else {
         dev_dft8(u, v, SIGN, bk);
         if(m > 1) dev_apply_tw_r(8, v, step, j2, bk);
      }
      for(int k = 0; k < 8; k++) A[SWZ(base + k * W * m)] = v[k];
   }
}

static void kernel_conv2d(int H, int W, int bk,
                          const std::vector<cf>& a, const std::vector<cf>& b_dr,
                          std::vector<cf>& out, int batch)
{
   const int NP = H * W, T = NP / 8;
   std::vector<cf> smem(NP);
   out.assign(a.size(), cf(0, 0));
   for(int t = 0; t < batch; t++) {
      cf* A = smem.data();
      const size_t base = (size_t)t * NP;
      for(int lane = 0; lane < T; lane++)
         for(int i = lane; i < NP; i += T) A[SWZ(i)] = a[base + i];
      for(int M = W / 8; M >= 1; M /= 8) dev_pass_r(A, 8, M, -1, bk, T, false);
      for(int m = H / 8; m >= 1; m /= 8) dev_col_pass(A, W, m, -1, bk, T, false);
      for(int lane = 0; lane < T; lane++)
         for(int i = lane; i < NP; i += T)
            A[SWZ(i)] = dev_cmul(A[SWZ(i)], b_dr[base + i]);
      for(int m = 1; m <= H / 8; m *= 8) dev_col_pass(A, W, m, +1, bk, T, true);
      for(int M = 1; M <= W / 8; M *= 8) dev_pass_r(A, 8, M, +1, bk, T, true);
      for(int lane = 0; lane < T; lane++)
         for(int i = lane; i < NP; i += T) out[base + i] = A[SWZ(i)];
   }
}

static bool test_2d(int H, int W, int batch)
{
   const int NP = H * W;
   std::mt19937 rng(H * 131 + W);
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);
   const size_t n = (size_t)NP * batch;
   const float invN = 1.0f / (float)NP;
   const int SH = ilog8(H), SW = ilog8(W);

   std::vector<cf> a(n), b(n), b_dr(n);
   for(size_t i = 0; i < n; i++) {
      a[i] = cf(d(rng), d(rng));
      b[i] = cf(d(rng) * invN, d(rng) * invN);
   }
   // slot (r, c) holds spectrum (rev8_H(r), rev8_W(c))
   auto rev8 = [](int x, int S) {
      int r = 0;
      for(int s2 = 0; s2 < S; s2++) { r = r * 8 + (x & 7); x >>= 3; }
      return r;
   };
   for(int t = 0; t < batch; t++)
      for(int r = 0; r < H; r++)
         for(int c = 0; c < W; c++)
            b_dr[(size_t)t * NP + r * W + c] =
               b[(size_t)t * NP + rev8(r, SH) * W + rev8(c, SW)];

   std::vector<cf> OUT;
   kernel_conv2d(H, W, CPP_OPT, a, b_dr, OUT, batch);

   // independent fp64 reference: radix-2 row FFTs + column FFTs
   double worst = 0.0;
   for(int t = 0; t < batch; t++) {
      const size_t off = (size_t)t * NP;
      std::vector<cd> X(NP);
      for(int i = 0; i < NP; i++) X[i] = cd(a[off + i].real(), a[off + i].imag());
      std::vector<cd> row(W), col(H);
      auto rows = [&](int sign) {
         for(int r = 0; r < H; r++) {
            for(int c = 0; c < W; c++) row[c] = X[r * W + c];
            fft2_f64(row, sign);
            for(int c = 0; c < W; c++) X[r * W + c] = row[c];
         }
      };
      auto cols = [&](int sign) {
         for(int c = 0; c < W; c++) {
            for(int r = 0; r < H; r++) col[r] = X[r * W + c];
            fft2_f64(col, sign);
            for(int r = 0; r < H; r++) X[r * W + c] = col[r];
         }
      };
      rows(-1); cols(-1);
      for(int i = 0; i < NP; i++)
         X[i] *= cd(b[off + i].real(), b[off + i].imag());
      cols(+1); rows(+1);
      double num = 0, den = 0;
      for(int i = 0; i < NP; i++) {
         cd v(OUT[off + i].real(), OUT[off + i].imag());
         num += std::norm(v - X[i]);
         den += std::norm(X[i]);
      }
      worst = std::max(worst, sqrt(num / den));
   }
   const bool ok = worst < 3e-6;
   printf("   2D %3dx%-4d bt=%d | fused conv vs fp64 row/col fft %.2e | %s\n",
          H, W, batch, worst, ok ? "PASS" : "FAIL");
   return ok;
}

// ---- split fwd/apply transcription ----------------------------------------

static void kernel_pipeline_fwd(int N, int LEAD, int NR8, int CPB, int bk,
                                const std::vector<cf>& a, std::vector<cf>& spec, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   spec.assign(a.size(), cf(0, 0));
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
         int M = N / LEAD;
         if(LEAD > 1) dev_pass_r(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NR8; s++) { M /= 8; dev_pass_r(A, 8, M, -1, bk, T, false); }
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) spec[base + i] = A[SWZ(i)];
      }
}

static void kernel_pipeline_apply(int N, int LEAD, int NR8, int CPB, int bk,
                                  const std::vector<cf>& spec, const std::vector<cf>& b_dr,
                                  std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(spec.size(), cf(0, 0));
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ(i)] = dev_cmul(spec[base + i], b_dr[base + i]);
         int M = 1;
         for(int s = 0; s < NR8; s++) { dev_pass_r(A, 8, M, +1, bk, T, true); M *= 8; }
         if(LEAD > 1) dev_pass_r(A, LEAD, M, +1, bk, T, true);
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
      }
}

// ---- radix-16 lane-pair transcription ------------------------------------

static const float C16 = 0.92387953251128673848f, S16 = 0.38268343236508977173f;

static cf dev_mul_w16(int K, cf x, int SIGN, int bk)
{
   const float s = (float)SIGN;
   switch(K) {
      case 0:  return x;
      case 1:  return dev_cmul(x, cf( C16, s * S16));
      case 2:  return dev_cmul_w1(x, SIGN, bk);
      case 3:  return dev_cmul(x, cf( S16, s * C16));
      case 4:  return dev_cmul_i (x, SIGN, bk);
      case 5:  return dev_cmul(x, cf(-S16, s * C16));
      case 6:  return dev_cmul_w3(x, SIGN, bk);
      default: return dev_cmul(x, cf(-C16, s * S16));
   }
}

static void dev_pass_r16(cf* A, int M, int SIGN, int bk, int T, bool dit)
{
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(16 * M);
   std::vector<cf> u(T * 8), o(T * 8);
   for(int L = 0; L < T; L++) {
      const int h = L & 1, t = L >> 1, j = t % M, b = t / M, base = b * 16 * M + j;
      (void)j;
      for(int k = 0; k < 8; k++) u[L * 8 + k] = A[SWZ(base + (h * 8 + k) * M)];
   }
   for(int L = 0; L < T; L += 2) {
      const int t = L >> 1, j = t % M;
      cf w1(1.0f, 0.0f), w8(1.0f, 0.0f);
      if(M > 1) {
         w1 = dev_twiddle(step * (float)j);
         cf w2 = dev_cmul(w1, w1), w4 = dev_cmul(w2, w2);
         w8 = dev_cmul(w4, w4);
      }
      cf a[2][8], v[2][8], out[2][8];
      for(int h = 0; h < 2; h++) for(int k = 0; k < 8; k++) a[h][k] = u[(L + h) * 8 + k];

      if(dit) {                                   // twiddle first
         for(int h = 0; h < 2; h++) {
            cf w = (h == 0) ? cf(1.0f, 0.0f) : w8;
            a[h][0] = dev_cmul(a[h][0], w);
            for(int m = 1; m < 8; m++) { w = dev_cmul(w, w1); a[h][m] = dev_cmul(a[h][m], w); }
         }
      }
      cf s0[8], s1[8];
      for(int k = 0; k < 8; k++) {                // radix-2 across the pair
         s0[k] = a[0][k] + a[1][k];
         s1[k] = dev_mul_w16(k, a[0][k] - a[1][k], SIGN, bk);
      }
      dev_dft8(s0, v[0], SIGN, bk);
      dev_dft8(s1, v[1], SIGN, bk);
      for(int q = 0; q < 4; q++) {                // redistribute
         out[0][2 * q]     = v[0][q];
         out[0][2 * q + 1] = v[1][q];
         out[1][2 * q]     = v[0][4 + q];
         out[1][2 * q + 1] = v[1][4 + q];
      }
      if(!dit && M > 1) {                         // twiddle after
         for(int h = 0; h < 2; h++) {
            cf w = (h == 0) ? cf(1.0f, 0.0f) : w8;
            out[h][0] = dev_cmul(out[h][0], w);
            for(int m = 1; m < 8; m++) { w = dev_cmul(w, w1); out[h][m] = dev_cmul(out[h][m], w); }
         }
      }
      for(int h = 0; h < 2; h++) for(int m = 0; m < 8; m++) o[(L + h) * 8 + m] = out[h][m];
   }
   for(int L = 0; L < T; L++) {
      const int h = L & 1, t = L >> 1, j = t % M, b = t / M, base = b * 16 * M + j;
      for(int m = 0; m < 8; m++) A[SWZ(base + (h * 8 + m) * M)] = o[L * 8 + m];
   }
}

static void kernel_pipeline_r16(int N, int LEAD, int NR16, int CPB, int bk,
                                const std::vector<cf>& a, const std::vector<cf>& bspec,
                                std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(a.size(), cf(0, 0));
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_r(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NR16; s++) { M /= 16; dev_pass_r16(A, M, -1, bk, T, false); }

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ(i)] = dev_cmul(A[SWZ(i)], bspec[base + i]);

         M = 1;
         for(int s = 0; s < NR16; s++) { dev_pass_r16(A, M, +1, bk, T, true); M *= 16; }
         if(LEAD > 1) dev_pass_r(A, LEAD, M, +1, bk, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ(i)];
      }
}

// mixed-radix scramble for the radix-16 route: bit groups [log2 LEAD, 4, 4, ...]
static int mixedrev16(int idx, int LEAD, int NR16)
{
   int w[16], p = 0, m = 0;
   if(LEAD > 1) { int lw = 0; while((1 << lw) < LEAD) lw++; w[p++] = lw; m += lw; }
   for(int i = 0; i < NR16; i++) { w[p++] = 4; m += 4; }
   int d[16], pos = m;
   for(int i = 0; i < p; i++) { pos -= w[i]; d[i] = (idx >> pos) & ((1 << w[i]) - 1); }
   int out = 0, op = m;
   for(int i = p - 1; i >= 0; i--) { op -= w[i]; out |= d[i] << op; }
   return out;
}

// ---- V=16 transcription (one lane holds sixteen values) -------------------

#define SWZ16(i) ((i) ^ (((i) >> 3) & 15) ^ (((i) >> 7) & 15))

static void dev_dft16(const cf u[16], cf v[16], int SIGN, int bk)
{
   cf a[8], b[8], va[8], vb[8];
   for(int k = 0; k < 8; k++) {
      a[k] = u[k] + u[k + 8];
      b[k] = dev_mul_w16(k, u[k] - u[k + 8], SIGN, bk);
   }
   dev_dft8(a, va, SIGN, bk);
   dev_dft8(b, vb, SIGN, bk);
   for(int k = 0; k < 8; k++) { v[2 * k] = va[k]; v[2 * k + 1] = vb[k]; }
}

static void dev_apply_tw_16(cf x[16], float step, int j, int bk)
{
   (void)bk;
   const cf w1 = dev_twiddle(step * (float)j);
   const cf w2 = dev_cmul(w1, w1);
   x[1] = dev_cmul(x[1], w1);
   cf ze = w2, zo = dev_cmul(w2, w1);
   for(int m = 1; m < 8; m++) {
      x[2 * m]     = dev_cmul(x[2 * m],     ze);
      x[2 * m + 1] = dev_cmul(x[2 * m + 1], zo);
      if(m < 7) { ze = dev_cmul(ze, w2); zo = dev_cmul(zo, w2); }
   }
}

static void dev_pass_v16(cf* A, int R, int M, int SIGN, int bk, int T, bool dit)
{
   const int G = 16 / R;
   const float step = (float)SIGN * 6.28318530717958647692f / (float)(R * M);
   for(int lane = 0; lane < T; lane++)
      for(int g = 0; g < G; g++) {
         const int gidx = lane + g * T;
         const int j = gidx % M, b = gidx / M, base = b * R * M + j;
         cf u[16], v[16];
         for(int k = 0; k < R; k++) u[k] = A[SWZ16(base + k * M)];
         if(dit && M > 1) {
            if(R == 16) dev_apply_tw_16(u, step, j, bk);
            else        dev_apply_tw_r(R, u, step, j, bk);
         }
         if(R == 16)     dev_dft16(u, v, SIGN, bk);
         else            dev_dftR(R, u, v, SIGN, bk);
         if(!dit && M > 1) {
            if(R == 16) dev_apply_tw_16(v, step, j, bk);
            else        dev_apply_tw_r(R, v, step, j, bk);
         }
         for(int k = 0; k < R; k++) A[SWZ16(base + k * M)] = v[k];
      }
}

static void kernel_pipeline_v16f(int N, int LEAD, int NRV, int CPB, int bk,
                                 const std::vector<cf>& a, const std::vector<cf>& bperm,
                                 std::vector<cf>& out, int batch)
{
   const int T = N / 16, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(a.size(), cf(0, 0));
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ16(i)] = a[base + i];

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_v16(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NRV - 1; s++) { M /= 16; dev_pass_v16(A, 16, M, -1, bk, T, false); }

         for(int lane = 0; lane < T; lane++) {          // register boundary
            const int b0 = lane * 16;
            cf x[16], y[16];
            for(int k = 0; k < 16; k++) x[k] = A[SWZ16(b0 + k)];
            dev_dft16(x, y, -1, bk);
            for(int k = 0; k < 16; k++) y[k] = dev_cmul(y[k], bperm[base + lane + k * T]);
            dev_dft16(y, x, +1, bk);
            for(int k = 0; k < 16; k++) A[SWZ16(b0 + k)] = x[k];
         }

         M = 16;
         for(int s = 1; s < NRV; s++) { dev_pass_v16(A, 16, M, +1, bk, T, true); M *= 16; }
         if(LEAD > 1) dev_pass_v16(A, LEAD, M, +1, bk, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ16(i)];
      }
}

static void kernel_pipeline_v16(int N, int LEAD, int NRV, int CPB, int bk,
                                const std::vector<cf>& a, const std::vector<cf>& bspec,
                                std::vector<cf>& out, int batch)
{
   const int T = N / 16, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * N);
   out.assign(a.size(), cf(0, 0));
   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * N];
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ16(i)] = a[base + i];

         int M = N / LEAD;
         if(LEAD > 1) dev_pass_v16(A, LEAD, M, -1, bk, T, false);
         for(int s = 0; s < NRV; s++) { M /= 16; dev_pass_v16(A, 16, M, -1, bk, T, false); }

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               A[SWZ16(i)] = dev_cmul(A[SWZ16(i)], bspec[base + i]);

         M = 1;
         for(int s = 0; s < NRV; s++) { dev_pass_v16(A, 16, M, +1, bk, T, true); M *= 16; }
         if(LEAD > 1) dev_pass_v16(A, LEAD, M, +1, bk, T, true);

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = A[SWZ16(i)];
      }
}


// ---- int16 BFP transcription ----------------------------------------------

struct BfpQuant {
   std::vector<short> qre, qim;
   float scale;
};

static BfpQuant bfp_quantize(const cf* x, int N)
{
   float mx = 0.0f;
   for(int i = 0; i < N; i++)
      mx = std::max(mx, std::max(std::fabs(x[i].real()), std::fabs(x[i].imag())));
   mx = std::max(mx, 1e-30f);
   const float inv = 32767.0f / mx;
   BfpQuant q; q.qre.resize(N); q.qim.resize(N); q.scale = mx / 32767.0f;
   for(int i = 0; i < N; i++) {
      q.qre[i] = (short)lrintf(std::min(std::max(x[i].real() * inv, -32767.0f), 32767.0f));
      q.qim[i] = (short)lrintf(std::min(std::max(x[i].imag() * inv, -32767.0f), 32767.0f));
   }
   return q;
}

// mirrors fufft_pipeline_v16_bfp: dequant -> fp32 passes -> dequant filter mult
// -> inverse -> per-transform abs-max -> int16 quantise
static void kernel_v16_bfp(int N, int LEAD, int NRV, int bk,
                           const BfpQuant& aq, const BfpQuant& bq,
                           BfpQuant& outq)
{
   const int T = N / 16;
   std::vector<cf> A(N);
   for(int lane = 0; lane < T; lane++)
      for(int i = lane; i < N; i += T)
         A[SWZ16(i)] = cf(aq.qre[i] * aq.scale, aq.qim[i] * aq.scale);

   int M = N / LEAD;
   if(LEAD > 1) dev_pass_v16(A.data(), LEAD, M, -1, bk, T, false);
   for(int s = 0; s < NRV; s++) { M /= 16; dev_pass_v16(A.data(), 16, M, -1, bk, T, false); }

   for(int lane = 0; lane < T; lane++)
      for(int i = lane; i < N; i += T)
         A[SWZ16(i)] = dev_cmul(A[SWZ16(i)], cf(bq.qre[i] * bq.scale, bq.qim[i] * bq.scale));

   M = 1;
   for(int s = 0; s < NRV; s++) { dev_pass_v16(A.data(), 16, M, +1, bk, T, true); M *= 16; }
   if(LEAD > 1) dev_pass_v16(A.data(), LEAD, M, +1, bk, T, true);

   std::vector<cf> lin(N);
   for(int lane = 0; lane < T; lane++)
      for(int i = lane; i < N; i += T) lin[i] = A[SWZ16(i)];
   outq = bfp_quantize(lin.data(), N);
}

static bool test_bfp(int N, int seed)
{
   int LEAD16 = 1, NRV = 0;
   { int n = N; while(n % 16 == 0) { n /= 16; NRV++; } LEAD16 = n; }

   std::mt19937 rng(seed * 991 + N);
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);
   std::vector<cf> a(N), b(N);
   const float invN = 1.0f / (float)N;
   for(int i = 0; i < N; i++) {
      a[i] = cf(d(rng), d(rng));
      b[i] = cf(d(rng) * invN, d(rng) * invN);
   }
   // filter delivered in the v16 scrambled order, then quantised
   std::vector<cf> b_dr(N);
   for(int i = 0; i < N; i++) b_dr[i] = b[mixedrev16(i, LEAD16, NRV)];

   BfpQuant aq = bfp_quantize(a.data(), N);
   BfpQuant bq = bfp_quantize(b_dr.data(), N);
   BfpQuant oq;
   kernel_v16_bfp(N, LEAD16, NRV, CPP_OPT, aq, bq, oq);

   // fp64 reference on the UNQUANTISED inputs: the measured error is the whole
   // pipeline's, input quantisation included -- that is what a user sees.
   std::vector<cd> ad(N), Bd, prod(N), res;
   for(int i = 0; i < N; i++) ad[i] = cd(a[i].real(), a[i].imag());
   ref_dft(ad, Bd, -1);
   for(int i = 0; i < N; i++)
      prod[i] = Bd[i] * cd(b[i].real(), b[i].imag());
   ref_dft(prod, res, +1);

   double num = 0, den = 0;
   for(int i = 0; i < N; i++) {
      const cd got(oq.qre[i] * (double)oq.scale, oq.qim[i] * (double)oq.scale);
      num += std::norm(got - res[i]);
      den += std::norm(res[i]);
   }
   const double rel = sqrt(num / den);
   const double bits = -std::log2(rel);
   const bool ok = rel < 3e-4 && bits >= 12.0;
   printf("   bfp N=%-5d seed=%d | vs fp64 (unquantised ref) %.2e = %.1f effective bits | %s\n",
          N, seed, rel, bits, ok ? "PASS" : "FAIL");
   return ok;
}
// ---- kernels -------------------------------------------------------------

static void kernel_c2c(int N, int LOG8N, int CPB, int bk,
                       const std::vector<cf>& in, std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * 2 * N);
   out.assign(in.size(), cf(0, 0));

   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * 2 * N];
         cf* B = A + N;
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = in[base + i];
         cf* p = dev_stages(N, LOG8N, A, B, -1, bk, T);
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = p[SWZ(i)];
      }
}

static void kernel_conv(int N, int LOG8N, int CPB, int bk,
                        const std::vector<cf>& a, const std::vector<cf>& bspec,
                        std::vector<cf>& out, int batch)
{
   const int T = N / 8, tpb = CPB * T, grid = batch / CPB;
   std::vector<cf> smem((size_t)CPB * 2 * N);
   out.assign(a.size(), cf(0, 0));

   for(int blk = 0; blk < grid; blk++)
      for(int tid = 0; tid < tpb; tid += T) {
         const int slot = tid / T;
         cf* A = &smem[(size_t)slot * 2 * N];
         cf* B = A + N;
         const size_t base = ((size_t)blk * CPB + slot) * N;

         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) A[SWZ(i)] = a[base + i];
         cf* p = dev_stages(N, LOG8N, A, B, -1, bk, T);
         cf* q = (p == A) ? B : A;
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T)
               p[SWZ(i)] = dev_cmul(p[SWZ(i)], bspec[base + i]);
         p = dev_stages(N, LOG8N, p, q, +1, bk, T);
         for(int lane = 0; lane < T; lane++)
            for(int i = lane; i < N; i += T) out[base + i] = p[SWZ(i)];
      }
}

// ---- reference -----------------------------------------------------------

static void ref_dft(const std::vector<cd>& in, std::vector<cd>& out, int sign)
{
   int n = (int)in.size();
   out.assign(n, cd(0, 0));
   for(int k = 0; k < n; k++) {
      cd acc(0, 0);
      for(int j = 0; j < n; j++) {
         double t = sign * 2.0 * PI * (double)k * (double)j / (double)n;
         acc += in[j] * cd(cos(t), sin(t));
      }
      out[k] = acc;
   }
}

static double rel_l2(const std::vector<cf>& x, const std::vector<cd>& y, size_t off, int n)
{
   double num = 0, den = 0;
   for(int i = 0; i < n; i++) {
      cd v((double)x[off + i].real(), (double)x[off + i].imag());
      num += std::norm(v - y[i]);
      den += std::norm(y[i]);
   }
   return sqrt(num / den);
}

static double rel_l2_cf(const std::vector<cf>& x, const std::vector<cf>& y)
{
   double num = 0, den = 0;
   for(size_t i = 0; i < x.size(); i++) {
      cd d((double)x[i].real() - y[i].real(), (double)x[i].imag() - y[i].imag());
      num += std::norm(d);
      den += (double)std::norm(y[i]);
   }
   return sqrt(num / den);
}

static int ilog8(int n) { int s = 0; while(n > 1) { n /= 8; s++; } return s; }

static bool test(int N, int CPB, int batch)
{
   int LEAD, NR8; factorise(N, LEAD, NR8);
   const int LOG8N = ilog8(N);   // Stockham path is still pure radix 8
   std::mt19937 rng(N * 131 + CPB);
   std::uniform_real_distribution<float> d(-1.0f, 1.0f);

   const size_t n = (size_t)N * batch;
   std::vector<cf> a(n), b(n);
   const float invN = 1.0f / (float)N;
   for(size_t i = 0; i < n; i++) {
      a[i] = cf(d(rng), d(rng));
      b[i] = cf(d(rng) * invN, d(rng) * invN);
   }

   const bool pow8 = (LEAD == 1);      // the Stockham path is radix 8 only
   std::vector<cf> A0, A1, B0, B1, BIP;
   if(pow8) {
      kernel_c2c (N, LOG8N, CPB, CPP_NAIVE, a, A0, batch);
      kernel_c2c (N, LOG8N, CPB, CPP_OPT,   a, A1, batch);
      kernel_conv(N, LOG8N, CPB, CPP_NAIVE, a, b, B0, batch);
      kernel_conv(N, LOG8N, CPB, CPP_OPT,   a, b, B1, batch);
   }

   // the in-place kernel consumes the filter spectrum in digit-reversed order
   std::vector<cf> b_dr(n);
   for(int t = 0; t < batch; t++)
      for(int i = 0; i < N; i++)
         b_dr[(size_t)t * N + i] = b[(size_t)t * N + mixedrev(i, LEAD, NR8)];
   kernel_pipeline_ip(N, LEAD, NR8, CPB, CPP_OPT, a, b_dr, BIP, batch);

   double worstA = 0, worstB = 0, worstIP = 0;
   for(int t = 0; t < batch; t++) {
      const size_t off = (size_t)t * N;
      std::vector<cd> ad(N), bd(N), Ad, Bd;
      for(int i = 0; i < N; i++) {
         ad[i] = cd(a[off + i].real(), a[off + i].imag());
         bd[i] = cd(b[off + i].real(), b[off + i].imag());
      }
      ref_dft(ad, Ad, -1);
      if(pow8) worstA = std::max(worstA, rel_l2(A1, Ad, off, N));

      std::vector<cd> prod(N);
      for(int i = 0; i < N; i++) prod[i] = Ad[i] * bd[i];
      ref_dft(prod, Bd, +1);
      if(pow8) worstB = std::max(worstB, rel_l2(B1, Bd, off, N));
      // the in-place path is the one that must work at every power of two
      worstIP = std::max(worstIP, rel_l2(BIP, Bd, off, N));
   }

   const double dA = pow8 ? rel_l2_cf(A1, A0) : 0.0;
   const double dB = pow8 ? rel_l2_cf(B1, B0) : 0.0;
   const double dIP = pow8 ? rel_l2_cf(BIP, B1) : 0.0;

   // split fwd/apply must reproduce the fused conv bit-for-bit: the global
   // round trip through the spectrum stores exact values
   double dSPL = -1.0;
   {
      std::vector<cf> SP, AP;
      kernel_pipeline_fwd(N, LEAD, NR8, CPB, CPP_OPT, a, SP, batch);
      kernel_pipeline_apply(N, LEAD, NR8, CPB, CPP_OPT, SP, b_dr, AP, batch);
      dSPL = rel_l2_cf(AP, BIP);
   }

   // register-resident boundary: needs the filter permuted so lane L's eight
   // values land at {L, L+T, ..., L+7T} instead of {8L .. 8L+7}
   const int T_ = N / 8;
   std::vector<cf> b_perm(n);
   for(int t = 0; t < batch; t++)
      for(int lane = 0; lane < T_; lane++)
         for(int k = 0; k < 8; k++)
            b_perm[(size_t)t * N + lane + k * T_] = b_dr[(size_t)t * N + 8 * lane + k];
   std::vector<cf> BF;
   kernel_pipeline_fused(N, LEAD, NR8, CPB, a, b_perm, BF, batch);
   const double dFU = rel_l2_cf(BF, BIP);

   // radix-16 route: same convolution, one fewer pass, lane-pair butterflies
   double dR16 = -1.0, dV16 = -1.0, dV16F = -1.0;
   {
      int L16, NR16; factorise16(N, L16, NR16);
      if(NR16 >= 1) {
         (void)0;
         std::vector<cf> b16(n);
         for(int t = 0; t < batch; t++)
            for(int i = 0; i < N; i++)
               b16[(size_t)t * N + i] = b[(size_t)t * N + mixedrev16(i, L16, NR16)];
         std::vector<cf> BR16;
         kernel_pipeline_r16(N, L16, NR16, CPB, CPP_OPT, a, b16, BR16, batch);
         // V=16 consumes the same scrambled filter, so one permutation serves both
         std::vector<cf> BV16;
         if(N % 16 == 0 && (N / 16) % CPB == 0)
            kernel_pipeline_v16(N, L16, NR16, CPB, CPP_OPT, a, b16, BV16, batch);

         dR16 = 0.0;
         for(int t = 0; t < batch; t++) {
            const size_t off = (size_t)t * N;
            std::vector<cd> ad(N), bd(N), Ad, Bd;
            for(int i = 0; i < N; i++) {
               ad[i] = cd(a[off + i].real(), a[off + i].imag());
               bd[i] = cd(b[off + i].real(), b[off + i].imag());
            }
            ref_dft(ad, Ad, -1);
            std::vector<cd> pr(N);
            for(int i = 0; i < N; i++) pr[i] = Ad[i] * bd[i];
            ref_dft(pr, Bd, +1);
            dR16 = std::max(dR16, rel_l2(BR16, Bd, off, N));
            if(!BV16.empty()) dV16 = std::max(dV16, rel_l2(BV16, Bd, off, N));
         }
         if(!BV16.empty()) {
            const int T16 = N / 16;
            std::vector<cf> bp(n);
            for(int t = 0; t < batch; t++)
               for(int L = 0; L < T16; L++)
                  for(int k = 0; k < 16; k++)
                     bp[(size_t)t * N + L + k * T16] = b16[(size_t)t * N + 16 * L + k];
            std::vector<cf> BF16;
            kernel_pipeline_v16f(N, L16, NR16, CPB, CPP_OPT, a, bp, BF16, batch);
            dV16F = rel_l2_cf(BF16, BV16);       // must be bit-identical
         }
      }
   }

   const double tol = 3e-6, bktol = 1e-6;
   bool ok = worstA < tol && worstB < tol && worstIP < tol && dA < bktol && dB < bktol
             && dIP < bktol && dFU < bktol
             && (dR16 < 0 || dR16 < tol) && (dV16 < 0 || dV16 < tol)
             && (dV16F <= 0.0) && (dSPL == 0.0);
   char r16[16], v16[16];
   if(dR16 < 0) snprintf(r16, sizeof r16, "(n/a)");
   else         snprintf(r16, sizeof r16, "%.2e", dR16);
   if(dV16 < 0) snprintf(v16, sizeof v16, "(n/a)");
   else         snprintf(v16, sizeof v16, "%.2e", dV16);
   printf("   N=%-5d %dx8^%d CPB=%-2d bt=%-3d | fp64 r8 %.2e r16 %-9s"
          " v16 %-9s | fu %.0e v16f %.0e spl %.0e | %s\n",
          N, LEAD, NR8, CPB, batch, worstIP, r16, v16, dFU,
          dV16F < 0 ? 0.0 : dV16F, dSPL, ok ? "PASS" : "FAIL");
   return ok;
}

int main(void)
{
   printf("verify_kernel: CPU transcription of fufft.cuh\n");
   printf("  slot/lane decomposition, ping-pong parity, base offsets\n");
   printf("  plus CPP_OPT vs CPP_NAIVE: are the algebraic specialisations right?\n\n");

   bool all = true;
   all &= test(8,    1, 3);
   all &= test(8,    4, 8);
   all &= test(64,   1, 3);
   all &= test(64,   8, 16);
   all &= test(512,  1, 2);
   all &= test(512,  8, 16);
   all &= test(4096, 1, 2);
   all &= test(64,  1, 8);
   all &= test(512, 1, 8);
   // the sizes radix 2 and 4 unlock
   all &= test(256,  1, 4);
   all &= test(256,  4, 8);
   all &= test(1024, 1, 4);
   all &= test(2048, 1, 2);
   all &= test(128,  1, 8);
   printf("\nint16 BFP conv (quantisation at the HBM boundary only):\n");
   all &= test_bfp(512, 0);
   all &= test_bfp(512, 1);
   all &= test_bfp(4096, 0);
   all &= test_bfp(4096, 1);

   printf("\ntwo-for-one real conv (real filter; packing commutes with conv):\n");
   all &= test_r2(512, 1, 4);
   all &= test_r2(512, 8, 8);
   all &= test_r2(256, 1, 4);

   printf("\n2D fused conv (one kernel: rows, cols, pointwise, inverse):\n");
   all &= test_2d(64, 64, 3);
   all &= test_2d(8, 512, 2);
   all &= test_2d(512, 8, 2);

   printf("\nlarge-N four-step (outer flat passes + the unchanged 4096 middle):\n");
   all &= test_large(32768, 2);
   all &= test_large(262144, 1);

   printf("\n   configurable four-step (outer radix x middle size x middle kernel):\n");
   all &= test_large_cfg(32768,  2,  8, 4096, false);   // today's shipping config
   all &= test_large_cfg(32768,  2,  8, 4096, true);    // V=16 middle
   all &= test_large_cfg(32768,  2, 16, 2048, true);    // radix-16 outer
   all &= test_large_cfg(131072, 1, 16, 8192, true);    // the 2M-class recipe
   // N/MID must be an exact power of the outer radix; 262144/4096 = 64 is not
   all &= test_large_cfg(524288, 1, 16, 2048, true);

   printf("\nflop count per radix-8 butterfly (the thing the backends differ on):\n");
   printf("   %-12s %s\n", "CPP_NAIVE", "3 generic cmul (x w1, x i, x w3) + 2 generic cmul (x i) = 5 x (4mul+2add)");
   printf("   %-12s %s\n", "CPP_OPT",   "x w1 / x w3 -> 2mul+2add each ; 3 x (x i) -> 0 flops");
   printf("   %-12s %s\n", "",          "saves 3*(4mul+2add) + 2*(2mul) = 16 mul + 6 add per butterfly");

   printf("\n%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
   return all ? 0 : 1;
}
