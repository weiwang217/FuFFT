// In-place radix-8 Cooley-Tukey, single shared buffer instead of Stockham's
// ping-pong pair.  Halves shared memory, which is what caps occupancy at
// 2 blocks/SM in the current kernel.
//
//   forward  DIF : natural order in, digit-reversed (base 8) out.
//                  butterfly first, then twiddle.
//   inverse  DIT : digit-reversed in, natural order out.
//                  twiddle first, then butterfly.  Stages run in the opposite
//                  stride order to DIF.
//
// For a convolution the digit reversal costs nothing: fwd DIF -> pointwise
// against a digit-reversed filter spectrum -> inv DIT lands back in natural
// order, and the filter is permuted once on the host.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <random>

typedef std::complex<double> cd;
typedef std::complex<float>  cf;
static const double PI = 3.14159265358979323846;

static int ilog8(int n) { int s = 0; while (n > 1) { n /= 8; s++; } return s; }

static int digitrev8(int i, int S)
{
   int r = 0;
   for (int s = 0; s < S; s++) { r = r * 8 + (i & 7); i >>= 3; }
   return r;
}

template<typename C>
static void dft8(const C u[8], C v[8], int sign)
{
   typedef typename C::value_type R;
   const R r2 = (R)0.70710678118654752440;
   const C w1( r2, (R)sign * r2);
   const C w2((R)0, (R)sign);
   const C w3(-r2, (R)sign * r2);

   C a0 = u[0] + u[4], a1 = u[1] + u[5], a2 = u[2] + u[6], a3 = u[3] + u[7];
   C a4 = u[0] - u[4];
   C a5 = (u[1] - u[5]) * w1;
   C a6 = (u[2] - u[6]) * w2;
   C a7 = (u[3] - u[7]) * w3;

   C b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2;
   C b3 = (a1 - a3) * w2;
   C b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6;
   C b7 = (a5 - a7) * w2;

   v[0] = b0 + b1;  v[4] = b0 - b1;
   v[2] = b2 + b3;  v[6] = b2 - b3;
   v[1] = b4 + b5;  v[5] = b4 - b5;
   v[3] = b6 + b7;  v[7] = b6 - b7;
}

// One in-place DIF pass.  The stride M fully determines the geometry, so N
// is not needed here.  lane indexes the group (b, j) = (lane/M, lane%M).
template<typename C>
static void dif_pass(C* A, int M, int sign, int T)
{
   typedef typename C::value_type R;
   for (int lane = 0; lane < T; lane++) {
      const int j = lane % M;
      const int b = lane / M;
      const int base = b * 8 * M + j;

      C u[8], v[8];
      for (int k = 0; k < 8; k++) u[k] = A[base + k * M];
      dft8<C>(u, v, sign);
      // twiddle AFTER the butterfly
      for (int k = 1; k < 8; k++) {
         double ang = sign * 2.0 * PI * (double)(j * k) / (double)(8 * M);
         v[k] *= C((R)cos(ang), (R)sin(ang));
      }
      for (int k = 0; k < 8; k++) A[base + k * M] = v[k];
   }
}

// One in-place DIT pass.  Same geometry, but twiddle BEFORE the butterfly.
template<typename C>
static void dit_pass(C* A, int M, int sign, int T)
{
   typedef typename C::value_type R;
   for (int lane = 0; lane < T; lane++) {
      const int j = lane % M;
      const int b = lane / M;
      const int base = b * 8 * M + j;

      C u[8], v[8];
      for (int k = 0; k < 8; k++) u[k] = A[base + k * M];
      for (int k = 1; k < 8; k++) {
         double ang = sign * 2.0 * PI * (double)(j * k) / (double)(8 * M);
         u[k] *= C((R)cos(ang), (R)sin(ang));
      }
      dft8<C>(u, v, sign);
      for (int k = 0; k < 8; k++) A[base + k * M] = v[k];
   }
}

// forward: natural -> digit-reversed.  M runs N/8, N/64, ... 1
template<typename C>
static void fft_dif(C* A, int N, int sign)
{
   const int S = ilog8(N), T = N / 8;
   for (int s = 0; s < S; s++) {
      int M = N;
      for (int t = 0; t <= s; t++) M /= 8;
      dif_pass<C>(A, M, sign, T);
   }
}

// inverse: digit-reversed -> natural.  M runs 1, 8, 64, ...
template<typename C>
static void fft_dit(C* A, int N, int sign)
{
   const int S = ilog8(N), T = N / 8;
   int M = 1;
   for (int s = 0; s < S; s++) { dit_pass<C>(A, M, sign, T); M *= 8; }
}

static void naive_dft(const std::vector<cd>& in, std::vector<cd>& out, int sign)
{
   int n = (int)in.size();
   out.assign(n, cd(0, 0));
   for (int k = 0; k < n; k++) {
      cd acc(0, 0);
      for (int j = 0; j < n; j++) {
         double a = sign * 2.0 * PI * (double)k * (double)j / (double)n;
         acc += in[j] * cd(cos(a), sin(a));
      }
      out[k] = acc;
   }
}

static double rel(const std::vector<cd>& a, const std::vector<cd>& b)
{
   double n = 0, d = 0;
   for (size_t i = 0; i < a.size(); i++) { n += std::norm(a[i] - b[i]); d += std::norm(b[i]); }
   return sqrt(n / d);
}

int main(void)
{
   printf("in-place radix-8 DIF/DIT -- algorithm check\n\n");
   std::mt19937 rng(7);
   std::uniform_real_distribution<double> u(-1.0, 1.0);
   bool all = true;

   printf("== forward DIF produces the DFT in digit-reversed order ==\n");
   for (int N : {8, 64, 512, 4096}) {
      const int S = ilog8(N);
      std::vector<cd> x(N);
      for (int i = 0; i < N; i++) x[i] = cd(u(rng), u(rng));

      std::vector<cd> A = x, ref;
      fft_dif<cd>(A.data(), N, -1);
      naive_dft(x, ref, -1);

      std::vector<cd> unscrambled(N);
      for (int i = 0; i < N; i++) unscrambled[digitrev8(i, S)] = A[i];

      double e = rel(unscrambled, ref);
      bool ok = e < 100.0 * N * 2.3e-16;
      all &= ok;
      printf("   N=%-5d rel %.3e  %s\n", N, e, ok ? "PASS" : "FAIL");
   }

   printf("\n== DIF then DIT round trip (no reordering in between) ==\n");
   for (int N : {8, 64, 512, 4096}) {
      std::vector<cd> x(N);
      for (int i = 0; i < N; i++) x[i] = cd(u(rng), u(rng));
      std::vector<cd> A = x;
      fft_dif<cd>(A.data(), N, -1);
      fft_dit<cd>(A.data(), N, +1);
      for (int i = 0; i < N; i++) A[i] /= (double)N;
      double e = rel(A, x);
      bool ok = e < 1e-13;
      all &= ok;
      printf("   N=%-5d rel %.3e  %s\n", N, e, ok ? "PASS" : "FAIL");
   }

   printf("\n== cyclic convolution: fwd DIF -> x digitrev(bspec) -> inv DIT ==\n");
   for (int N : {8, 64, 512}) {
      const int S = ilog8(N);
      std::vector<cd> a(N), b(N);
      for (int i = 0; i < N; i++) { a[i] = cd(u(rng), u(rng)); b[i] = cd(u(rng), u(rng)); }

      std::vector<cd> ref(N, cd(0, 0));
      for (int k = 0; k < N; k++) {
         cd acc(0, 0);
         for (int i = 0; i < N; i++) acc += a[i] * b[(k - i + N) % N];
         ref[k] = acc;
      }

      // the filter spectrum, permuted once into digit-reversed order
      std::vector<cd> B = b, Bs(N);
      fft_dif<cd>(B.data(), N, -1);
      for (int i = 0; i < N; i++) Bs[i] = B[i] * (1.0 / (double)N);

      std::vector<cd> A = a;
      fft_dif<cd>(A.data(), N, -1);
      for (int i = 0; i < N; i++) A[i] *= Bs[i];   // both already digit-reversed
      fft_dit<cd>(A.data(), N, +1);

      double e = rel(A, ref);
      bool ok = e < 1e-13;
      all &= ok;
      printf("   N=%-5d rel %.3e  %s   (S=%d, no in-kernel reorder)\n", N, e, ok ? "PASS" : "FAIL", S);
   }

   printf("\n== digit reversal is an involution (so it can be done in place) ==\n");
   for (int N : {8, 64, 512, 4096}) {
      const int S = ilog8(N);
      bool ok = true;
      for (int i = 0; i < N; i++) if (digitrev8(digitrev8(i, S), S) != i) ok = false;
      all &= ok;
      printf("   N=%-5d %s\n", N, ok ? "PASS" : "FAIL");
   }

   printf("\n%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
   return all ? 0 : 1;
}
