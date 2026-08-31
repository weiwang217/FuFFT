// Mixed-radix in-place DIF/DIT, so N can be any power of two rather than only
// a power of eight.
//
// The lane count stays T = N/8 and every lane still holds eight values, which
// is what keeps shared memory, register pressure and the block shape unchanged.
// A pass of radix R simply becomes 8/R independent butterflies on those eight
// registers:
//
//     R = 8  -> 1 radix-8 butterfly
//     R = 4  -> 2 radix-4 butterflies
//     R = 2  -> 4 radix-2 butterflies
//
// Factorisation of N = 2^m, greedy on radix 8:
//     m = 3a      ->  8^a
//     m = 3a + 1  ->  8^a x 2
//     m = 3a + 2  ->  8^a x 4
//
// DIF stride for pass i is N / (R_1 ... R_i), so the last pass always lands at
// M = 1.  DIT runs the same radices in reverse with strides 1, R_p, R_p*R_(p-1),
// and so on.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <vector>
#include <random>

typedef std::complex<double> cd;
static const double PI = 3.14159265358979323846;

// ---- factorisation --------------------------------------------------------

// Greedy on the largest radix a lane can hold, V.  The leftover goes into a
// single leading pass so the trailing passes are all radix V, which keeps the
// M == 1 boundary at radix V for the fused kernel.
static int factorise(int N, int V, int* R)
{
   int m = 0;
   while ((1 << m) < N) m++;
   if ((1 << m) != N || m < 1) return 0;

   int vb = 0; while ((1 << vb) < V) vb++;    // bits per radix-V pass
   const int a = m / vb, r = m % vb;

   int p = 0;
   if (r) R[p++] = 1 << r;
   for (int i = 0; i < a; i++) R[p++] = V;
   return p;
}

// ---- butterflies ----------------------------------------------------------

static void dft2(const cd u[2], cd v[2], int)
{
   v[0] = u[0] + u[1];
   v[1] = u[0] - u[1];
}

static void dft4(const cd u[4], cd v[4], int sign)
{
   const cd i2(0.0, (double)sign);
   cd a0 = u[0] + u[2], a1 = u[1] + u[3];
   cd a2 = u[0] - u[2], a3 = (u[1] - u[3]) * i2;
   v[0] = a0 + a1;
   v[2] = a0 - a1;
   v[1] = a2 + a3;
   v[3] = a2 - a3;
}

static void dft8(const cd u[8], cd v[8], int sign)
{
   const double r2 = 0.70710678118654752440;
   const cd w1(r2, sign * r2), w2(0.0, (double)sign), w3(-r2, sign * r2);

   cd a0 = u[0] + u[4], a1 = u[1] + u[5], a2 = u[2] + u[6], a3 = u[3] + u[7];
   cd a4 = u[0] - u[4];
   cd a5 = (u[1] - u[5]) * w1;
   cd a6 = (u[2] - u[6]) * w2;
   cd a7 = (u[3] - u[7]) * w3;

   cd b0 = a0 + a2, b1 = a1 + a3, b2 = a0 - a2;
   cd b3 = (a1 - a3) * w2;
   cd b4 = a4 + a6, b5 = a5 + a7, b6 = a4 - a6;
   cd b7 = (a5 - a7) * w2;

   v[0] = b0 + b1;  v[4] = b0 - b1;
   v[2] = b2 + b3;  v[6] = b2 - b3;
   v[1] = b4 + b5;  v[5] = b4 - b5;
   v[3] = b6 + b7;  v[7] = b6 - b7;
}

// radix 16 as one radix-2 DIF stage feeding two radix-8s:
//    X[2k]   = DFT8( u[i] + u[i+8] )[k]
//    X[2k+1] = DFT8( (u[i] - u[i+8]) * W16^i )[k]
// Reuses the verified dft8, and W16^0 = 1 and W16^4 = +-i are free.
static void dft16(const cd u[16], cd v[16], int sign)
{
   cd a[8], b[8], va[8], vb[8];
   for (int i = 0; i < 8; i++) {
      a[i] = u[i] + u[i + 8];
      cd d = u[i] - u[i + 8];
      double ang = sign * 2.0 * PI * (double)i / 16.0;
      b[i] = d * cd(cos(ang), sin(ang));
   }
   dft8(a, va, sign);
   dft8(b, vb, sign);
   for (int k = 0; k < 8; k++) { v[2 * k] = va[k]; v[2 * k + 1] = vb[k]; }
}

static void dftR(int R, const cd* u, cd* v, int sign)
{
   if (R == 2) dft2(u, v, sign);
   else if (R == 4) dft4(u, v, sign);
   else if (R == 8) dft8(u, v, sign);
   else dft16(u, v, sign);
}

// ---- one pass -------------------------------------------------------------
// lane L owns the groups { L, L+T, ..., L+(G-1)T } with G = 8/R.

static void pass_R(cd* A, int R, int M, int sign, int T, int N, bool dit, int V)
{
   const int G = V / R;
   (void)N;
   for (int lane = 0; lane < T; lane++) {
      for (int s = 0; s < G; s++) {
         const int gidx = lane + s * T;
         const int b = gidx / M;
         const int j = gidx % M;
         const int base = b * R * M + j;

         cd u[16], v[16];
         for (int k = 0; k < R; k++) u[k] = A[base + k * M];

         if (dit && M > 1)
            for (int k = 1; k < R; k++) {
               double ang = sign * 2.0 * PI * (double)(j * k) / (double)(R * M);
               u[k] *= cd(cos(ang), sin(ang));
            }

         dftR(R, u, v, sign);

         if (!dit && M > 1)
            for (int k = 1; k < R; k++) {
               double ang = sign * 2.0 * PI * (double)(j * k) / (double)(R * M);
               v[k] *= cd(cos(ang), sin(ang));
            }

         for (int k = 0; k < R; k++) A[base + k * M] = v[k];
      }
   }
}

// forward DIF: strides N/R1, N/(R1 R2), ... 1
static void fft_dif(cd* A, int N, int sign, const int* R, int p, int V)
{
   const int T = N / V;
   int prod = 1;
   for (int i = 0; i < p; i++) {
      prod *= R[i];
      pass_R(A, R[i], N / prod, sign, T, N, false, V);
   }
}

// inverse DIT: radices in reverse, strides 1, Rp, Rp*R(p-1), ...
static void fft_dit(cd* A, int N, int sign, const int* R, int p, int V)
{
   const int T = N / V;
   int M = 1;
   for (int i = p - 1; i >= 0; i--) {
      pass_R(A, R[i], M, sign, T, N, true, V);
      M *= R[i];
   }
}

// The scramble a DIF leaves behind is a mixed-radix digit reversal: with
// radices R1..Rp the input digit weights are read out in reverse order.
// A DIF splits by the MOST significant digit first, so R[0] is the top group
// of bits, not the bottom.  Every radix is a power of two, so this is a
// reversal of bit *groups*: the widths permute, the bits inside a group do not.
static int mixedrev(int idx, const int* R, int p)
{
   int w[16], m = 0;
   for (int i = 0; i < p; i++) { w[i] = 0; while ((1 << w[i]) < R[i]) w[i]++; m += w[i]; }

   int d[16], pos = m;
   for (int i = 0; i < p; i++) { pos -= w[i]; d[i] = (idx >> pos) & (R[i] - 1); }

   int out = 0, outpos = m;
   for (int i = p - 1; i >= 0; i--) { outpos -= w[i]; out |= d[i] << outpos; }
   return out;
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

static bool run_for_V(int V, std::mt19937& rng)
{
   std::uniform_real_distribution<double> u(-1.0, 1.0);
   const int sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
   const int ns = (int)(sizeof(sizes) / sizeof(sizes[0]));
   bool all = true;

   printf("\n########## V = %d values per lane  (T = N/%d, max radix %d) ##########\n", V, V, V);
   printf("  %-7s %-22s %6s %8s\n", "N", "radices", "passes", "bits/pass");
   for (int t = 0; t < ns; t++) {
      const int N = sizes[t];
      int R[16], p = factorise(N, V, R);
      if (!p) { printf("  N=%-5d  unsupported\n", N); all = false; continue; }
      char buf[64] = {0}; int off = 0, prod = 1;
      for (int i = 0; i < p; i++) { off += snprintf(buf + off, sizeof buf - off, "%d ", R[i]); prod *= R[i]; }
      int m = 0; while ((1 << m) < N) m++;
      all &= (prod == N);
      printf("  N=%-5d  %-22s %6d %8.2f%s\n", N, buf, p, (double)m / p, prod == N ? "" : "  FAIL");
   }

   printf("\n  round trip / convolution:\n");
   for (int t = 0; t < ns; t++) {
      const int N = sizes[t];
      if (N < V) continue;                      // a lane must hold at least one butterfly
      int R[16], p = factorise(N, V, R);
      if (!p) continue;

      std::vector<cd> x(N);
      for (int i = 0; i < N; i++) x[i] = cd(u(rng), u(rng));
      std::vector<cd> A = x;
      fft_dif(A.data(), N, -1, R, p, V);
      fft_dit(A.data(), N, +1, R, p, V);
      for (int i = 0; i < N; i++) A[i] /= (double)N;
      double e1 = rel(A, x);

      double e2 = -1.0;
      if (N <= 1024) {
         std::vector<cd> a2(N), b(N);
         for (int i = 0; i < N; i++) { a2[i] = cd(u(rng), u(rng)); b[i] = cd(u(rng), u(rng)); }
         std::vector<cd> ref(N, cd(0, 0));
         for (int k = 0; k < N; k++) {
            cd acc(0, 0);
            for (int i = 0; i < N; i++) acc += a2[i] * b[(k - i + N) % N];
            ref[k] = acc;
         }
         std::vector<cd> B = b;
         fft_dif(B.data(), N, -1, R, p, V);
         for (int i = 0; i < N; i++) B[i] /= (double)N;
         std::vector<cd> C = a2;
         fft_dif(C.data(), N, -1, R, p, V);
         for (int i = 0; i < N; i++) C[i] *= B[i];
         fft_dit(C.data(), N, +1, R, p, V);
         e2 = rel(C, ref);
      }
      // the only check on the output ORDER: unscramble with the mixed-radix
      // digit reversal and it must equal a naive DFT
      double e3 = -1.0;
      if (N <= 1024) {
         std::vector<cd> y(N);
         for (int i = 0; i < N; i++) y[i] = cd(u(rng), u(rng));
         std::vector<cd> D = y, refd;
         fft_dif(D.data(), N, -1, R, p, V);
         naive_dft(y, refd, -1);
         std::vector<cd> un(N);
         for (int i = 0; i < N; i++) un[mixedrev(i, R, p)] = D[i];
         e3 = rel(un, refd);
      }

      bool ok = e1 < 1e-13 && (e2 < 0 || e2 < 1e-13)
                && (e3 < 0 || e3 < 100.0 * N * 2.3e-16);
      all &= ok;
      printf("    N=%-5d roundtrip %.2e  conv %-9s order %-9s %s\n", N, e1,
             e2 < 0 ? "(skip)" : ([&]{ static char b2[16]; snprintf(b2, sizeof b2, "%.2e", e2); return b2; })(),
             e3 < 0 ? "(skip)" : ([&]{ static char b3[16]; snprintf(b3, sizeof b3, "%.2e", e3); return b3; })(),
             ok ? "PASS" : "FAIL");
   }
   return all;
}

int main(void)
{
   printf("mixed-radix in-place DIF/DIT, radix 2/4/8/16\n");
   printf("  a lane holds V values, so the largest usable radix is V;\n");
   printf("  a pass of radix R runs V/R independent butterflies on those registers.\n");
   printf("  Shared traffic per pass is 2N regardless of R, so more bits per pass\n");
   printf("  means fewer passes means less shared traffic -- paid for in registers.\n");

   std::mt19937 rng(11);
   bool all = true;
   all &= run_for_V(8, rng);
   all &= run_for_V(16, rng);

   printf("\n########## pass count: V=8 vs V=16 ##########\n");
   const int sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
   printf("  %-7s %8s %8s %8s\n", "N", "V=8", "V=16", "saved");
   for (int t = 0; t < (int)(sizeof(sizes)/sizeof(sizes[0])); t++) {
      int R8[16], R16[16];
      int p8 = factorise(sizes[t], 8, R8), p16 = factorise(sizes[t], 16, R16);
      printf("  N=%-5d %8d %8d %8s\n", sizes[t], p8, p16,
             p16 < p8 ? "-1 pass" : "same");
   }

   printf("\n%s\n", all ? "ALL PASS" : "FAILURES PRESENT");
   return all ? 0 : 1;
}
