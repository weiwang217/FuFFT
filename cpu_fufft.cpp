#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <complex>
#include <vector>
#include <random>

typedef std::complex<double> cd;
typedef std::complex<float>  cf;

static const double PI = 3.14159265358979323846;

static void naive_dft(const std::vector<cd>& in, std::vector<cd>& out, int sign)
{
   int n = (int)in.size();
   out.assign(n, cd(0.0, 0.0));
   for(int k = 0; k < n; k++) {
      cd acc(0.0, 0.0);
      for(int j = 0; j < n; j++) {
         double a = sign * 2.0 * PI * (double)k * (double)j / (double)n;
         acc += in[j] * cd(cos(a), sin(a));
      }
      out[k] = acc;
   }
}

template<typename C>
static void dft8_direct(const C u[8], C v[8], int sign)
{
   typedef typename C::value_type R;
   for(int k = 0; k < 8; k++) {
      C acc(0, 0);
      for(int j = 0; j < 8; j++) {
         double a = sign * 2.0 * PI * (double)(k * j) / 8.0;
         acc += u[j] * C((R)cos(a), (R)sin(a));
      }
      v[k] = acc;
   }
}

template<typename C>
static void dft8_staged(const C u[8], C v[8], int sign)
{
   typedef typename C::value_type R;
   const R s = (R)(sign * 0.70710678118654752440);
   const R one = (R)1;

   C w1((R)0.70710678118654752440, s);
   C w2((R)0, (R)sign);
   C w3((R)-0.70710678118654752440, s);
   (void)one;

   C a0 = u[0] + u[4];
   C a1 = u[1] + u[5];
   C a2 = u[2] + u[6];
   C a3 = u[3] + u[7];
   C a4 = u[0] - u[4];
   C a5 = (u[1] - u[5]) * w1;
   C a6 = (u[2] - u[6]) * w2;
   C a7 = (u[3] - u[7]) * w3;

   C b0 = a0 + a2;
   C b1 = a1 + a3;
   C b2 = a0 - a2;
   C b3 = (a1 - a3) * w2;
   C b4 = a4 + a6;
   C b5 = a5 + a7;
   C b6 = a4 - a6;
   C b7 = (a5 - a7) * w2;

   v[0] = b0 + b1;
   v[4] = b0 - b1;
   v[2] = b2 + b3;
   v[6] = b2 - b3;
   v[1] = b4 + b5;
   v[5] = b4 - b5;
   v[3] = b6 + b7;
   v[7] = b6 - b7;
}

static int ilog8(int n)
{
   int s = 0;
   while(n > 1) { n /= 8; s++; }
   return s;
}

static bool is_pow8(int n)
{
   if(n < 8) return false;
   while(n > 1) {
      if(n % 8) return false;
      n /= 8;
   }
   return true;
}

template<typename C>
static void fufft_stockham8(std::vector<C>& A, std::vector<C>& B, int N, int sign)
{
   typedef typename C::value_type R;
   const int r = 8;
   const int S = ilog8(N);
   const int nthreads = N / r;

   int m = 1;
   for(int s = 0; s < S; s++) {
      m *= r;
      const int mh = m / r;

      for(int t = 0; t < nthreads; t++) {
         const int j = t % mh;
         const int i = t / mh;

         C u[8];
         for(int k = 0; k < r; k++)
            u[k] = A[(size_t)i * mh + j + (size_t)k * (N / r)];

         for(int k = 1; k < r; k++) {
            double ang = sign * 2.0 * PI * (double)(j * k) / (double)m;
            u[k] *= C((R)cos(ang), (R)sin(ang));
         }

         C v[8];
         dft8_staged<C>(u, v, sign);

         for(int k = 0; k < r; k++)
            B[(size_t)i * m + j + (size_t)k * mh] = v[k];
      }
      A.swap(B);
   }
}

template<typename C>
static void fufft_forward(std::vector<C>& A, std::vector<C>& B, int N)
{
   fufft_stockham8<C>(A, B, N, -1);
}

template<typename C>
static void fufft_inverse(std::vector<C>& A, std::vector<C>& B, int N)
{
   typedef typename C::value_type R;
   fufft_stockham8<C>(A, B, N, +1);
   const R inv = (R)1 / (R)N;
   for(int i = 0; i < N; i++) A[i] *= inv;
}

static double max_abs_err(const std::vector<cd>& a, const std::vector<cd>& b)
{
   double e = 0.0;
   for(size_t i = 0; i < a.size(); i++) e = std::max(e, std::abs(a[i] - b[i]));
   return e;
}

static double rms(const std::vector<cd>& a)
{
   double s = 0.0;
   for(size_t i = 0; i < a.size(); i++) s += std::norm(a[i]);
   return sqrt(s / (double)a.size());
}

static void check_dft8()
{
   printf("== radix-8 kernel: staged vs direct ==\n");
   std::mt19937 rng(1);
   std::uniform_real_distribution<double> d(-1.0, 1.0);
   double worst = 0.0;
   for(int trial = 0; trial < 200; trial++) {
      cd u[8], vd[8], vs[8];
      for(int i = 0; i < 8; i++) u[i] = cd(d(rng), d(rng));
      for(int sign = -1; sign <= 1; sign += 2) {
         dft8_direct<cd>(u, vd, sign);
         dft8_staged<cd>(u, vs, sign);
         for(int i = 0; i < 8; i++) worst = std::max(worst, std::abs(vd[i] - vs[i]));
      }
   }
   printf("   max |staged - direct| over 200 random vectors = %.3e  %s\n\n",
          worst, worst < 1e-12 ? "PASS" : "FAIL");
}

static void check_transform(int N)
{
   std::mt19937 rng(N);
   std::uniform_real_distribution<double> d(-1.0, 1.0);

   std::vector<cd> x(N), ref, A(N), B(N);
   for(int i = 0; i < N; i++) x[i] = cd(d(rng), d(rng));

   A = x;
   fufft_forward<cd>(A, B, N);

   double fwd_err = -1.0;
   if(N <= 4096) {
      naive_dft(x, ref, -1);
      fwd_err = max_abs_err(A, ref) / rms(ref);
   }

   std::vector<cd> C2 = A, D(N);
   fufft_inverse<cd>(C2, D, N);
   double rt_err = max_abs_err(C2, x) / rms(x);

   // The naive DFT reference is itself only accurate to O(N*eps): it sums N
   // terms per output, while the FFT sums log8(N). Scale the tolerance so we
   // are testing our transform, not the reference.
   const double eps = 2.220446049250313e-16;
   double dft_tol = 100.0 * (double)N * eps;
   double rt_tol  = 100.0 * (double)ilog8(N) * eps;

   char fwd_txt[32];
   if(fwd_err < 0) snprintf(fwd_txt, sizeof fwd_txt, "(skipped)");
   else            snprintf(fwd_txt, sizeof fwd_txt, "%.3e", fwd_err);

   bool ok = (rt_err < rt_tol) && (fwd_err < 0 || fwd_err < dft_tol);
   printf("   N=%-7d  fwd-vs-DFT %-12s  roundtrip %.3e  (tol %.1e)  %s\n",
          N, fwd_txt, rt_err, rt_tol, ok ? "PASS" : "FAIL");
}

static void check_convolution(int N)
{
   std::mt19937 rng(N * 7 + 1);
   std::uniform_real_distribution<double> d(-1.0, 1.0);

   std::vector<cd> a(N), b(N);
   for(int i = 0; i < N; i++) { a[i] = cd(d(rng), d(rng)); b[i] = cd(d(rng), d(rng)); }

   std::vector<cd> ref(N, cd(0, 0));
   for(int k = 0; k < N; k++) {
      cd acc(0, 0);
      for(int i = 0; i < N; i++) acc += a[i] * b[(k - i + N) % N];
      ref[k] = acc;
   }

   std::vector<cd> A = a, B = b, T(N);
   fufft_forward<cd>(A, T, N);
   fufft_forward<cd>(B, T, N);
   for(int i = 0; i < N; i++) A[i] *= B[i];
   fufft_inverse<cd>(A, T, N);

   double e = max_abs_err(A, ref) / rms(ref);
   printf("   N=%-7d  cyclic-conv rel err %.3e  %s\n", N, e, e < 1e-12 ? "PASS" : "FAIL");
}

static void float_accuracy(int N)
{
   std::mt19937 rng(N + 99);
   std::uniform_real_distribution<double> d(-1.0, 1.0);

   std::vector<cd> xd(N);
   std::vector<cf> xf(N);
   for(int i = 0; i < N; i++) {
      cd v(d(rng), d(rng));
      xd[i] = v;
      xf[i] = cf((float)v.real(), (float)v.imag());
   }

   std::vector<cd> Ad = xd, Td(N);
   fufft_forward<cd>(Ad, Td, N);

   std::vector<cf> Af = xf, Tf(N);
   fufft_forward<cf>(Af, Tf, N);

   double num = 0.0, den = 0.0;
   for(int i = 0; i < N; i++) {
      cd f((double)Af[i].real(), (double)Af[i].imag());
      num += std::norm(f - Ad[i]);
      den += std::norm(Ad[i]);
   }
   printf("   N=%-7d  fp32 vs fp64 rel L2 = %.3e   (~2^%.1f)\n",
          N, sqrt(num / den), log2(sqrt(num / den)));
}

int main(int argc, char** argv)
{
   printf("fuFFT CPU reference -- same algorithm the CUDA kernel implements\n");
   printf("  Stockham autosort, radix 8, 8 values per thread in registers\n");
   printf("  radix-8, registers-first dataflow; reference for the device kernels\n\n");

   check_dft8();

   printf("== full transform vs naive DFT, and forward/inverse roundtrip ==\n");
   int sizes[] = { 8, 64, 512, 4096, 32768, 262144 };
   for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
      if(!is_pow8(sizes[i]) && sizes[i] != 8) continue;
      check_transform(sizes[i]);
   }
   printf("\n== cyclic convolution (fwd -> pointwise -> inv) ==\n");
   for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
      if(sizes[i] > 4096) break;
      check_convolution(sizes[i]);
   }
   printf("\n== single precision accuracy of the same algorithm ==\n");
   for(size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
      float_accuracy(sizes[i]);

   printf("\nsupported sizes are powers of 8: ");
   for(int n = 8; n <= (1 << 24); n *= 8) printf("%d ", n);
   printf("\n");
   (void)argc; (void)argv;
   return 0;
}
