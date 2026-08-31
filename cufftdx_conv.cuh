// cuFFTDx comparison -- the incumbent this project should have been measured
// against from the start.
//
// cuFFTDx executes an FFT INSIDE the caller's kernel from registers + shared
// memory, so it does the same fused FFT x multiply x IFFT in one kernel that
// fufft_pipeline_fused does: 3 HBM passes, pointwise in registers.  Structure
// below mirrors NVIDIA's own 06_convolution sample.
//
// Build: cuFFTDx ships in MathDx, not the CUDA toolkit.
//   make run-dx SM=90 CUFFTDX_DIR=/path/to/nvidia-mathdx-*/include
// Without CUFFTDX_DIR the target is skipped and the rest of the bench is
// unaffected.
#pragma once
#ifdef FUFFT_HAVE_CUFFTDX
#include <cufftdx.hpp>

template<int N, int EPT, int FPB, int ARCH>
struct DxConv {
   using FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>()
                      + cufftdx::Type<cufftdx::fft_type::c2c>()
                      + cufftdx::Direction<cufftdx::fft_direction::forward>()
                      + cufftdx::Block() + cufftdx::ElementsPerThread<EPT>()
                      + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>());
   using IFFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<float>()
                       + cufftdx::Type<cufftdx::fft_type::c2c>()
                       + cufftdx::Direction<cufftdx::fft_direction::inverse>()
                       + cufftdx::Block() + cufftdx::ElementsPerThread<EPT>()
                       + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>());
   using cx = typename FFT::value_type;
};

// One kernel: load -> FFT -> multiply by the filter spectrum IN REGISTERS ->
// IFFT -> store.  `bspec` must be in the same element order the forward FFT
// leaves in thread_data, which is the identical (threadIdx.x + i*stride) walk
// used for the load, so indexing it the same way pairs them correctly.
template<int N, int EPT, int FPB, int ARCH>
__launch_bounds__(DxConv<N, EPT, FPB, ARCH>::FFT::max_threads_per_block)
__global__ void dx_conv_kernel(const float2* __restrict__ a,
                               const float2* __restrict__ bspec,
                               float2* __restrict__ out)
{
   using D  = DxConv<N, EPT, FPB, ARCH>;
   using FFT = typename D::FFT;
   using IFFT = typename D::IFFT;
   using cx = typename D::cx;

   extern __shared__ __align__(alignof(float4)) unsigned char smem_raw[];
   cx* smem = reinterpret_cast<cx*>(smem_raw);

   cx thread_data[FFT::storage_size];

   const unsigned int local_id = threadIdx.y;
   const unsigned int fft_id   = blockIdx.x * FPB + local_id;
   const size_t base = (size_t)fft_id * N;

   unsigned int idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      const float2 v = a[base + idx];
      thread_data[i] = cx{v.x, v.y};
      idx += FFT::stride;
   }

   FFT().execute(thread_data, smem);

   idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      const float2 w = bspec[base + idx];
      const cx x = thread_data[i];
      thread_data[i] = cx{x.x * w.x - x.y * w.y, x.x * w.y + x.y * w.x};
      idx += FFT::stride;
   }

   IFFT().execute(thread_data, smem);

   idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      out[base + idx] = make_float2(thread_data[i].x, thread_data[i].y);
      idx += FFT::stride;
   }
}

template<int N, int EPT, int FPB, int ARCH>
static float dx_bench(const float2* d_a, const float2* d_bspec, float2* d_out,
                      int batch, int iters, Timer& tm, double* err_out,
                      const std::vector<float2>& h_ref)
{
   using D = DxConv<N, EPT, FPB, ARCH>;
   using FFT = typename D::FFT;
   const auto smem = FFT::shared_memory_size;
   const dim3 blk(FFT::block_dim.x, FFT::block_dim.y, 1);
   const int grid = batch / FPB;

   CK(cudaFuncSetAttribute((dx_conv_kernel<N, EPT, FPB, ARCH>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

   dx_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_a, d_bspec, d_out);
   CK(cudaGetLastError());
   CK(cudaDeviceSynchronize());
   if(err_out) {
      std::vector<float2> h(h_ref.size());
      CK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(float2), cudaMemcpyDeviceToHost));
      *err_out = rel_l2(h, h_ref);
   }
   for(int i = 0; i < 5; i++)
      dx_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_a, d_bspec, d_out);
   CK(cudaDeviceSynchronize());
   tm.start();
   for(int i = 0; i < iters; i++)
      dx_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_a, d_bspec, d_out);
   return tm.stop() / iters;
}

// ---------------------------------------------------------------------------
// fp16.  cuFFTDx half precision is implicitly batched: value_type is
// complex<__half2>, and each element carries ONE point of TWO transforms
// (lane 0 = even batch, lane 1 = odd batch; .x holds the two reals, .y the
// two imags).  Same walk and registers-multiply structure as the fp32 kernel
// above.  The filter spectrum arrives as the bench's natural-order fp32
// spectrum with 1/N already folded in -- that folding is also what keeps
// every intermediate inside fp16 range -- and is packed to the paired-half
// layout once, off the clock.  4 bytes per point: the same traffic as the
// int16-BFP kernel, so this is the like-for-like 16-bit comparison.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>

template<int N, int EPT, int FPB, int ARCH>
struct DxConvH {
   using FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<__half>()
                      + cufftdx::Type<cufftdx::fft_type::c2c>()
                      + cufftdx::Direction<cufftdx::fft_direction::forward>()
                      + cufftdx::Block() + cufftdx::ElementsPerThread<EPT>()
                      + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>());
   using IFFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<__half>()
                       + cufftdx::Type<cufftdx::fft_type::c2c>()
                       + cufftdx::Direction<cufftdx::fft_direction::inverse>()
                       + cufftdx::Block() + cufftdx::ElementsPerThread<EPT>()
                       + cufftdx::FFTsPerBlock<FPB>() + cufftdx::SM<ARCH>());
   using cx = typename FFT::value_type;                     // complex<__half2>
};

// batches (2p, 2p+1) of a float2 stream -> one paired-half element stream
template<typename CX>
__global__ void dx16_pack(const float2* __restrict__ a, CX* __restrict__ o,
                          int npts, size_t total)
{
   const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(k >= total) return;
   const size_t p = k / npts, i = k % npts;
   const float2 va = a[(2 * p) * npts + i], vb = a[(2 * p + 1) * npts + i];
   o[k] = CX{__floats2half2_rn(va.x, vb.x), __floats2half2_rn(va.y, vb.y)};
}

template<typename CX>
__global__ void dx16_unpack(const CX* __restrict__ o, float2* __restrict__ out,
                            int npts, size_t total)
{
   const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
   if(k >= total) return;
   const size_t p = k / npts, i = k % npts;
   out[(2 * p)     * npts + i] = make_float2(__low2float(o[k].x),  __low2float(o[k].y));
   out[(2 * p + 1) * npts + i] = make_float2(__high2float(o[k].x), __high2float(o[k].y));
}

template<int N, int EPT, int FPB, int ARCH>
__launch_bounds__(DxConvH<N, EPT, FPB, ARCH>::FFT::max_threads_per_block)
__global__ void dx16_conv_kernel(
      const typename DxConvH<N, EPT, FPB, ARCH>::cx* __restrict__ a,
      const typename DxConvH<N, EPT, FPB, ARCH>::cx* __restrict__ bspec,
      typename DxConvH<N, EPT, FPB, ARCH>::cx* __restrict__ out)
{
   using D  = DxConvH<N, EPT, FPB, ARCH>;
   using FFT = typename D::FFT;
   using IFFT = typename D::IFFT;
   using cx = typename D::cx;

   extern __shared__ __align__(alignof(float4)) unsigned char smem_raw[];
   cx* smem = reinterpret_cast<cx*>(smem_raw);

   cx thread_data[FFT::storage_size];

   // FPB counts logical FFTs; the block covers blockDim.y = FPB/2 pairs
   const unsigned int pair_id = blockIdx.x * blockDim.y + threadIdx.y;
   const size_t base = (size_t)pair_id * N;

   unsigned int idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      thread_data[i] = a[base + idx];
      idx += FFT::stride;
   }

   FFT().execute(thread_data, smem);

   idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      const cx w = bspec[base + idx];
      const __half2 xr = thread_data[i].x, xi = thread_data[i].y;
      thread_data[i].x = __hsub2(__hmul2(xr, w.x), __hmul2(xi, w.y));
      thread_data[i].y = __hadd2(__hmul2(xr, w.y), __hmul2(xi, w.x));
      idx += FFT::stride;
   }

   IFFT().execute(thread_data, smem);

   idx = threadIdx.x;
#pragma unroll
   for(int i = 0; i < FFT::elements_per_thread; i++) {
      out[base + idx] = thread_data[i];
      idx += FFT::stride;
   }
}

// d_scratch: any n-point float2 buffer no longer being timed (unpack target).
template<int N, int EPT, int FPB, int ARCH>
static float dx16_bench(const float2* d_a, const float2* d_bspec,
                        float2* d_scratch, int batch, int iters, Timer& tm,
                        double* err_out, const std::vector<float2>& h_ref)
{
   using D = DxConvH<N, EPT, FPB, ARCH>;
   using FFT = typename D::FFT;
   using cx = typename D::cx;
   const size_t total = (size_t)N * (batch / 2);      // paired elements

   cx *d_ah, *d_bh, *d_oh;
   CK(cudaMalloc(&d_ah, total * sizeof(cx)));
   CK(cudaMalloc(&d_bh, total * sizeof(cx)));
   CK(cudaMalloc(&d_oh, total * sizeof(cx)));
   const int PT = 256;
   const int pg = (int)((total + PT - 1) / PT);
   dx16_pack<cx><<<pg, PT>>>(d_a,     d_ah, N, total);
   dx16_pack<cx><<<pg, PT>>>(d_bspec, d_bh, N, total);

   const auto smem = FFT::shared_memory_size;
   const dim3 blk(FFT::block_dim.x, FFT::block_dim.y, 1);
   const int grid = (int)((batch / 2) / FFT::block_dim.y);
   CK(cudaFuncSetAttribute((dx16_conv_kernel<N, EPT, FPB, ARCH>),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

   dx16_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_ah, d_bh, d_oh);
   CK(cudaGetLastError());
   CK(cudaDeviceSynchronize());
   if(err_out) {
      dx16_unpack<cx><<<pg, PT>>>(d_oh, d_scratch, N, total);
      std::vector<float2> h(h_ref.size());
      CK(cudaMemcpy(h.data(), d_scratch, h.size() * sizeof(float2), cudaMemcpyDeviceToHost));
      *err_out = rel_l2(h, h_ref);
   }
   for(int i = 0; i < 5; i++)
      dx16_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_ah, d_bh, d_oh);
   CK(cudaDeviceSynchronize());
   tm.start();
   for(int i = 0; i < iters; i++)
      dx16_conv_kernel<N, EPT, FPB, ARCH><<<grid, blk, smem>>>(d_ah, d_bh, d_oh);
   const float ms = tm.stop() / (iters ? iters : 1);
   for(cx* q : {d_ah, d_bh, d_oh}) cudaFree(q);
   return ms;
}
#endif  // FUFFT_HAVE_CUFFTDX
