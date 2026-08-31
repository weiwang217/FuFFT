CUDA ?= /usr/local/cuda
SM   ?= 86
NVCC := $(CUDA)/bin/nvcc
CXX  ?= g++

NVFLAGS  ?= -O3 -std=c++14 -lineinfo --ptxas-options=-v
BOUNDS   ?= 3
DEFS     := -DFUFFT_MIN_BLOCKS=$(BOUNDS)
GENCODE  := -gencode=arch=compute_$(SM),code=sm_$(SM) -gencode=arch=compute_$(SM),code=compute_$(SM)
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra
ITERS    ?= 200

BACKENDS := cpp_naive cpp_opt ptx

# cuFFTDx (MathDx) is a separate download, not part of the CUDA toolkit.
#   make run-dx SM=90 CUFFTDX_DIR=/opt/nvidia/mathdx/24.08/include
DXB ?= 1
CUFFTDX_DIR ?=
ifneq ($(CUFFTDX_DIR),)
# cuFFTDx pulls in CUTLASS; the mathdx wheel bundles it at ../external/cutlass.
CUTLASS_DIR ?= $(wildcard $(CUFFTDX_DIR)/../external/cutlass/include)
DXFLAGS := -I$(CUFFTDX_DIR) $(if $(CUTLASS_DIR),-I$(CUTLASS_DIR),) \
           -DFUFFT_HAVE_CUFFTDX -DFUFFT_DX_ARCH=$(SM)0 --extended-lambda
else
DXFLAGS :=
endif

.PHONY: all cpu gpu run-dx check-cuda run-cpu run-gpu sass ncu occ counts audit clean

all: cpu

# ---- host-only: no GPU needed ----
cpu: cpu_fufft cpu_inplace cpu_mixed verify_kernel

cpu_fufft: cpu_fufft.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu_inplace: cpu_inplace.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

cpu_mixed: cpu_mixed.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

verify_kernel: verify_kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run-cpu: cpu audit
	./cpu_fufft
	@echo
	./cpu_inplace
	@echo
	./cpu_mixed
	@echo
	./verify_kernel

# static audit of the inline PTX: opcode arity, operand range, and the
# early-clobber hazard that '=f' (without &) allows
audit:
	@python3 audit_ptx.py
	@python3 audit_smem.py

# analytic instruction count -- the reference that `make ncu` measures against
counts:
	@python3 count_instructions.py

# ---- device targets ----
# Fail with something readable instead of "/bin/sh: nvcc: No such file".
check-cuda:
	@test -x "$(NVCC)" || { \
	  echo ""; \
	  echo "  nvcc not found at $(NVCC)"; \
	  echo ""; \
	  echo "  The GPU targets need CUDA and an NVIDIA device.  Point CUDA at your"; \
	  echo "  toolkit and set SM for the architecture, e.g."; \
	  echo ""; \
	  echo "      make run-gpu CUDA=/usr/local/cuda-12.6 SM=120"; \
	  echo ""; \
	  echo "  SM: 120 Blackwell (RTX 50xx), 89 Ada (RTX 40xx), 86 Ampere (RTX 30xx)."; \
	  echo "  Host-only checks need none of this:  make run-cpu"; \
	  echo ""; \
	  exit 1; }

gpu: check-cuda $(addprefix bench_,$(BACKENDS))

# stderr carries the ptxas -v register report; keep it so `sass` can show it,
# and preserve nvcc's exit status rather than losing it to a pipe
define build_backend
@$(NVCC) $(NVFLAGS) $(GENCODE) $(DEFS) -DFUFFT_BACKEND=$(2) -o $(1) bench_fufft.cu -lcufft \
   2> $(1).log || (cat $(1).log; rm -f $(1).log; false)
@grep -E 'Used [0-9]+ registers' $(1).log | head -4 | sed 's/^/  [$(1)] /' || true
	@if grep -qE '[1-9][0-9]* bytes (stack frame|spill)' $(1).log; then \
	   echo "  [$(1)] *** REGISTER SPILLS -- __launch_bounds__ may be too tight;"; \
	   echo "  [$(1)]     try  make ... FUFFT_MIN_BLOCKS=2  or  =0 to disable"; \
	   grep -E '[1-9][0-9]* bytes (stack frame|spill)' $(1).log | head -3 | sed 's/^/  [$(1)] /'; \
	 fi
endef

bench_cpp_naive: bench_fufft.cu fufft.cuh
	$(call build_backend,bench_cpp_naive,0)

bench_cpp_opt: bench_fufft.cu fufft.cuh
	$(call build_backend,bench_cpp_opt,1)

bench_ptx: bench_fufft.cu fufft.cuh
	$(call build_backend,bench_ptx,2)

bench_dx: bench_fufft.cu fufft.cuh cufftdx_conv.cuh | check-cuda
	@test -n "$(CUFFTDX_DIR)" || { \
	  echo ""; \
	  echo "  CUFFTDX_DIR is not set, so the cuFFTDx comparison cannot build."; \
	  echo ""; \
	  echo "  cuFFTDx ships in NVIDIA MathDx, separately from the CUDA toolkit:"; \
	  echo "    https://developer.nvidia.com/cufftdx-downloads"; \
	  echo "    tar xf nvidia-mathdx-*.tar.gz"; \
	  echo "    make run-dx SM=$(SM) CUFFTDX_DIR=\$$PWD/nvidia-mathdx-*/include"; \
	  echo ""; \
	  echo "  This is the comparison that decides whether the fused kernel has"; \
	  echo "  a reason to exist -- worth the download."; \
	  echo ""; exit 1; }
	@$(NVCC) $(subst -std=c++14,-std=c++17,$(NVFLAGS)) $(GENCODE) $(DEFS) $(DXFLAGS) \
	   -DFUFFT_BACKEND=$(DXB) -o $@ bench_fufft.cu -lcufft 2> $@.log || (cat $@.log; false)
	@grep -E "registers|spill" $@.log | head -8 || true

run-dx: bench_dx        ## ours vs cuFFTDx, the like-for-like incumbent
	@./bench_dx $(ITERS)

run-gpu: gpu
	@for b in $(BACKENDS); do \
	   echo ""; echo "##################### backend: $$b #####################"; \
	   ./bench_$$b $(ITERS); \
	done

# Instruction-level comparison.  If cpp_opt and ptx produce the same float-ALU
# count, ptxas already did everything the hand-written assembly does and the
# assembly is pure risk for zero gain.
sass: gpu
	@for b in $(BACKENDS); do \
	   echo ""; echo "======================= $$b ======================="; \
	   $(CUDA)/bin/cuobjdump -sass bench_$$b | python3 sass_report.py fufft_ ; \
	   grep -E 'Used [0-9]+ registers' bench_$$b.log 2>/dev/null \
	     | sed 's/^/  ptxas: /' || true ; \
	done

# Which resource actually caps occupancy, from the ptxas -v lines the build
# already captured.  This is the number that decides whether radix 16 is worth
# porting: it doubles data registers and halves the lane count.
occ: gpu
	@python3 occupancy.py --sm $(SM)

# Instruction counts vs cuFFT, measured with Nsight Compute.  cuFFT is closed
# source and ships thousands of kernel variants, so only a profile of the real
# run can tell you which one was dispatched and what it cost.
ncu: bench_cpp_opt
	@NCU=$(CUDA)/bin/ncu python3 ncu_compare.py ./bench_cpp_opt

clean:
	rm -f cpu_fufft cpu_inplace cpu_mixed verify_kernel bench_dx bench_dx.log \
	      $(addprefix bench_,$(BACKENDS)) \
	      $(addsuffix .log,$(addprefix bench_,$(BACKENDS)))
