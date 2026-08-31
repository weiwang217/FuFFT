#!/usr/bin/env python3
"""Analytic SASS instruction count for the fuFFT kernels.

This is the prediction that `make ncu` measures.  Counting by hand first means
a large disagreement with the profiler points at something real -- spills,
unexpected address arithmetic, a twiddle that did not constant-fold -- rather
than being absorbed as noise.

Costs are in SASS instructions, not flops:

  cadd / csub          2 FADD
  cmul                 2 FMUL + 2 FFMA                       = 4
  cmul_w1 / cmul_w3    naive: same as cmul = 4
                       opt:   2 FADD + 2 FMUL = 4  (same count, fewer FMUL)
  cmul_i               naive: 4 ; opt: 0, the sign folds into the consumer
  __sincosf            2 MUFU + ~4 FMA/FMUL of range reduction = 6,
                       but MUFU issues at ~1/8 the FMA rate, so it is also
                       reported weighted
"""
import sys

MUFU_WEIGHT = 8          # MUFU issue rate relative to FMA


def dft8_cost(backend):
    """(instructions, mufu) for one radix-8 butterfly."""
    # 12 cadd + 12 csub, each 2 FADD
    inst = 24 * 2
    if backend == 'naive':
        inst += 4          # cmul_w1
        inst += 4          # cmul_w3
        inst += 3 * 4      # three cmul_i
    else:                  # opt / ptx
        inst += 4          # cmul_w1 -> 2 FADD + 2 FMUL
        inst += 4          # cmul_w3
        inst += 0          # cmul_i is free
    return inst


def twiddle_cost(recurrence):
    """(instructions, mufu) spent generating and applying one pass's twiddles."""
    if recurrence:
        return 4 + 6 * 4 + 7 * 4, 2
    return 7 * 4 + 7 * 4, 7 * 2


def pass_cost(backend, recurrence):
    """(instructions, mufu_count, shared_ld, shared_st) for one fufft_pass."""
    inst = 0
    mufu = 0
    if recurrence:
        inst += 4          # range reduction around one __sincosf
        mufu += 2
        inst += 6 * 4      # w = cmul(w, w1) six times
        inst += 7 * 4      # u[k] = cmul(u[k], w) seven times
    else:
        inst += 7 * 4      # range reduction for seven __sincosf
        mufu += 7 * 2
        inst += 7 * 4      # u[k] = cmul(u[k], w)
    inst += dft8_cost(backend)
    return inst, mufu, 8, 8


def kernel_cost(N, log8n, backend, recurrence, conv, fused=False):
    passes = log8n * (2 if conv else 1)
    pi, pm, pl, ps = pass_cost(backend, recurrence)

    # Passes at stride 1 have j == 0, so every twiddle is W^0 = 1 and the whole
    # twiddle block is guarded out.  Stockham hits this at its first pass;
    # DIF/DIT hit it at the last forward and first inverse pass.
    tw_i, tw_m = twiddle_cost(recurrence)
    free = 2 if conv else 1

    inst = passes * pi - free * tw_i
    mufu = passes * pm - free * tw_m
    sld = passes * pl
    sst = passes * ps
    # linear load / store loops: 8 iterations each, per lane
    gld = 8
    gst = 8
    sld += 8               # store into smem after the global load
    if conv:
        gld += 8           # bspec
        inst += 8 * 4      # the pointwise cmul, wherever it happens
        if fused:
            # the multiply rides in registers between the last DIF and the first
            # DIT butterfly, so the scatter that ended the forward transform and
            # the gather that began the inverse both disappear
            sld -= 8
            sst -= 8
        else:
            sld += 8       # read spectrum out of smem
            sst += 8       # write it back
    sld += 8               # read out of smem before the global store
    return dict(inst=inst, mufu=mufu, shared_ld=sld, shared_st=sst,
                global_ld=gld, global_st=gst,
                weighted=inst + mufu * MUFU_WEIGHT)


def main():
    print("analytic per-lane SASS instruction count (one lane handles 8 points)\n")
    configs = [(512, 3), (4096, 4)]
    print("  a lane handles 8 points, so per-point = per-lane / 8.  ncu reports")
    print("  per-point numbers, so the 'inst/pt' column is what ncu_compare.py prints.\n")
    print(f"{'N':>6} {'kernel':>7} {'backend':>7} {'recur':>6} "
          f"{'inst':>6} {'inst/pt':>8} {'MUFU':>5} {'wtd':>6} {'wtd/pt':>7} "
          f"{'sLD/pt':>7} {'sST/pt':>7}")
    print('-' * 84)
    rows = {}
    for N, l8 in configs:
        for conv, fused, tag in ((False, False, 'c2c'), (True, False, 'conv'),
                                 (True, True, 'conv+fu')):
            for backend in ('naive', 'opt'):
                for rec in (True, False):
                    c = kernel_cost(N, l8, backend, rec, conv, fused)
                    rows[(N, conv, backend, rec, fused)] = c
                    print(f"{N:6d} {tag:>7} {backend:>7} "
                          f"{'yes' if rec else 'no':>6} "
                          f"{c['inst']:6d} {c['inst']/8.0:8.2f} {c['mufu']:5d} "
                          f"{c['weighted']:6d} {c['weighted']/8.0:7.2f} "
                          f"{c['shared_ld']/8.0:7.3f} {c['shared_st']/8.0:7.3f}")
        print()

    print("what the three backends actually differ by, per conv kernel:")
    for N, l8 in configs:
        a = rows[(N, True, 'naive', True, False)]['inst']
        b = rows[(N, True, 'opt', True, False)]['inst']
        print(f"  N={N:5d}: naive {a} -> opt {b} instructions "
              f"({100.0 * (a - b) / a:.1f}% fewer). "
              f"ptx implements the same specialisations, so it should match opt.")

    print("\nwhy the twiddle recurrence matters for the backend comparison:")
    for N, l8 in configs:
        no = rows[(N, True, 'opt', False, False)]
        yes = rows[(N, True, 'opt', True, False)]
        d = rows[(N, True, 'naive', True, False)]['inst'] - yes['inst']
        print(f"  N={N:5d}: weighted cost {no['weighted']} -> {yes['weighted']} "
              f"({no['weighted'] / yes['weighted']:.2f}x cheaper); the naive-vs-opt gap "
              f"of {d} instructions is {100.0 * d / no['weighted']:.1f}% of the old total "
              f"but {100.0 * d / yes['weighted']:.1f}% of the new one")

    print("\nto compare against cuFFT: ncu_compare.py divides")
    print("smsp__thread_inst_executed.sum by the number of transform points, giving")
    print("the same units as the inst/pt column above.  bench_fufft sizes both batches")
    print("to 8.4M points, so the numbers are comparable across N and against")
    print("whatever kernels cuFFT actually dispatches.")
    print("\nregister-resident boundary (conv+fu) vs the separate pointwise pass:")
    for N, l8 in configs:
        a = rows[(N, True, 'opt', True, False)]
        b = rows[(N, True, 'opt', True, True)]
        print(f"  N={N:5d}: LDS/pt {a['shared_ld']/8:.0f} -> {b['shared_ld']/8:.0f}, "
              f"STS/pt {a['shared_st']/8:.0f} -> {b['shared_st']/8:.0f}, "
              f"and two fewer barriers.  Instructions are unchanged: the multiply")
        print(f"          still happens, it just never leaves registers.")

    print("\nin-place vs Stockham -- the claim that instruction counts are unchanged:")
    print("  per pass both do 8 shared loads and 8 shared stores; Stockham reads A and")
    print("  writes B, in-place reads and writes the same eight slots.  Same butterfly,")
    print("  same twiddles, same pass count.  The only difference is the buffer size.")
    for N, l8 in configs:
        c = rows[(N, True, 'opt', True, False)]
        cpb = 8 if N == 512 else 1
        st = 2 * N * 8 * cpb
        ip = 1 * N * 8 * cpb
        print(f"  N={N:5d}: inst/pt {c['inst']/8.0:.1f} either way; "
              f"shared/block {st>>10} KB -> {ip>>10} KB")
    print("  so `make ncu` must show identical inst/pt, LDS/pt and STS/pt for the two")
    print("  kernels.  If it does not, one of them is not doing what this model says.")

    print("\nthe prediction to check on hardware:")
    for N, l8 in configs:
        c = rows[(N, True, 'opt', True, False)]
        print(f"  N={N:5d} fufft_conv, cpp_opt: expect ~{c['inst']/8.0:.1f} thread-inst/pt, "
              f"{c['shared_ld']/8.0:.2f} LDS/pt, {c['shared_st']/8.0:.2f} STS/pt, "
              f"0 bank conflicts")
    return 0


if __name__ == '__main__':
    sys.exit(main())
