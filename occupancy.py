#!/usr/bin/env python3
"""Turn `ptxas -v` output into an occupancy answer.

Three things can cap resident blocks per SM -- shared memory, threads and
registers -- and only the tightest one matters.  It is easy to spend effort on
the wrong one: halving a shared buffer buys nothing if registers bind first,
which is exactly the risk for these kernels.

Reads the build logs the Makefile already writes (bench_*.log) and reports, per
kernel, which resource binds and what the occupancy is.  Then projects the
radix-16 trade: one fewer pass, but roughly double the data registers.

Usage:
    python3 occupancy.py                     # all bench_*.log in the cwd
    python3 occupancy.py bench_cpp_opt.log
    python3 occupancy.py --sm 120 --regs 52  # what-if, no log needed
"""
import glob
import re
import sys

# per-SM limits.  sm_120 = Blackwell consumer, sm_89 = Ada.
ARCH = {
    120: dict(smem_kb=128, max_threads=1536, regs=65536, name='sm_120 Blackwell'),
    89:  dict(smem_kb=100, max_threads=1536, regs=65536, name='sm_89 Ada'),
    86:  dict(smem_kb=100, max_threads=1536, regs=65536, name='sm_86 Ampere'),
}

ENTRY = re.compile(r"Compiling entry function '([^']+)'")
USED = re.compile(r'Used (\d+) registers')
SPILL = re.compile(r'(\d+) bytes spill stores')

# (threads, shared KB) each kernel is launched with, from bench_fufft.cu
LAUNCH = {
    'fufft_conv_r2':        (512, 32),   # two-for-one real conv
    'fufft_conv2d':         (512, 32),   # fused 2D tile conv
    'fufft_pipeline_fwd':   (512, 32),   # forward-only, split entry
    'fufft_pipeline_apply': (512, 32),   # mult + inverse only
    'fufft_pipeline_v16f':  (256, 32),   # V=16 + register boundary
    'fufft_pipeline_v16':   (256, 32),   # V=16: half the threads, own bounds
    'fufft_pipeline_r16':   (512, 32),
    'fufft_pipeline_ip':    (512, 32),
    'fufft_pipeline_fused': (512, 32),
    'fufft_pipeline_st':    (512, 64),
    'fufft_c2c':            (512, 64),
}


def short(mangled):
    for k in LAUNCH:
        if k in mangled:
            return k
    # ptxas emits MANGLED names: _Z16fufft_pipeline_ipILi4096E...
    # \w+ is greedy and swallows the template suffix, so restrict to the
    # lowercase identifier, which stops at the 'I' that begins template args.
    m = re.search(r'(fufft_[a-z0-9_]+)', mangled)
    return m.group(1) if m else mangled[:40]


def blocks(regs, threads, smem_kb, a):
    by_smem = a['smem_kb'] // smem_kb if smem_kb else 99
    by_thr = a['max_threads'] // threads
    by_reg = a['regs'] // (regs * threads) if regs else 99
    b = min(by_smem, by_thr, by_reg)
    which = min([(by_smem, 'shared'), (by_thr, 'threads'), (by_reg, 'registers')])[1]
    return b, which, (by_smem, by_thr, by_reg)


def report(name, regs, spills, arch):
    a = ARCH[arch]
    threads, smem_kb = LAUNCH.get(name, (512, 32))
    b, which, (bs, bt, br) = blocks(regs, threads, smem_kb, a)
    occ = 100.0 * b * threads / a['max_threads']
    flag = '  *** SPILLS' if spills else ''
    print(f"  {name:22} {regs:3d} regs  {smem_kb:3d} KB  {threads} thr -> "
          f"{b} blocks/SM  {occ:3.0f}%  bound by {which}{flag}")
    print(f"  {'':22} caps: shared {bs}, threads {bt}, registers {br}")
    return b, which, regs, threads, smem_kb


def radix16_projection(regs8, threads8, smem_kb, arch):
    """V=16 halves the lane count, doubles the data registers, drops one pass."""
    a = ARCH[arch]
    threads16 = threads8 // 2
    # 8 float2 of data = 16 regs; V=16 adds another 16
    regs16 = regs8 + 16
    b8, w8, _ = blocks(regs8, threads8, smem_kb, a)
    b16, w16, _ = blocks(regs16, threads16, smem_kb, a)
    occ8 = 100.0 * b8 * threads8 / a['max_threads']
    occ16 = 100.0 * b16 * threads16 / a['max_threads']

    print(f"\n  radix-16 projection (V=16: T halves, +16 data registers)")
    print(f"    V=8 : {regs8:3d} regs, {threads8} thr -> {b8} blocks, {occ8:3.0f}% "
          f"(bound by {w8})")
    print(f"    V=16: {regs16:3d} regs, {threads16} thr -> {b16} blocks, {occ16:3.0f}% "
          f"(bound by {w16})")
    passes = 0.75          # 4 passes -> 3 at N=1024/2048/4096
    shared_share = 0.20    # shared is about a fifth of the time
    gain = 1 - (1 - passes) * shared_share
    occ_term = occ8 / occ16 if occ16 else 99
    net = gain * occ_term
    print(f"    one fewer pass: shared -25%          -> time x{gain:.3f}")
    print(f"    occupancy {occ8:.0f}% -> {occ16:.0f}%")
    print()
    print(f"    A memory-bound kernel does not scale linearly with occupancy: what")
    print(f"    matters is whether the resident warps still cover the load latency.")
    print(f"    So the occupancy term is an upper bound on the harm, not an estimate.")
    print(f"      best case  (occupancy already sufficient): x{gain:.3f}  -> {1/gain:.2f}x faster")
    print(f"      worst case (scales linearly)             : x{net:.3f}  -> {net:.2f}x slower")
    if occ16 >= occ8:
        verdict = "DO IT -- occupancy does not drop"
    elif net < 1.0:
        verdict = "DO IT -- even the worst case wins"
    elif gain * 1.0 < 1.0 and net < 1.15:
        verdict = "MEASURE -- the range straddles 1.0, the model cannot decide"
    else:
        verdict = "PROBABLY NOT -- occupancy loss is large"
    print(f"      -> {verdict}")


def main():
    # NB: --sm/--regs take a VALUE, so a naive "not startswith('--')" filter
    # leaves the value in the positional list and it gets treated as a log
    # filename.  `make occ SM=86` hit exactly that.
    argv, args, skip = sys.argv[1:], [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a in ('--sm', '--regs'):
            skip = True
        elif not a.startswith('--'):
            args.append(a)
    arch = 120
    if '--sm' in sys.argv:
        arch = int(sys.argv[sys.argv.index('--sm') + 1])

    print(f"occupancy on {ARCH[arch]['name']}: "
          f"{ARCH[arch]['smem_kb']} KB shared, {ARCH[arch]['max_threads']} threads, "
          f"{ARCH[arch]['regs']} registers per SM\n")

    if '--regs' in sys.argv:
        r = int(sys.argv[sys.argv.index('--regs') + 1])
        b, w, rg, th, sk = report('what-if', r, 0, arch)
        radix16_projection(rg, th, sk, arch)
        return 0

    logs = args or sorted(glob.glob('bench_*.log'))
    if not logs:
        print("  no bench_*.log found -- run `make gpu` first, or pass --regs N")
        return 1

    best = None
    for lg in logs:
        try:
            text = open(lg).read()
        except IOError:
            continue
        print(f"{lg}:")
        cur = None
        seen = set()
        for line in text.splitlines():
            m = ENTRY.search(line)
            if m:
                cur = short(m.group(1))
                continue
            s = SPILL.search(line)
            sp = int(s.group(1)) if s else 0
            u = USED.search(line)
            if u and cur and cur not in seen:
                seen.add(cur)
                r = report(cur, int(u.group(1)), sp, arch)
                if cur == 'fufft_pipeline_ip':
                    best = r
        print()

    if best:
        _, _, regs, threads, smem_kb = best
        radix16_projection(regs, threads, smem_kb, arch)
    else:
        print("  fufft_pipeline_ip not found in the logs; pass --regs N to project")
    return 0


if __name__ == '__main__':
    sys.exit(main())
