#!/usr/bin/env python3
"""Shared-memory bank-conflict audit for the fuFFT passes.

An 8-byte shared access (LDS.64/STS.64) is serviced in two 16-lane phases; in
each phase 16 lanes x 8 B = 128 B, which is exactly the 32 banks.  Two distinct
float2 addresses whose first bank collides serialise.

The gather in fufft_pass is  A[SWZ(i*mh + j + k*(N/8))]  and the scatter is
B[SWZ(i*m + j + k*mh)], with j = lane%mh and i = lane/mh, so i*mh + j == lane.

Without a swizzle the stage-0 scatter has stride 8 float2 = 64 B, which lands a
whole phase on 2 banks.  This script reports the worst conflict degree over
every stage and every k, with and without the swizzle, and fails if the
configured swizzle leaves anything above 1-way.

Keep SWZ here in sync with fufft.cuh.
"""
import sys

SIZES = [(512, 3, 8), (4096, 4, 1)]     # (N, log8 N, transforms per block)


def swz(i):
    # matches SWZ in fufft.cuh
    return i ^ ((i >> 3) & 15) ^ ((i >> 4) & 8) ^ ((i >> 8) & 8)


def identity(i):
    return i


def worst(addr_of_lane, perm):
    """Max number of distinct float2 addresses sharing a first bank in a phase."""
    w = 1
    for phase in (range(0, 16), range(16, 32)):
        banks = {}
        for lane in phase:
            a = perm(addr_of_lane(lane))
            banks.setdefault((2 * a) % 32, set()).add(a)
        w = max(w, max(len(v) for v in banks.values()))
    return w


def scan_linear(N, perm):
    """The load/store loops: for(i=lane; i<N; i+=T) smem[SWZ(i)]."""
    T = N // 8
    return max(worst(lambda L, t=t: L + t * T, perm) for t in range(8))


def scan_inplace(N, log8n, perm):
    """fufft_conv_ip: A[b*8M + j + k*M], read and written at the same address.
    DIF strides run N/8 .. 1, DIT strides 1 .. N/8, so one sweep covers both."""
    w = 1
    per_stage = []
    for s in range(log8n):
        M = N // (8 ** (s + 1))
        sw = 1
        for k in range(8):
            sw = max(sw, worst(lambda L, k=k, M=M:
                               (L // M) * 8 * M + (L % M) + k * M, perm))
        per_stage.append((M, sw))
        w = max(w, sw)
    return w, per_stage


def scan_digitrev(N, log8n, perm):
    """What a plain forward transform would need on the way out of an in-place
    kernel: out[base+i] = A[SWZ(digitrev(i))].  Reported to justify why
    fufft_c2c stays on Stockham."""
    T = N // 8

    def dr(i):
        r = 0
        for _ in range(log8n):
            r = r * 8 + (i & 7)
            i >>= 3
        return r
    return max(worst(lambda L, t=t: dr(L + t * T), perm) for t in range(8))


def swz16(i):
    # matches SWZ16 in fufft.cuh -- the V=16 kernel's private layout
    return i ^ ((i >> 3) & 15) ^ ((i >> 7) & 15)


def strides_16(N):
    """radix-16 stride sequence: N = LEAD * 16^k, leading radix first."""
    m = 0
    while (1 << m) < N:
        m += 1
    out, M = [], N
    lead = 1 << (m % 4)
    if lead > 1:
        M //= lead
        out.append((lead, M))
    for _ in range(m // 4):
        M //= 16
        out.append((16, M))
    return out


def scan_r16(N, perm):
    """lane-pair radix-16: gather and scatter both at base + (8h+k)*M."""
    w = 1
    for R, M in strides_16(N):
        if R != 16:
            continue
        for k in range(8):
            w = max(w, worst(lambda L, k=k, M=M:
                             ((L >> 1) // M) * 16 * M + ((L >> 1) % M)
                             + (((L & 1) * 8 + k)) * M, perm))
    return w


def scan_v16(N, perm):
    """V=16: T = N/16 lanes, radix-R pass touches base + k*M, G=16/R groups."""
    T = N // 16
    w = 1
    for R, M in strides_16(N):
        G = 16 // R
        for g in range(G):
            for k in range(R):
                w = max(w, worst(lambda L, g=g, k=k, M=M, R=R, T=T:
                                 ((L + g * T) // M) * R * M + ((L + g * T) % M)
                                 + k * M, perm))
    for t in range(16):
        w = max(w, worst(lambda L, t=t, T=T: L + t * T, perm))
    return w


def scan(N, log8n, perm):
    g = s = 1
    m = 1
    per_stage = []
    for _ in range(log8n):
        m *= 8
        mh = m // 8
        sg = ss = 1
        for k in range(8):
            sg = max(sg, worst(lambda L, k=k: L + k * (N // 8), perm))
            ss = max(ss, worst(lambda L, k=k, m=m, mh=mh:
                               (L // mh) * m + (L % mh) + k * mh, perm))
        per_stage.append((m, mh, sg, ss))
        g, s = max(g, sg), max(s, ss)
    return g, s, per_stage


def bijective(perm, n):
    img = {perm(i) for i in range(n)}
    return len(img) == n and max(img) < n


def main():
    fail = 0
    print("bank-conflict audit (8-byte shared accesses, 16-lane phases)\n")

    for N, log8n, cpb in SIZES:
        if not bijective(swz, N):
            print(f"  N={N}: SWZ IS NOT A BIJECTION -- it would corrupt data")
            fail += 1
            continue

        g0, s0, _ = scan(N, log8n, identity)
        g1, s1, stages = scan(N, log8n, swz)
        print(f"  N={N:5d}  CPB={cpb}  (swizzle is a bijection: ok)")
        l0, l1 = scan_linear(N, identity), scan_linear(N, swz)
        print(f"    without swizzle : gather {g0}-way, scatter {s0}-way, linear {l0}-way")
        print(f"    with    swizzle : gather {g1}-way, scatter {s1}-way, linear {l1}-way")
        for m, mh, sg, ss in stages:
            flag = '' if max(sg, ss) == 1 else '   <-- conflict remains'
            print(f"      m={m:6d} mh={mh:5d}  gather {sg}-way  scatter {ss}-way{flag}")
        ip0, _ = scan_inplace(N, log8n, identity)
        ip1, ip_stages = scan_inplace(N, log8n, swz)
        print(f"    in-place conv   : without swizzle {ip0}-way, with swizzle {ip1}-way")
        for M, sw in ip_stages:
            flag = '' if sw == 1 else '   <-- conflict remains'
            print(f"      M={M:6d}  {sw}-way{flag}")
        w_r16 = scan_r16(N, swz)
        w_v16 = scan_v16(N, swz16)
        v16_bij = bijective(swz16, N)
        print(f"    r16 lane-pair   : {w_r16}-way with the shared swizzle")
        print(f"    V=16 (own SWZ16): {w_v16}-way "
              f"(swizzle {'is a bijection' if v16_bij else 'NOT BIJECTIVE -- CORRUPTS'})")
        if max(w_r16, w_v16) > 1 or not v16_bij:
            fail += 1

        dr1 = scan_digitrev(N, log8n, swz)
        print(f"    digit-rev readout: {dr1}-way even with the swizzle -- this is why")
        print(f"                       fufft_c2c stays on Stockham; a convolution")
        print(f"                       never needs it")

        if max(g1, s1, l1, ip1) > 1:
            fail += 1
        print()

    print("audit_smem: " + ("clean" if not fail else f"{fail} configurations still conflict"))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
