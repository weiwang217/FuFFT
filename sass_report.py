#!/usr/bin/env python3
"""Summarise `cuobjdump -sass` output per kernel.

Reads cuobjdump output on stdin (or from a file argument) and prints, for each
kernel whose mangled name matches the filter, a histogram of SASS opcodes plus
totals for the categories that matter here: float ALU, shared memory, global
memory, and barriers.

Usage:
    cuobjdump -sass bench_ptx | python3 sass_report.py fufft_conv
"""
import re
import sys
from collections import Counter

FILTER = sys.argv[1] if len(sys.argv) > 1 else 'fufft_'
STREAM = open(sys.argv[2]) if len(sys.argv) > 2 else sys.stdin

FUNC = re.compile(r'Function\s*:\s*(\S+)')
# a SASS line looks like:  /*0010*/   FFMA R5, R2, R3, -R4 ;
INSTR = re.compile(r'/\*[0-9a-fA-F]{4,}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9._]*)')

FLOAT_ALU = ('FADD', 'FMUL', 'FFMA', 'FSETP', 'FSEL', 'FMNMX', 'MUFU', 'FCHK')
SHARED    = ('LDS', 'STS', 'LDSM')
GLOBAL    = ('LDG', 'STG', 'LD', 'ST')
BARRIER   = ('BAR', 'BSSY', 'BSYNC', 'WARPSYNC')
MOVE      = ('MOV', 'IMAD.MOV', 'SHFL', 'PRMT', 'SEL')


def bucket(op, names):
    return any(op == n or op.startswith(n + '.') for n in names)


def main():
    kernels = {}
    cur = None
    for line in STREAM:
        m = FUNC.search(line)
        if m:
            name = m.group(1)
            cur = name if FILTER in name else None
            if cur:
                kernels.setdefault(cur, Counter())
            continue
        if cur:
            mi = INSTR.search(line)
            if mi:
                kernels[cur][mi.group(1)] += 1

    if not kernels:
        print(f"no kernel matching '{FILTER}' found in the input", file=sys.stderr)
        return 1

    for name, hist in kernels.items():
        total = sum(hist.values())
        f = sum(v for k, v in hist.items() if bucket(k, FLOAT_ALU))
        s = sum(v for k, v in hist.items() if bucket(k, SHARED))
        g = sum(v for k, v in hist.items() if bucket(k, GLOBAL))
        b = sum(v for k, v in hist.items() if bucket(k, BARRIER))
        mv = sum(v for k, v in hist.items() if bucket(k, MOVE))
        short = name if len(name) < 62 else name[:59] + '...'
        print(f"\n{short}")
        print(f"  total {total:5d} | float-ALU {f:5d} | shared {s:4d} | "
              f"global {g:4d} | barrier {b:3d} | move {mv:4d}")
        top = ', '.join(f"{k}:{v}" for k, v in hist.most_common(8))
        print(f"  top: {top}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
