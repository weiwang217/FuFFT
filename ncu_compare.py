#!/usr/bin/env python3
"""Instruction-count comparison against cuFFT, measured with Nsight Compute.

cuFFT is closed source and ships thousands of kernel variants, so a static
cuobjdump cannot tell you which one a given plan actually launches.  The only
apples-to-apples way is to profile the real run and read the counters off
whichever kernels were dispatched.

`bench_fufft <binary> 0` runs in profile mode: exactly one launch of each of
fufft_c2c, the fused pipeline kernels, the cuFFT forward/inverse and pointwise_mul,
per size.  That keeps ncu fast and makes every row unambiguous.

Counts are normalised per transform point so our kernels and cuFFT's are
directly comparable despite different block shapes and batch factors.

Usage:
    python3 ncu_compare.py                 # runs ncu itself
    python3 ncu_compare.py --csv out.csv   # parse an existing ncu --csv dump
"""
import csv
import io
import os
import subprocess
import sys

METRICS = [
    'smsp__inst_executed.sum',
    'smsp__thread_inst_executed.sum',
    'sm__sass_thread_inst_executed_op_fadd_pred_on.sum',
    'sm__sass_thread_inst_executed_op_fmul_pred_on.sum',
    'sm__sass_thread_inst_executed_op_ffma_pred_on.sum',
    'smsp__inst_executed_op_shared_ld.sum',
    'smsp__inst_executed_op_shared_st.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'dram__bytes.sum',
    'launch__registers_per_thread',
    'gpu__time_duration.sum',
]

SHORT = {
    'smsp__inst_executed.sum': 'warp_inst',
    'smsp__thread_inst_executed.sum': 'thread_inst',
    'sm__sass_thread_inst_executed_op_fadd_pred_on.sum': 'FADD',
    'sm__sass_thread_inst_executed_op_fmul_pred_on.sum': 'FMUL',
    'sm__sass_thread_inst_executed_op_ffma_pred_on.sum': 'FFMA',
    'smsp__inst_executed_op_shared_ld.sum': 'LDS',
    'smsp__inst_executed_op_shared_st.sum': 'STS',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum': 'conf_ld',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum': 'conf_st',
    'dram__bytes.sum': 'dram_B',
    'launch__registers_per_thread': 'regs',
    'gpu__time_duration.sum': 'ns',
}

# points processed per launch, from bench_fufft's run_size calls
POINTS = {512: 512 * 16384, 4096: 4096 * 2048}


def run_ncu(binary):
    ncu = os.environ.get('NCU', 'ncu')   # the Makefile points this at $(CUDA)/bin/ncu
    cmd = [ncu, '--csv', '--target-processes', 'all',
           '--metrics', ','.join(METRICS), binary, '0']
    print('  $ ' + ' '.join(cmd), file=sys.stderr)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit(f"'{ncu}' not found -- Nsight Compute ships with the CUDA toolkit; "
                 "set NCU=/path/to/ncu")
    if p.returncode != 0:
        # ncu reports its own errors (permissions, driver mismatch) on stdout
        diag = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith('==')]
        print('\n'.join(diag[-8:]) or p.stderr[-3000:], file=sys.stderr)
        sys.exit(f'ncu failed with status {p.returncode}')
    return p.stdout


def parse(text):
    """-> {kernel_name: {metric: float}}.  ncu prefixes junk before the header."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.startswith('"ID"') or l.startswith('ID,')), None)
    if start is None:
        sys.exit('no CSV header in the ncu output; was --csv passed?')
    rdr = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    out = {}
    for row in rdr:
        name = (row.get('Kernel Name') or '').strip()
        metric = (row.get('Metric Name') or '').strip()
        val = (row.get('Metric Value') or '').replace(',', '').strip()
        if not name or metric not in SHORT:
            continue
        try:
            out.setdefault(name, {})[metric] = float(val)
        except ValueError:
            pass
    return out


def classify(name):
    # kernel names as of the pipeline refactor; the old fufft_conv names are
    # kept as a fallback so archived CSVs still parse
    for tag, kind in (('fufft_conv_r2', 'ours-real2'),
                      ('fufft_conv2d', 'ours-2d'),
                      ('fufft_pipeline_fwd', 'ours-fwd'),
                      ('fufft_pipeline_apply', 'ours-apply'),
                      ('fufft_pipeline_v16f', 'ours-v16fused'),
                      ('fufft_pipeline_st', 'ours-stockham'),
                      ('fufft_pipeline_ip', 'ours-inplace'),
                      ('fufft_pipeline_fused', 'ours-fused'),
                      ('fufft_pipeline_r16', 'ours-r16pair'),
                      ('fufft_pipeline_v16', 'ours-v16'),
                      ('fufft_conv_ip', 'ours-inplace'),
                      ('fufft_conv', 'ours-conv'),
                      ('fufft_c2c', 'ours-c2c'),
                      ('pointwise_mul', 'cufft-chain-mul')):
        if tag in name:
            return kind
    return 'cufft'


def guess_size(name, m):
    # ncu may show the mangled name (Li512E) or the demangled one ((int)512)
    for n in (512, 4096):
        if f'Li{n}E' in name or f'(int){n}' in name or f'<{n},' in name:
            return n
    # cuFFT kernels do not encode our template args; fall back on dram bytes
    b = m.get('dram__bytes.sum', 0)
    best, bd = None, None
    for n, pts in POINTS.items():
        d = abs(b - 2 * pts * 8)
        if bd is None or d < bd:
            best, bd = n, d
    return best


def main():
    if '--csv' in sys.argv:
        text = open(sys.argv[sys.argv.index('--csv') + 1]).read()
    else:
        binary = next((a for a in sys.argv[1:] if not a.startswith('-')),
                      './bench_cpp_opt')
        text = run_ncu(binary)

    data = parse(text)
    if not data:
        sys.exit('no kernels matched; check the metric names for your ncu version')

    hdr = ['kind', 'N', 'thread_inst/pt', 'FADD/pt', 'FMUL/pt', 'FFMA/pt',
           'LDS/pt', 'STS/pt', 'conflicts', 'regs', 'us']
    print(f"\n{hdr[0]:<16}{hdr[1]:>6}{hdr[2]:>15}{hdr[3]:>9}{hdr[4]:>9}"
          f"{hdr[5]:>9}{hdr[6]:>8}{hdr[7]:>8}{hdr[8]:>11}{hdr[9]:>6}{hdr[10]:>9}")
    print('-' * 106)

    rows = []
    for name, m in sorted(data.items()):
        kind = classify(name)
        N = guess_size(name, m)
        pts = POINTS.get(N, 1)
        g = lambda k: m.get(k, 0.0)
        conf = g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum') + \
               g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum')
        rows.append((kind, N, name, g('smsp__thread_inst_executed.sum') / pts,
                     g('sm__sass_thread_inst_executed_op_fadd_pred_on.sum') / pts,
                     g('sm__sass_thread_inst_executed_op_fmul_pred_on.sum') / pts,
                     g('sm__sass_thread_inst_executed_op_ffma_pred_on.sum') / pts,
                     # shared ld/st are warp-level counters while
                     # thread_inst_executed is thread-level; x32 puts them in
                     # the same per-point units as the analytic count
                     g('smsp__inst_executed_op_shared_ld.sum') * 32.0 / pts,
                     g('smsp__inst_executed_op_shared_st.sum') * 32.0 / pts,
                     conf, g('launch__registers_per_thread'),
                     g('gpu__time_duration.sum') / 1000.0))

    for r in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{r[0]:<16}{r[1]:>6}{r[3]:>15.2f}{r[4]:>9.2f}{r[5]:>9.2f}"
              f"{r[6]:>9.2f}{r[7]:>8.3f}{r[8]:>8.3f}{r[9]:>11.0f}{r[10]:>6.0f}{r[11]:>9.1f}")

    print("\nkernel names in full:")
    for r in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"  {r[0]:<16} {r[2]}")

    print("\nnotes")
    print("  thread_inst/pt   total thread-instructions divided by transform points;")
    print("                   this is the number to compare against cuFFT")
    print("  conflicts        shared-memory bank conflicts; the swizzle in fufft.cuh")
    print("                   should hold ours near zero.  A large number here means")
    print("                   audit_smem.py's model does not match the hardware.")
    print("  LDS/pt, STS/pt   warp-level counters scaled by 32 so they match the")
    print("                   per-point units of count_instructions.py")
    print("  cuFFT rows are per-kernel: the convolution chain is the sum of the two")
    print("  cuFFT transform rows plus pointwise_mul.")
    print("\n  run  python3 count_instructions.py  for the analytic prediction;")
    print("  a large gap there points at spills or address arithmetic, not noise.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
