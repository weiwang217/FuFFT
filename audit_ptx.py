#!/usr/bin/env python3
"""Static audit of the inline PTX in fufft.cuh.

Checks three things nvcc will not tell you about:

  1. opcode validity and operand arity against the PTX ISA
  2. %N operand indices in range
  3. the early-clobber hazard -- if an asm block writes an output before the
     last read of an input, and the output constraint lacks '&', the register
     allocator is permitted to place that output in an input's register and
     silently miscompile the block

Exit status is non-zero if anything fails.
"""
import re
import sys

ARITY = {
    'add.f32': 3, 'sub.f32': 3, 'mul.f32': 3,
    'neg.f32': 2, 'mov.f32': 2, 'fma.rn.f32': 4,
    # BFP I/O helpers (bfp_quant2 / bfp_dequant2)
    'cvt.rn.f32.s16': 2, 'cvt.rni.s16.f32': 2,
    'mul.f32': 3, 'mov.b32': 2,
}

SRC = sys.argv[1] if len(sys.argv) > 1 else 'fufft.cuh'


def blocks(src):
    for m in re.finditer(r'asm\s*\((.*?)\)\s*;', src, re.S):
        yield m.group(1)


def parse(block):
    head = block.split(':')[0]
    body = "".join(re.findall(r'"([^"]*)"', head))
    body = body.replace('\\n', '\n').replace('\\t', ' ')
    stmts = [s.strip() for s in re.split(r'[;\n]', body)
             if s.strip() not in ('', '{', '}') and not s.strip().startswith('.reg')]
    # count every register constraint kind, not just f -- the BFP helpers
    # introduced "r" (b32) operands and the f-only count mis-numbered them
    outs = len(re.findall(r'"=&?[frld]"', block))
    ins = len(re.findall(r'(?<![=&])"[frld]"', block))
    early = '=&f' in block
    return stmts, outs, ins, early


def main():
    src = open(SRC).read()
    nblk = ninstr = nfail = 0

    for bi, blk in enumerate(blocks(src)):
        stmts, outs, ins, early = parse(blk)
        nblk += 1
        first_out, last_in = None, -1

        for idx, st in enumerate(stmts):
            op = st.split()[0]
            # a {lo, hi} vector pack/unpack is ONE operand; make brace
            # groups atomic before splitting on commas
            rest = st[len(op):]
            rest = re.sub(r'\{[^}]*\}', 'BRACEGROUP', rest)
            ops = [o.strip() for o in rest.split(',') if o.strip()]
            ninstr += 1

            if op not in ARITY:
                print(f"  blk{bi}: unknown opcode '{op}'")
                nfail += 1
                continue
            if len(ops) != ARITY[op]:
                print(f"  blk{bi}: {op} has {len(ops)} operands, ISA wants {ARITY[op]}")
                nfail += 1

            m = re.match(r'%(\d+)$', ops[0])
            if m and int(m.group(1)) < outs and first_out is None:
                first_out = idx
            for s in ops[1:]:
                m2 = re.match(r'%(\d+)$', s)
                if m2:
                    if int(m2.group(1)) >= outs + ins:
                        print(f"  blk{bi}: operand %{m2.group(1)} out of range "
                              f"({outs} out + {ins} in)")
                        nfail += 1
                    elif int(m2.group(1)) >= outs:
                        last_in = idx

        if first_out is not None and last_in > first_out and not early:
            print(f"  blk{bi}: EARLY-CLOBBER HAZARD -- writes an output at stmt "
                  f"{first_out} but still reads an input at stmt {last_in}, "
                  f"and the constraint is '=f' not '=&f'")
            nfail += 1

    print(f"audit_ptx: {nblk} asm blocks, {ninstr} instructions, {nfail} problems")
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
