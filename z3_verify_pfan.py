#!/usr/bin/env python3
# =========================================================================
#  z3_verify_pfan.py
#  -----------------------------------------------------------------------
#  SMT‑BASED SAFETY CHECKS FOR THE 5‑16‑5 “PFAN‑1” NETWORK
#  =========================================================================
#
#  WHAT IS PFAN‑1?
#  ----------------
#  A simple feed‑forward model (5 inputs – 16 ReLU – 5 outputs) trained to
#  emulate an aircraft collision‑avoidance logic.  Outputs map to:
#
#        0: COC          (Clear‑of‑Conflict)
#        1: WeakLeft     2: StrongLeft
#        3: WeakRight    4: StrongRight
#
#  Supported numerical variants: posit8 / posit16 / posit32 / float16 / float32.
#  We assume checkpoints named **<dtype>_pfan.ckpt** live in the cwd.
#
#  ------------------------------------------------------------------------
#  THREE VERIFICATION MODES
#  ------------------------------------------------------------------------
#  1)  “noise”   (default)  • **Input‑noise robustness**
#                               – All five inputs become symbolic.
#                               – Each ε‑slice: |Δfeature| ≤ ε (ℓ∞ box).
#                               – Reports first ε that flips the advisory,
#                                 then minimises ℓ₁ inside that slice.
#
#  2)  --weights           • **Weight‑tampering attack**
#                               – Only the final layer W₂ is symbolic.
#                               – Objective: smallest ℓ₁ shift to misclassify.
#
#  3)  --mono              • **Advisory‑monotonicity check**
#                               – Creates a second input that is
#                                 geometrically *more dangerous*:
#                                     ρ₂ < ρ₁  and  |ψ₂| ≤ |ψ₁|
#                               – SAT ⇨ network gave a *weaker* command.
#
#  ------------------------------------------------------------------------
#  BASIC USAGE
#  ------------------------------------------------------------------------
#    python3 z3_verify_pfan.py posit16
#
#  ROBUSTNESS GRID CONTROL
#    --eps        starting ε  (default 0.10)
#    --eps-step   step Δε     (omit to test exactly --eps once)
#    --eps-max    ceiling     (default 2.0)
#
#    example:  python3 z3_verify_pfan.py posit16 --eps 0.1 --eps-step 0.1 --eps-max 50
#
#  WEIGHT‑TAMPERING
#    python3 z3_verify_pfan.py posit8  --weights
#
#  MONOTONICITY (override ρ₂ or ψ₂ if desired)
#    python3 z3_verify_pfan.py posit32 --mono --rho2 400 --psi2 0
#
#  ------------------------------------------------------------------------
#  OUTPUT CHEAT‑SHEET
#  ------------------------------------------------------------------------
#   • Robustness grid lines:   eps=Δ  SAT/UNSAT/TIMEOUT  t=secs
#   • On first SAT:            new logits   new advisory   eps_first  eps_star
#   • Final CSV line (print‑only, not saved):
#         Model,Input Var,GT Advisory,ADV Advisory,eps_first,eps_star,R_0.05
#
#   • Weight mode prints:  L1*, max|ΔW|, mean|ΔW|, relative %
#   • Mono mode prints:    SAT + counter‑example  or  UNSAT
#
#  Author : Suleiman Sadiq (all‑ASCII version, April 2025)
# =========================================================================

import argparse, time, sys
import numpy as np, tensorflow as tf, z3

ADV = ['COC', 'WeakLeft', 'StrongLeft', 'WeakRight', 'StrongRight']
DTYPE = {'posit32': tf.posit32, 'posit16': tf.posit16,
         'posit8': tf.posit8,   'float32': tf.float32, 'float16': tf.float16}
X1 = [500., 10., 5., 120., 150.]  # reference point
ms = lambda s: int(s * 1000)

# ---------------- CLI ----------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('dtype', choices=DTYPE.keys())
g = ap.add_mutually_exclusive_group()
g.add_argument('--weights', action='store_true')
g.add_argument('--mono',    action='store_true')
ap.add_argument('--eps',      type=float, default=0.10)
ap.add_argument('--eps-step', type=float, default=0.10,
                help='omit to use a single --eps value')
ap.add_argument('--eps-max',  type=float, default=2.0)
ap.add_argument('--timeout',  type=int,   default=5000)
args = ap.parse_args()
mode = 'weights' if args.weights else ('mono' if args.mono else 'noise')

print('\nMode :', mode, '\ndtype:', args.dtype)

# ---------------- load checkpoint ---------------------------------------
ckpt = f'./{args.dtype}_pfan.ckpt'
tf.reset_default_graph()
with tf.Session() as s:
    saver = tf.train.import_meta_graph(ckpt + '.meta')
    saver.restore(s, ckpt)
    W1, b1 = s.run(['Variable:0', 'Variable_1:0'])
    W2, b2 = s.run(['Variable_2:0', 'Variable_3:0'])
print('Checkpoint:', ckpt)

# ---------------- baseline logits ---------------------------------------
h = np.maximum(0, np.dot(X1, W1) + b1)
log1 = np.dot(h, W2) + b2
adv1 = int(np.argmax(log1))
print('\nBaseline logits :', np.round(np.asarray(log1, dtype=float), 3))
print('Baseline adv    :', ADV[adv1])

# ---------------- helper funcs ------------------------------------------
relu = lambda e: z3.If(e > 0, e, 0)
def logits_sym(x):
    h = [relu(b1[j] + sum(float(W1[i, j]) * x[i] for i in range(5)))
         for j in range(16)]
    return [b2[k] + sum(float(W2[j, k]) * h[j] for j in range(16))
            for k in range(5)]

def z3f(v):
    if z3.is_int_value(v):      return float(v.as_long())
    if z3.is_rational_value(v): return v.numerator_as_long() / v.denominator_as_long()
    return float(str(v))

# =======================================================================
# 1) INPUT‑NOISE ROBUSTNESS  (default)
# =======================================================================
if mode == 'noise':
    dr, dt, dp, dv1, dv2 = z3.Reals('dr dt dp dv1 dv2')
    xs   = [X1[i] + d for i, d in enumerate([dr, dt, dp, dv1, dv2])]
    log  = logits_sym(xs)
    better = [z3.And(*[log[k] > log[j] for j in range(5) if j != k])
              for k in range(5)]

    grid = ([args.eps] if args.eps_step is None else
            list(np.arange(args.eps, args.eps_max + 1e-9, args.eps_step)))

    t0 = time.perf_counter()
    sat, eps_first, eps_star, adv_adv = False, None, None, '--'

    for eps in grid:
        s = z3.Optimize(); s.set(timeout=ms(args.timeout))
        for d in [dr, dt, dp, dv1, dv2]:
            s.add(z3.Abs(d) <= eps)
        alt = z3.Int('alt')
        s.add(z3.And(alt >= 0, alt <= 4, alt != adv1))
        for k in range(5):
            s.add(z3.Implies(alt == k, better[k]))

        res = s.check()
        status = 'TIMEOUT' if s.reason_unknown() == 'timeout' else res
        print(f'eps={eps:.2f}  {status}  t={time.perf_counter()-t0:.1f}s')

        if res == z3.sat:
            sat, eps_first = True, eps
            # refine to minimal L1 in this slice
            eps_L1 = z3.Real('L1')
            s.add(eps_L1 == z3.Abs(dr)+z3.Abs(dt)+z3.Abs(dp)+z3.Abs(dv1)+z3.Abs(dv2))
            s.minimize(eps_L1); s.check()
            mdl = s.model()
            eps_star = z3f(mdl[eps_L1])
            adv_adv  = ADV[int(mdl[alt].as_long())]

            log_adv = [z3f(mdl.eval(lv)) for lv in log]
            print('\nNew logits :', np.round(np.asarray(log_adv, dtype=float), 3))
            print('Adv advisory :', adv_adv)
            print(f'Advisory changed at eps = {eps_first:.2f}')
            break

    if not sat:
        eps_first = f'> {args.eps_max}'
        eps_star  = f'> {args.eps_max}'

    robust005 = 'Y' if (not sat or (isinstance(eps_star, float) and eps_star > 0.05)) else 'N'

    # -------- CSV summary line -----------------------------------------
    csv_line = [f'PFAN-{args.dtype[-2:]}', 'ALL', ADV[adv1],
                adv_adv, eps_first, eps_star, robust005]
    print('\nCSV:', ','.join(map(str, csv_line)))
    sys.exit(0)

# =======================================================================
# 2) WEIGHT‑TAMPERING ATTACK  (--weights)
# =======================================================================
if mode == 'weights':
    hid = np.maximum(0, np.dot(X1, W1) + b1)
    L1 = z3.Real('L1'); abs_terms = []; w = {}
    opt = z3.Optimize(); opt.set(timeout=ms(args.timeout))

    for i in range(8):
        for k in range(5):
            v  = z3.Real(f'w_{i}_{k}'); w[(i,k)] = v
            ad = z3.Real(f'ad_{i}_{k}')
            opt.add(ad >= v - float(W2[i,k]))
            opt.add(ad >= float(W2[i,k]) - v)
            abs_terms.append(ad)
    opt.add(L1 == sum(abs_terms))

    log = [b2[k] + sum(w[(i,k)] * hid[i] for i in range(8)) for k in range(5)]
    alt = z3.Int('alt')
    opt.add(z3.And(alt >= 0, alt <= 4, alt != adv1))
    for k in range(5):
        opt.add(z3.Implies(alt == k,
                           z3.And(*[log[k] > log[j] for j in range(5) if j != k])))

    print('\nMinimising L1 on W2 …')
    opt.minimize(L1); res = opt.check()
    print('Z3:', res)
    if res == z3.sat:
        mdl   = opt.model()
        L1min = z3f(mdl[L1])
        delta = [z3f(mdl[z3.Real(f'ad_{i}_{k}')]) for i in range(8) for k in range(5)]
        print('L1* = %.6f  max=%.4f  mean=%.4f  rel=%.2f %%'
              % (L1min, max(delta), sum(delta)/len(delta),
                 100.*L1min/np.abs(W2).sum()))
    sys.exit(0)

# =======================================================================
# 3) ADVISORY MONOTONICITY  (--mono)
# =======================================================================
if mode == 'mono':
    d_rho, d_psi = z3.Reals('d_rho d_psi')
    rho1, psi1   = X1[0], X1[2]
    rho2, psi2   = rho1 + d_rho, psi1 + d_psi

    xs1 = [rho1, X1[1], psi1, X1[3], X1[4]]
    xs2 = [rho2, X1[1], psi2, X1[3], X1[4]]
    l1, l2 = logits_sym(xs1), logits_sym(xs2)

    opt = z3.Optimize(); opt.set(timeout=ms(args.timeout))
    def argmax_sym(lv, name):
        idx = z3.Int(name)
        opt.add(idx >= 0, idx <= 4)
        for k in range(5):
            opt.add(z3.Implies(idx == k,
                               z3.And(*[lv[k] > lv[j] for j in range(5) if j != k])))
        return idx

    a1 = argmax_sym(l1, 'a1')
    a2 = argmax_sym(l2, 'a2')

    opt.add(rho2 < rho1, z3.Abs(psi2) <= abs(psi1))
    rank = lambda idx: z3.If(idx == 0, 0,
                       z3.If(z3.Or(idx == 1, idx == 3), 1, 2))
    opt.add(rank(a2) < rank(a1))

    print('\nChecking monotonicity …')
    res = opt.check()
    print('Z3:', res)
    if res == z3.sat:
        m = opt.model()
        print('Violation: rho2', z3f(m[rho2]),
              'psi2', z3f(m[psi2]),
              'adv',  m[a1].as_long(), '->', m[a2].as_long())
