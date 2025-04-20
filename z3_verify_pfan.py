#!/usr/bin/env python3
# ------------------------------------------------------------------------
# z3_verify_pfan.py
#
# Unified SMT verifier for the 5‑16‑5 Posit Flight‑Advisory Network (PFAN).
#
# MODES
#   default  : input‑noise robustness scan
#   --weights: minimal L1 tampering of W2
#   --mono   : advisory‑monotonicity check
#
# FLAGS (common)
#   --eps       start ε  (default 0.10)
#   --eps-step  grid step Δε   [omit to treat --eps as a single value]
#   --eps-max   grid ceiling   (default 2.0)
#   --timeout   seconds per SMT call (default 5 000)
#   --random    random x1 in valid range
#   --rho --theta --psi --vown --vint   override x1 fields
#
# EXTRA (mono)
#   --rho2 --theta2 --psi2   override x2 geometric fields
#
# OUTPUT
#   • Per‑ε progress lines  (eps, SAT/UNSAT/TIMEOUT, cumulative time)
#   • On success: "Advisory changed at ε = ..."
#   • If robustness mode finishes, one CSV row is appended to
#       pfan_noise_results.csv : dtype,eps_found/bound,SATorUNSAT,time(s)
#
# CHECKPOINTS
#   posit8_pfan.ckpt   posit16_pfan.ckpt   posit32_pfan.ckpt
#
# Author : Suleiman Sadiq
# ------------------------------------------------------------------------

import argparse, random, time, csv, os, sys
import numpy as np
import tensorflow as tf
import z3

np.set_printoptions(precision=3, suppress=True)

ADV  = ['COC', 'WeakLeft', 'StrongLeft', 'WeakRight', 'StrongRight']
DTYPE = {'posit32': tf.posit32, 'posit16': tf.posit16,
         'posit8': tf.posit8,   'float32': tf.float32, 'float16': tf.float16}

DEFAULT_X1 = [500., 10., 5., 120., 150.]
RNG = random.Random(123)
RHO_R, THETA_R, PSI_R, V_R = (0, 10000), (-180, 180), (-180, 180), (0, 300)
ms = lambda s: int(s * 1000)

# ----------------------------- CLI ----------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('dtype', choices=DTYPE.keys())
g = ap.add_mutually_exclusive_group()
g.add_argument('--weights', action='store_true')
g.add_argument('--mono',    action='store_true')
ap.add_argument('--eps',      type=float, default=0.10)
ap.add_argument('--eps-step', type=float, default=0.10)
ap.add_argument('--eps-max',  type=float, default=2.0)
ap.add_argument('--timeout',  type=int,   default=5000)
ap.add_argument('--rho', '--theta', '--psi', '--vown', '--vint',
                dest='vals', nargs=5, type=float)
ap.add_argument('--random', action='store_true')
ap.add_argument('--rho2', '--theta2', '--psi2',
                dest='vals2', nargs=3, type=float)
args = ap.parse_args()

mode = 'weights' if args.weights else ('mono' if args.mono else 'noise')
print('\nMode :', mode)
print('dtype:', args.dtype)

# ----------------------------- build x1 / x2 ------------------------------
rv = lambda rng: RNG.uniform(*rng)
x1 = [rv(RHO_R), rv(THETA_R), rv(PSI_R), rv(V_R), rv(V_R)] \
     if args.random else list(DEFAULT_X1)
if args.vals:
    for i, v in enumerate(args.vals):
        if v is not None:
            x1[i] = v
print('x1 =', x1)

if mode == 'mono':
    x2 = list(x1)
    if args.vals2:
        for i, v in enumerate(args.vals2):
            if v is not None:
                x2[i] = v
    print('x2 =', x2)

# ----------------------------- load checkpoint ----------------------------
ckpt = f'./{args.dtype}_pfan.ckpt'
tf.reset_default_graph()
with tf.Session() as s:
    sv = tf.train.import_meta_graph(ckpt + '.meta')
    sv.restore(s, ckpt)
    W1, b1 = s.run(['Variable:0', 'Variable_1:0'])
    W2, b2 = s.run(['Variable_2:0', 'Variable_3:0'])
print('Checkpoint:', ckpt)

# ----------------------------- baseline -----------------------------------
def forward_np(vec):
    h = np.maximum(0, vec.dot(W1) + b1)
    lg = h.dot(W2) + b2
    return lg, int(np.argmax(lg))

log1_raw, adv1 = forward_np(np.array(x1))
log1 = np.asarray(log1_raw, dtype=float)
print('\nBaseline logits :', np.round(log1, 3))
print('Baseline adv    :', ADV[adv1])

# ----------------------------- helpers ------------------------------------
relu = lambda e: z3.If(e > 0, e, 0)
def logits_sym(xs):
    h = [relu(b1[j] + sum(float(W1[i, j]) * xs[i] for i in range(5)))
         for j in range(16)]
    return [b2[k] + sum(float(W2[j, k]) * h[j] for j in range(16))
            for k in range(5)]

def z3f(v):
    if z3.is_int_value(v):      return float(v.as_long())
    if z3.is_rational_value(v): return v.numerator_as_long() / v.denominator_as_long()
    return float(str(v).replace('?', ''))

# ======================================================================
# 1) Input‑noise robustness
# ======================================================================
if mode == 'noise':
    d = z3.Reals('dr dt dp dvow dvint')
    xs = [x1[i] + d[i] for i in range(5)]
    logs = logits_sym(xs)
    better = [z3.And(*[logs[k] > logs[j] for j in range(5) if j != k])
              for k in range(5)]

    grid = [args.eps] if args.eps_step is None else \
           list(np.arange(args.eps, args.eps_max + 1e-9, args.eps_step))
    t0 = time.perf_counter()
    sat_found = False
    for eps in grid:
        s = z3.Optimize(); s.set(timeout=ms(args.timeout))
        for v in d: s.add(z3.Abs(v) <= eps)
        alt = z3.Int('alt')
        s.add(z3.And(alt >= 0, alt <= 4, alt != adv1))
        for k in range(5):
            s.add(z3.Implies(alt == k, better[k]))
        res = s.check()
        status = 'TIMEOUT' if s.reason_unknown() == 'timeout' else res
        print(f'eps={eps:.2f}  {status}  t={time.perf_counter() - t0:.1f}s')
        if res == z3.sat:
            print(f'Advisory changed at eps = {eps:.2f}')
            sat_found = True
            eps_star  = eps
            break
    if not sat_found:
        print(f'Input-noise UNSAT up to eps_max={args.eps_max}')
        eps_star = args.eps_max

    # ------------- CSV summary -------------
    row = [args.dtype, mode, f'{eps_star:.2f}',
           'SAT' if sat_found else 'UNSAT',
           f'{time.perf_counter() - t0:.1f}']
    csv_path = 'pfan_noise_results.csv'
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(['dtype', 'mode', 'eps', 'status', 'time_s'])
        w.writerow(row)
    print(f'CSV line appended to {csv_path}')
    sys.exit(0)

# ======================================================================
# 2) Weight‑tampering attack
#   (messages already plain ASCII so not changed)
# ======================================================================
if mode == 'weights':
    hid = np.maximum(0, np.dot(x1, W1)) + b1
    opt = z3.Optimize(); opt.set(timeout=ms(args.timeout))

    L1 = z3.Real('L1'); abs_terms = []; w = {}
    for i in range(8):
        for k in range(5):
            v  = z3.Real(f'w_{i}_{k}'); w[(i, k)] = v
            ad = z3.Real(f'ad_{i}_{k}')
            opt.add(ad >= v - float(W2[i, k]))
            opt.add(ad >= float(W2[i, k]) - v)
            abs_terms.append(ad)
    opt.add(L1 == sum(abs_terms))

    logs = [b2[k] + sum(w[(i, k)] * hid[i] for i in range(8)) for k in range(5)]
    alt = z3.Int('alt')
    opt.add(z3.And(alt >= 0, alt <= 4, alt != adv1))
    for k in range(5):
        opt.add(z3.Implies(alt == k,
                           z3.And(*[logs[k] > logs[j] for j in range(5) if j != k])))

    grid = [args.eps] if args.eps_step is None else \
           list(np.arange(args.eps, args.eps_max + 1e-9, args.eps_step))
    sat = False
    for eps in grid:
        tmp = z3.Optimize(); tmp.add(opt.assertions())
        tmp.add(L1 <= eps); tmp.set(timeout=ms(args.timeout))
        res = tmp.check()
        print(f'eps={eps:.2f}  {res}')
        if res == z3.sat:
            sat = True
            break
    if not sat:
        print('No SAT within grid - minimising globally ...')
        opt.minimize(L1); opt.check()

    mdl   = opt.model()
    L1min = z3f(mdl[L1])
    delta = [z3f(mdl[z3.Real(f'ad_{i}_{k}')]) for i in range(8) for k in range(5)]
    print('\nWeight SAT  L1* = %.6f  max=%.4f  mean=%.4f  rel=%.2f %%'
          % (L1min, max(delta), sum(delta) / len(delta),
             100. * L1min / np.abs(W2).sum()))
    sys.exit(0)

# ======================================================================
# 3) Advisory monotonicity
#   (messages already plain ASCII so not changed)
# ======================================================================
if mode == 'mono':
    d_rho, d_psi = z3.Reals('d_rho d_psi')
    rho1, psi1   = x1[0], x1[2]
    rho2 = rho1 + d_rho
    psi2 = psi1 + d_psi

    xs1 = [rho1, x1[1], psi1, x1[3], x1[4]]
    xs2 = [rho2, x1[1], psi2, x1[3], x1[4]]

    l1, l2 = logits_sym(xs1), logits_sym(xs2)
    opt = z3.Optimize(); opt.set(timeout=ms(args.timeout))

    def argmax_sym(lv, name):
        idx = z3.Int(name)
        for k in range(5):
            opt.add(z3.Implies(idx == k,
                               z3.And(*[lv[k] > lv[j] for j in range(5) if j != k])))
        opt.add(idx >= 0, idx <= 4)
        return idx

    a1 = argmax_sym(l1, 'a1')
    a2 = argmax_sym(l2, 'a2')

    opt.add(rho2 < rho1, z3.Abs(psi2) <= abs(psi1))
    rank = lambda idx: z3.If(idx == 0, 0,
                       z3.If(z3.Or(idx == 1, idx == 3), 1, 2))
    opt.add(rank(a2) < rank(a1))

    print('\nChecking monotonicity ...')
    res = opt.check()
    print('Result:', res)

    if res == z3.sat:
        m = opt.model()
        print('Counter example:',
              'rho2', z3f(m[rho2]),
              'psi2', z3f(m[psi2]),
              'adv',  m[a1].as_long(), '->', m[a2].as_long())
    else:
        print('No violation found (or timeout)')
