#!/usr/bin/env python3
# ======================================================================
#  z3_verify_pfan.py   --   PFAN‑1 SMT safety / robustness checker
#  (ASCII‑only edition, April 2025)
#
#  Numeric types  :  posit32 / posit16 / posit8 / float32 / float16
#
#  Modes
#    default            epsilon‑grid noise robustness
#    --weights          weight‑tampering  (min ||Delta W2||_1)
#    --mono             monotonicity check
#    --zones            per‑zone counter‑example search
#         --block R     axis‑aligned blocker radius   (default 0.10)
#         --limit N     cap witnesses per zone        (omit = inf)
#         --timeout S   budget per zone in seconds    (default 30)
#         --first-only  stop a zone after first CE
#
#  Output files
#     zones  ->  zones_stats_<dtype>.csv
#     noise  ->  robustness_log.csv   (append‑only)
# ======================================================================

import argparse, sys, time, csv
from collections import Counter
import numpy as np
import tensorflow as tf
import z3

# ----------------------------------------------------------------------
# 1) constants and helpers
# ----------------------------------------------------------------------
ADV = ['COC', 'WeakLeft', 'StrongLeft', 'WeakRight', 'StrongRight']
DTYPE = {
    'posit32': tf.posit32, 'posit16': tf.posit16, 'posit8': tf.posit8,
    'float32': tf.float32, 'float16': tf.float16
}
EPS_GUARD = 1e-6

relu = lambda e: z3.If(e > 0, e, 0)

def z3f(val):
    if z3.is_int_value(val):
        return float(val.as_long())
    if z3.is_rational_value(val):
        return val.numerator_as_long() / val.denominator_as_long()
    return float(str(val))

# ---------- zone predicate -------------------------------------------
def zone_pred(idx, rho, theta, psi):
    if idx == 0:                      # COC
        return rho > 2000 + EPS_GUARD
    left, right  = theta < 0, theta >= 0
    strong, weak = z3.Abs(psi) < 30, z3.Abs(psi) >= 30
    near         = rho < 2000
    if idx == 1: return z3.And(near, left,  weak)    # WeakLeft
    if idx == 2: return z3.And(near, left,  strong)  # StrongLeft
    if idx == 3: return z3.And(near, right, weak)    # WeakRight
    return           z3.And(near, right, strong)     # StrongRight

def zone_center(z):
    if z == 0: return (6000,   0,   0)
    if z == 1: return (1000, -90,  60)
    if z == 2: return ( 500, -90,   0)
    if z == 3: return (1000,  90,  60)
    return ( 500,  90,   0)

# ----------------------------------------------------------------------
# 2) command line
# ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('dtype', choices=DTYPE.keys())

mx = ap.add_mutually_exclusive_group()
mx.add_argument('--weights', action='store_true')
mx.add_argument('--mono',    action='store_true')
mx.add_argument('--zones',   action='store_true')

# point for noise mode
ap.add_argument('--rho',   type=float, default=500)
ap.add_argument('--theta', type=float, default=10)
ap.add_argument('--psi',   type=float, default=5)
ap.add_argument('--v1',    type=float, default=120)
ap.add_argument('--v2',    type=float, default=150)

# grid controls
ap.add_argument('--eps',      type=float, default=0.10)
ap.add_argument('--eps-step', type=float, default=0.10)
ap.add_argument('--eps-max',  type=float, default=2.0)

# zones extras
ap.add_argument('--block',    type=float, default=0.10)
ap.add_argument('--limit',    type=int)
ap.add_argument('--timeout',  type=float, default=30.0)
ap.add_argument('--first-only', action='store_true')

args  = ap.parse_args()
mode = ('weights' if args.weights else
        'mono'    if args.mono    else
        'zones'   if args.zones   else
        'noise')

# ----------------------------------------------------------------------
# 3) load checkpoint
# ----------------------------------------------------------------------
ckpt = './%s_pfan.ckpt' % args.dtype
tf.reset_default_graph()
with tf.Session() as sess:
    tf.train.import_meta_graph(ckpt + '.meta').restore(sess, ckpt)
    W1, b1, W2, b2 = sess.run(['Variable:0', 'Variable_1:0',
                               'Variable_2:0', 'Variable_3:0'])
print('Checkpoint :', ckpt)

# ----------------------------------------------------------------------
# 4) symbolic helpers
# ----------------------------------------------------------------------
def logits_sym(x):
    h = [relu(b1[j] + sum(float(W1[i, j]) * x[i] for i in range(5)))
         for j in range(16)]
    return [b2[k] + sum(float(W2[j, k]) * h[j] for j in range(16))
            for k in range(5)]

def argmax_sym(lv, s, name):
    idx = z3.Int(name)
    s.add(idx >= 0, idx <= 4)
    for k in range(5):
        s.add(z3.Implies(idx == k,
                         z3.And(*[lv[k] > lv[j] for j in range(5) if j != k])))
    return idx

# =========================================================================
# 5) ZONES MODE
# =========================================================================
if mode == 'zones':
    rho, theta, psi, v1, v2 = z3.Reals('rho theta psi v1 v2')
    ENVELOPE = [
        z3.And(rho   >= 0,    rho   <= 10000),
        z3.And(theta >= -180, theta <= 180),
        z3.And(psi   >= -180, psi   <= 180),
        z3.And(v1    >= 0,    v1    <= 300),
        z3.And(v2    >= 0,    v2    <= 300)
    ]

    stats = [{
        'Idx': z, 'Zone': ADV[z], 'Expected': ADV[z],
        'Violations': 0, 'WrongLabels': Counter(), 'First': None,
        'MaxDeltaCtr': 0.0, 'GapSum': 0.0, 'GapWorst': float('-inf'),
        'Status': 'INIT', 'Time': 0.0
    } for z in range(5)]

    for z in range(5):
        print('\nZone %d (%s zone) ...' % (z, ADV[z]))
        t0 = time.perf_counter()
        s  = z3.Solver(); s.set('random_seed', 1)
        s.add(*ENVELOPE, zone_pred(z, rho, theta, psi))
        adv = argmax_sym(logits_sym([rho, theta, psi, v1, v2]), s, 'adv')

        while True:
            left = args.timeout - (time.perf_counter() - t0)
            if left <= 0:
                if stats[z]['Status'] == 'INIT':
                    stats[z]['Status'] = 'TIMEOUT'
                break

            s.push(); s.set('timeout', int(left * 1000))
            s.add(adv != z)                    # look for wrong advisory
            res = s.check()

            if res == z3.sat:
                mdl = s.model()
                bad = mdl[adv].as_long()
                pt  = (z3f(mdl[rho]), z3f(mdl[theta]), z3f(mdl[psi]),
                       z3f(mdl[v1]),  z3f(mdl[v2]))
                print('  FAIL: %s at rho=%.3f theta=%.3f psi=%.3f v1=%.3f v2=%.3f'
                      % ((ADV[bad],) + pt))

                st = stats[z]
                st['Violations'] += 1
                st['WrongLabels'][ADV[bad]] += 1
                if st['First'] is None:
                    st['First'] = pt

                rc, tc, pc = zone_center(z)
                st['MaxDeltaCtr'] = max(
                    st['MaxDeltaCtr'],
                    max(abs(pt[0] - rc), abs(pt[1] - tc), abs(pt[2] - pc))
                )

                hvec = np.maximum(0, np.dot(pt, W1) + b1)
                lvec = np.dot(hvec, W2) + b2
                gap  = (lvec[bad] - lvec[z]) / (abs(lvec[z]) + 1e-9)
                st['GapSum']   += gap
                st['GapWorst']  = max(st['GapWorst'], gap)
                st['Status']    = 'FAIL'

                if args.first_only or (args.limit and st['Violations'] >= args.limit):
                    s.pop(); break

                r, t, p_, v1v, v2v = pt
                s.pop()
                s.add(z3.Or(
                    z3.Abs(rho   - r)   > args.block,
                    z3.Abs(theta - t)   > args.block,
                    z3.Abs(psi   - p_)  > args.block,
                    z3.Abs(v1    - v1v) > args.block,
                    z3.Abs(v2    - v2v) > args.block))
                continue

            s.pop()
            if res == z3.unsat and stats[z]['Status'] == 'INIT':
                stats[z]['Status'] = 'PASS'
            if res == z3.unknown and stats[z]['Status'] == 'INIT':
                stats[z]['Status'] = 'UNKNOWN'
            break

        stats[z]['Time'] = time.perf_counter() - t0
        if stats[z]['Status'] == 'INIT':
            stats[z]['Status'] = 'TIMEOUT'

    # final statistics table
    for st in stats:
        st['GapAvg']   = st['GapSum'] / st['Violations'] if st['Violations'] else 0.0
        st['GapWorst'] = 0.0 if st['GapWorst'] == float('-inf') else st['GapWorst']
        st['WrongLblStr'] = '/'.join('%s(%d)' % kv
                              for kv in st['WrongLabels'].most_common()) or '-'
        st['FirstStr'] = '-' if st['First'] is None else \
                         '(%.1f,%.1f,%.1f,%.1f,%.1f)' % st['First']

    header = ("Idx  Zone          Exp   Viol   WrongLabels(count)         "
              "MaxDeltaCtr  GapAvg  GapWorst  FirstCounterexample              "
              "Status  Time")
    print("\n" + header)
    print('-' * len(header))
    rowfmt = ("{Idx:<3} {Zone:<12} {Expected:<8} {Violations:<5} "
              "{WrongLblStr:<25} {MaxDeltaCtr:<11.1f} {GapAvg:<7.3f} "
              "{GapWorst:<8.3f} {FirstStr:<30} {Status:<7} {Time:.2f}")
    for st in stats:
        print(rowfmt.format(**st))
    print('-' * len(header))

    # CSV file
    csv_path = 'zones_stats_%s.csv' % args.dtype
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Idx', 'Zone', 'Expected', 'Violations', 'WrongLabels',
                    'MaxDeltaCtr', 'GapAvg', 'GapWorst',
                    'FirstCounterexample', 'Status', 'Time'])
        for st in stats:
            w.writerow([st['Idx'], st['Zone'], st['Expected'], st['Violations'],
                        st['WrongLblStr'], '%.3f' % st['MaxDeltaCtr'],
                        '%.6f' % st['GapAvg'], '%.6f' % st['GapWorst'],
                        st['FirstStr'], st['Status'], '%.2f' % st['Time']])
    print('CSV saved to', csv_path)
    sys.exit(0)

# =========================================================================
# 6) NOISE‑ROBUSTNESS GRID
# =========================================================================
if mode == 'noise':
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    dr, dt, dp, dv1, dv2 = z3.Reals('dr dt dp dv1 dv2')
    xs = [X0[i] + d for i, d in enumerate([dr, dt, dp, dv1, dv2])]
    log_sym = logits_sym(xs)

    log0 = np.dot(np.maximum(0, np.dot(X0, W1) + b1), W2) + b2
    adv0 = int(np.argmax(log0))
    print('\nBaseline logits :', np.round(log0.astype(float), 3))
    print('Baseline adv    :', ADV[adv0])

    better = [z3.And(*[log_sym[k] > log_sym[j] for j in range(5) if j != k])
              for k in range(5)]

    start_time = time.perf_counter()
    eps_first, eps_star, adv_flip = None, None, '--'
    for eps in np.arange(args.eps, args.eps_max + 1e-9, args.eps_step):
        tic = time.perf_counter()
        s = z3.Optimize(); s.set('timeout', int(args.timeout * 1000))
        for d in [dr, dt, dp, dv1, dv2]:
            s.add(z3.Abs(d) <= eps)
        alt = z3.Int('alt')
        s.add(z3.And(alt >= 0, alt <= 4, alt != adv0))
        for k in range(5):
            s.add(z3.Implies(alt == k, better[k]))
        res = s.check()
        print("eps=%5.2f  %-7s  t=%5.2fs" %
              (eps, str(res), time.perf_counter() - tic))
        if res == z3.sat:
            eps_first = eps
            adv_flip  = ADV[int(s.model()[alt].as_long())]

            # refine to minimal L1
            dr2, dt2, dp2, dv12, dv22 = z3.Reals('dr2 dt2 dp2 dv12 dv22')
            xs2 = [X0[i] + d for i, d in
                   enumerate([dr2, dt2, dp2, dv12, dv22])]
            log2 = logits_sym(xs2)
            opt  = z3.Optimize(); opt.set('timeout', int(args.timeout * 1000))
            for d in [dr2, dt2, dp2, dv12, dv22]:
                opt.add(z3.Abs(d) <= eps)
            alt2 = z3.Int('alt2')
            opt.add(z3.And(alt2 >= 0, alt2 <= 4, alt2 != adv0))
            for k in range(5):
                opt.add(z3.Implies(alt2 == k,
                                   z3.And(*[log2[k] > log2[j]
                                            for j in range(5) if j != k])))
            L1 = z3.Real('L1')
            opt.add(L1 == z3.Abs(dr2) + z3.Abs(dt2) + z3.Abs(dp2) +
                              z3.Abs(dv12) + z3.Abs(dv22))
            opt.minimize(L1)
            if opt.check() == z3.sat:
                eps_star = z3f(opt.model()[L1])
            else:
                eps_star = eps_first
            print('\nFirst flip: %s -> %s  at eps = %.2f'
                  % (ADV[adv0], adv_flip, eps_first))
            break

    if eps_first is None:
        eps_first = '> %.2f' % args.eps_max
        eps_star  = '> %.2f' % args.eps_max

    robust5 = ('Y' if (isinstance(eps_star, str) or eps_star > 5.0) else 'N')
    run_time = round(time.perf_counter() - start_time, 2)

    csv_row = ['PFAN-' + args.dtype[-2:],
               'ALL', ADV[adv0], adv_flip, eps_first, eps_star,
               robust5, run_time]
    print('\nCSV:', ','.join(map(str, csv_row)))
    with open('robustness_log.csv', 'a', newline='') as f:
        csv.writer(f).writerow(csv_row)
    sys.exit(0)

# =========================================================================
# 7) WEIGHT‑TAMPERING  (same as before)
# =========================================================================
if mode == 'weights':
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    hid = np.maximum(0, np.dot(X0, W1) + b1)
    L1 = z3.Real('L1')
    opt = z3.Optimize(); opt.set('timeout', int(args.timeout * 1000))

    abs_terms, nW = [], {}
    for i in range(16):
        for k in range(5):
            v = z3.Real('w%d_%d' % (i, k)); nW[(i, k)] = v
            a = z3.Real('a%d_%d' % (i, k))
            opt.add(a >= v - float(W2[i, k]),
                    a >= float(W2[i, k]) - v)
            abs_terms.append(a)
    opt.add(L1 == sum(abs_terms))

    log = [b2[k] + sum(nW[(i, k)] * hid[i] for i in range(16)) for k in range(5)]
    alt = z3.Int('alt')
    opt.add(z3.And(alt >= 0, alt <= 4, alt != int(np.argmax(log))))
    for k in range(5):
        opt.add(z3.Implies(alt == k,
                           z3.And(*[log[k] > log[j] for j in range(5) if j != k])))
    opt.minimize(L1)
    print('\nMinimising L1 ...')
    res = opt.check(); print('Z3:', res)
    if res == z3.sat:
        print('L1* =', z3f(opt.model()[L1]))
    sys.exit(0)

# =========================================================================
# 8) MONOTONICITY  (same as before)
# =========================================================================
if mode == 'mono':
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    d_rho, d_psi = z3.Reals('d_rho d_psi')
    rho1, psi1 = X0[0], X0[2]
    rho2, psi2 = rho1 + d_rho, psi1 + d_psi

    xs1 = [rho1, X0[1], psi1, X0[3], X0[4]]
    xs2 = [rho2, X0[1], psi2, X0[3], X0[4]]
    l1, l2 = logits_sym(xs1), logits_sym(xs2)

    s = z3.Solver(); s.set('timeout', int(args.timeout * 1000))
    a1 = argmax_sym(l1, s, 'a1')
    a2 = argmax_sym(l2, s, 'a2')
    rank = lambda idx: z3.If(idx == 0, 0,
                       z3.If(z3.Or(idx == 1, idx == 3), 1, 2))
    s.add(rho2 < rho1, z3.Abs(psi2) <= abs(psi1), rank(a2) < rank(a1))

    print('\nChecking monotonicity ...')
    res = s.check(); print('Z3:', res)
    if res == z3.sat:
        m = s.model()
        print('Violation at rho2 %.2f psi2 %.2f  adv %d -> %d'
              % (z3f(m[rho2]), z3f(m[psi2]),
                 m[a1].as_long(), m[a2].as_long()))
    sys.exit(0)
