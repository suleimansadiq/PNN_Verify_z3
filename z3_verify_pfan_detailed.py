#!/usr/bin/env python3
# =========================================================================
#  PFAN‑1 VERIFICATION SCRIPT (ASCII‑only)
#  ------------------------------------------------------------------------
#  File         :  z3_verify_pfan.py
#  Author       :  Suleiman Sadiq
#  Revision     :  April 2025 demonstration version
#
#  Purpose      :  Performs SMT‑based safety / robustness checks on the
#                  “PFAN‑1” 5‑16‑5 network that emulates ACAS advisories.
#
#  ------------------------------------------------------------------------
#  NUMERIC TYPES / CHECKPOINTS
#  ------------------------------------------------------------------------
#    posit32   -> posit32_pfan.ckpt
#    posit16   -> posit16_pfan.ckpt
#    posit8    -> posit8_pfan.ckpt
#    float32   -> float32_pfan.ckpt
#    float16   -> float16_pfan.ckpt
#  (All .ckpt files must reside in the current working directory.)
#
#  ------------------------------------------------------------------------
#  MODES (choose exactly one)
#  ------------------------------------------------------------------------
#    default            : epsilon‑grid input‑noise robustness
#    --zones            : per‑zone counter‑example search
#    --weights          : weight‑tampering (minimise L1 shift on final layer)
#    --mono             : advisory‑monotonicity check
#
#  ------------------------------------------------------------------------
#  COMMON OPTIONAL FLAGS
#  ------------------------------------------------------------------------
#    --timeout  S       : wall‑clock budget in seconds (default 30)
#
#  ------------------------------------------------------------------------
#  NOISE MODE FLAGS
#  ------------------------------------------------------------------------
#    --rho   R          : nominal rho        (m)
#    --theta T          : nominal theta      (deg)
#    --psi   P          : nominal psi        (deg)
#    --v1    V1         : ownship speed      (m/s – may be negative)
#    --v2    V2         : intruder speed     (m/s – may be negative)
#    --eps      E       : starting epsilon   (default 0.10)
#    --eps-step D       : grid step          (default 0.10)
#    --eps-max  M       : ceiling            (default 2.0)
#
#  Output (noise mode)
#    • Per‑grid‑point line:  eps=E  (un)sat/unknown  time
#    • On first SAT:        "First flip: AAA -> BBB at eps = EF"
#    • Final CSV line       appended to robustness_log.csv
#
#  ------------------------------------------------------------------------
#  ZONES MODE FLAGS
#  ------------------------------------------------------------------------
#    --block   R        : axis‑aligned blocker radius (default 0.10)
#    --limit   N        : max witnesses per zone (omit = unlimited)
#    --timeout S        : budget per zone in seconds (default 30)
#    --first-only       : stop a zone after first witness
#
#  Output (zones mode)
#    • One "FAIL:" line per counter‑example (rho,theta,psi,v1,v2)
#    • Aligned ASCII summary table for all 5 zones
#    • zones_stats_<dtype>.csv written with the same data
#
#  ------------------------------------------------------------------------
#  QUICK EXAMPLES
#  ------------------------------------------------------------------------
#    1) 10‑second zone scan:
#         python3 z3_verify_pfan.py posit16 --zones --timeout 10
#
#    2) Deeper scan (block 50 m, 100‑s budget):
#         python3 z3_verify_pfan.py posit16 --zones --block 50 --timeout 100
#
#    3) Local robustness sweep around a Zone‑3 encounter:
#         python3 z3_verify_pfan.py posit16 --rho 1000 --theta 90 --psi 60 \
#                  --v1 0 --v2 0 --eps 1.0 --eps-step 1.0 --eps-max 50
#
#  ------------------------------------------------------------------------
#  DEPENDENCIES
#  ------------------------------------------------------------------------
#    Python 3.6+ , TensorFlow 1.x CPU,  z3‑solver 4.8+,  NumPy ≥ 1.16
# =========================================================================
import signal
import sys
import argparse
import time
import csv
from collections import Counter

import numpy as np
import tensorflow as tf
import z3

# ------------------------------------------------------------------------
# 0) graceful Ctrl‑C
# ------------------------------------------------------------------------
def _die(*_):
    print("\nInterrupted – exiting.", file=sys.stderr)
    sys.exit(130)

signal.signal(signal.SIGINT, _die)
signal.signal(signal.SIGTERM, _die)

# ------------------------------------------------------------------------
# 1) constants and helpers
# ------------------------------------------------------------------------
ADV = ["COC", "WeakLeft", "StrongLeft", "WeakRight", "StrongRight"]
DTYPE = {
    "posit32": tf.posit32,
    "posit16": tf.posit16,
    "posit8": tf.posit8,
    "float32": tf.float32,
    "float16": tf.float16,
}
EPS_GUARD = 1e-6
relu = lambda e: z3.If(e > 0, e, 0)


def z3f(val):
    if z3.is_int_value(val):
        return float(val.as_long())
    if z3.is_rational_value(val):
        return val.numerator_as_long() / val.denominator_as_long()
    return float(str(val))


def zone_pred(idx, rho, theta, psi):
    if idx == 0:  # COC
        return rho > 2000 + EPS_GUARD
    left, right = theta < 0, theta >= 0
    strong, weak = z3.Abs(psi) < 30, z3.Abs(psi) >= 30
    near = rho < 2000
    if idx == 1:
        return z3.And(near, left, weak)
    if idx == 2:
        return z3.And(near, left, strong)
    if idx == 3:
        return z3.And(near, right, weak)
    return z3.And(near, right, strong)


def zone_center(z):
    return [
        (6000, 0, 0),
        (1000, -90, 60),
        (500, -90, 0),
        (1000, 90, 60),
        (500, 90, 0),
    ][z]


def logits_sym(x, W1, b1, W2, b2):
    h = [
        relu(b1[j] + sum(float(W1[i, j]) * x[i] for i in range(5)))
        for j in range(16)
    ]
    return [
        b2[k] + sum(float(W2[j, k]) * h[j] for j in range(16)) for k in range(5)
    ]


def argmax_sym(lv, s, name):
    idx = z3.Int(name)
    s.add(idx >= 0, idx <= 4)
    for k in range(5):
        s.add(
            z3.Implies(
                idx == k, z3.And(*[lv[k] > lv[j] for j in range(5) if j != k])
            )
        )
    return idx


# ------------------------------------------------------------------------
# 2) command line
# ------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("dtype", choices=DTYPE.keys())
mx = ap.add_mutually_exclusive_group()
mx.add_argument("--weights", action="store_true")
mx.add_argument("--mono", action="store_true")
mx.add_argument("--zones", action="store_true")

ap.add_argument("--rho", type=float, default=500)
ap.add_argument("--theta", type=float, default=10)
ap.add_argument("--psi", type=float, default=5)
ap.add_argument("--v1", type=float, default=120)
ap.add_argument("--v2", type=float, default=150)

ap.add_argument("--eps", type=float, default=0.10)
ap.add_argument("--eps-step", type=float, default=0.10)
ap.add_argument("--eps-max", type=float, default=2.0)

ap.add_argument("--block", type=float, default=0.10)
ap.add_argument("--limit", type=int)
ap.add_argument("--timeout", type=float, default=30.0)
ap.add_argument("--first-only", action="store_true")
args = ap.parse_args()

mode = (
    "weights"
    if args.weights
    else "mono"
    if args.mono
    else "zones"
    if args.zones
    else "noise"
)

# ------------------------------------------------------------------------
# 3) load checkpoint
# ------------------------------------------------------------------------
ckpt = "./%s_pfan.ckpt" % args.dtype
tf.reset_default_graph()
with tf.Session() as sess:
    tf.train.import_meta_graph(ckpt + ".meta").restore(sess, ckpt)
    W1, b1, W2, b2 = sess.run(
        ["Variable:0", "Variable_1:0", "Variable_2:0", "Variable_3:0"]
    )
print("Checkpoint :", ckpt)

script_start = time.perf_counter()

# =========================================================================
# 5) ZONES MODE
# =========================================================================
if mode == "zones":
    rho, theta, psi, v1, v2 = z3.Reals("rho theta psi v1 v2")
    ENVELOPE = [
        z3.And(rho >= 0, rho <= 10000),
        z3.And(theta >= -180, theta <= 180),
        z3.And(psi >= -180, psi <= 180),
        z3.And(v1 >= 0, v1 <= 300),
        z3.And(v2 >= 0, v2 <= 300),
    ]

    stats = [
        {
            "Idx": z,
            "Zone": ADV[z],
            "Expected": ADV[z],
            "Viol": 0,
            "Wrong": Counter(),
            "First": None,
            "MaxDelta": 0.0,
            "GapSum": 0.0,
            "GapWorst": float("-inf"),
            "Status": "INIT",
            "Time": 0.0,
        }
        for z in range(5)
    ]

    for z in range(5):
        print("\nZone %d (%s zone) ..." % (z, ADV[z]))
        t0 = time.perf_counter()
        s = z3.Solver()
        s.set("random_seed", 1)
        s.add(*ENVELOPE, zone_pred(z, rho, theta, psi))
        adv = argmax_sym(logits_sym([rho, theta, psi, v1, v2], W1, b1, W2, b2), s, "adv")

        while True:
            left = args.timeout - (time.perf_counter() - t0)
            if left <= 0:
                if stats[z]["Status"] == "INIT":
                    stats[z]["Status"] = "TIMEOUT"
                break

            s.push()
            s.set("timeout", int(left * 1000))
            s.add(adv != z)
            res = s.check()

            if res == z3.sat:
                mdl = s.model()
                bad = mdl[adv].as_long()
                pt = (
                    z3f(mdl[rho]),
                    z3f(mdl[theta]),
                    z3f(mdl[psi]),
                    z3f(mdl[v1]),
                    z3f(mdl[v2]),
                )
                print(
                    "  FAIL: %s at rho=%.3f theta=%.3f psi=%.3f v1=%.3f v2=%.3f"
                    % ((ADV[bad],) + pt)
                )

                st = stats[z]
                st["Viol"] += 1
                st["Wrong"][ADV[bad]] += 1
                if st["First"] is None:
                    st["First"] = pt

                rc, tc, pc = zone_center(z)
                st["MaxDelta"] = max(
                    st["MaxDelta"],
                    max(abs(pt[0] - rc), abs(pt[1] - tc), abs(pt[2] - pc)),
                )

                hvec = np.maximum(0, np.dot(pt, W1) + b1)
                lvec = np.dot(hvec, W2) + b2
                gap = (lvec[bad] - lvec[z]) / (abs(lvec[z]) + 1e-9)
                st["GapSum"] += gap
                st["GapWorst"] = max(st["GapWorst"], gap)
                st["Status"] = "FAIL"

                if args.first_only or (args.limit and st["Viol"] >= args.limit):
                    s.pop()
                    break

                r, t, p_, v1v, v2v = pt
                s.pop()
                s.add(
                    z3.Or(
                        z3.Abs(rho - r) > args.block,
                        z3.Abs(theta - t) > args.block,
                        z3.Abs(psi - p_) > args.block,
                        z3.Abs(v1 - v1v) > args.block,
                        z3.Abs(v2 - v2v) > args.block,
                    )
                )
                continue

            s.pop()
            if res == z3.unsat and stats[z]["Status"] == "INIT":
                stats[z]["Status"] = "PASS"
            if res == z3.unknown and stats[z]["Status"] == "INIT":
                stats[z]["Status"] = "UNKNOWN"
            break

        stats[z]["Time"] = time.perf_counter() - t0
        if stats[z]["Status"] == "INIT":
            stats[z]["Status"] = "TIMEOUT"

    # summary
    header = "Idx Zone Exp Viol Wrong(max5) MaxDelta GapAvg GapWorst Time(s) Status"
    print("\n" + header)
    print("-" * len(header))
    for st in stats:
        wrong = "/".join("%s(%d)" % kv for kv in st["Wrong"].most_common(5)) or "-"
        gapavg = st["GapSum"] / st["Viol"] if st["Viol"] else 0.0
        print(
            "%3d %-4s %-8s %4d %-15s %8.1f %7.3f %8.3f %7.2f %-7s"
            % (
                st["Idx"],
                st["Zone"][:4],
                st["Expected"],
                st["Viol"],
                wrong,
                st["MaxDelta"],
                gapavg,
                st["GapWorst"],
                st["Time"],
                st["Status"],
            )
        )

    # CSV
    rows = []
    for st in stats:
        rows.append(
            list(
                map(
                    str,
                    [
                        st["Idx"],
                        st["Zone"],
                        st["Expected"],
                        st["Viol"],
                        "/".join("%s(%d)" % kv for kv in st["Wrong"].most_common())
                        or "-",
                        "%.3f" % st["MaxDelta"],
                        "%.6f"
                        % (
                            st["GapSum"] / st["Viol"]
                            if st["Viol"]
                            else 0.0
                        ),
                        "%.6f" % st["GapWorst"],
                        "-"
                        if st["First"] is None
                        else "(%.1f,%.1f,%.1f,%.1f,%.1f)" % st["First"],
                        st["Status"],
                        "%.3f" % st["Time"],
                    ],
                )
            )
        )
    with open("zones_stats_%s.csv" % args.dtype, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    sys.exit(0)

# =========================================================================
# 6) NOISE ROBUSTNESS GRID
# =========================================================================
if mode == "noise":
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    dr, dt, dp, dv1, dv2 = z3.Reals("dr dt dp dv1 dv2")
    xs = [X0[i] + d for i, d in enumerate([dr, dt, dp, dv1, dv2])]
    log_sym = logits_sym(xs, W1, b1, W2, b2)

    log0 = np.dot(np.maximum(0, np.dot(X0, W1) + b1), W2) + b2
    adv0 = int(np.argmax(log0))
    print("\nBaseline logits :", np.round(log0.astype(float), 3))
    print("Baseline adv    :", ADV[adv0])

    better = [
        z3.And(*[log_sym[k] > log_sym[j] for j in range(5) if j != k])
        for k in range(5)
    ]

    eps_first = eps_star = adv_flip = None
    for eps in np.arange(args.eps, args.eps_max + 1e-9, args.eps_step):
        t0 = time.perf_counter()
        s = z3.Optimize()
        s.set("timeout", int(args.timeout * 1000))
        for d in [dr, dt, dp, dv1, dv2]:
            s.add(z3.Abs(d) <= eps)
        alt = z3.Int("alt")
        s.add(z3.And(alt >= 0, alt <= 4, alt != adv0))
        for k in range(5):
            s.add(z3.Implies(alt == k, better[k]))
        res = s.check()
        eps_time = time.perf_counter() - t0
        total_time = time.perf_counter() - script_start
        print(
            "eps=%.3f  %-7s  (eps time=%.3fs, total=%.1fs)"
            % (eps, str(res), eps_time, total_time)
        )

        if res == z3.sat:
            eps_first = eps
            adv_flip = ADV[int(s.model()[alt].as_long())]
            # minimal L1 refinement
            dr2, dt2, dp2, dv12, dv22 = z3.Reals("dr2 dt2 dp2 dv12 dv22")
            xs2 = [
                X0[i] + d
                for i, d in enumerate([dr2, dt2, dp2, dv12, dv22])
            ]
            log2 = logits_sym(xs2, W1, b1, W2, b2)
            opt = z3.Optimize()
            opt.set("timeout", int(args.timeout * 1000))
            for d in [dr2, dt2, dp2, dv12, dv22]:
                opt.add(z3.Abs(d) <= eps)
            alt2 = z3.Int("alt2")
            opt.add(z3.And(alt2 >= 0, alt2 <= 4, alt2 != adv0))
            for k in range(5):
                opt.add(
                    z3.Implies(
                        alt2 == k,
                        z3.And(
                            *[log2[k] > log2[j] for j in range(5) if j != k]
                        ),
                    )
                )
            L1 = z3.Real("L1")
            opt.add(
                L1
                == z3.Abs(dr2)
                + z3.Abs(dt2)
                + z3.Abs(dp2)
                + z3.Abs(dv12)
                + z3.Abs(dv22)
            )
            opt.minimize(L1)
            eps_star = (
                z3f(opt.model()[L1]) if opt.check() == z3.sat else eps_first
            )
            print(
                "\nSAT  eps=%.3f  eps time=%.3fs  total=%.1fs  L1=%.6f"
                % (eps_first, eps_time, total_time, eps_star)
            )
            break

    if eps_first is None:
        eps_first = eps_star = "> %.2f" % args.eps_max
    robust5 = "Y" if (isinstance(eps_star, str) or eps_star > 5.0) else "N"
    run_time = time.perf_counter() - script_start

    csv_row = [
        "PFAN-" + args.dtype[-2:],
        "ALL",
        ADV[adv0],
        adv_flip,
        eps_first,
        eps_star,
        robust5,
        round(run_time, 3),
    ]
    csv_row_str = list(map(str, csv_row))
    print("\nCSV:", ",".join(csv_row_str))
    with open("robustness_log.csv", "a", newline="") as f:
        csv.writer(f).writerow(csv_row_str)
    sys.exit(0)

# =========================================================================
# 7) WEIGHT TAMPERING
# =========================================================================
if mode == "weights":
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    hid = np.maximum(0, np.dot(X0, W1) + b1)
    L1 = z3.Real("L1")
    opt = z3.Optimize()
    opt.set("timeout", int(args.timeout * 1000))

    abs_terms, nW = [], {}
    for i in range(16):
        for k in range(5):
            v = z3.Real(f"w{i}_{k}")
            nW[(i, k)] = v
            a = z3.Real(f"a{i}_{k}")
            opt.add(a >= v - float(W2[i, k]), a >= float(W2[i, k]) - v)
            abs_terms.append(a)
    opt.add(L1 == sum(abs_terms))

    log = [
        b2[k] + sum(nW[(i, k)] * hid[i] for i in range(16)) for k in range(5)
    ]
    alt = z3.Int("alt")
    opt.add(z3.And(alt >= 0, alt <= 4, alt != int(np.argmax(log))))
    for k in range(5):
        opt.add(
            z3.Implies(
                alt == k, z3.And(*[log[k] > log[j] for j in range(5) if j != k])
            )
        )
    opt.minimize(L1)

    t0 = time.perf_counter()
    res = opt.check()
    eps_time = time.perf_counter() - t0
    total_time = time.perf_counter() - script_start
    print("\nMinimising L1 ...")
    print("Z3 %-7s  (eps time=%.3fs, total=%.1fs)" % (str(res), eps_time, total_time))
    if res == z3.sat:
        print("L1* =", z3f(opt.model()[L1]))
    sys.exit(0)

# =========================================================================
# 8) MONOTONICITY
# =========================================================================
if mode == "mono":
    X0 = [args.rho, args.theta, args.psi, args.v1, args.v2]
    d_rho, d_psi = z3.Reals("d_rho d_psi")
    rho1, psi1 = X0[0], X0[2]
    rho2, psi2 = rho1 + d_rho, psi1 + d_psi
    xs1 = [rho1, X0[1], psi1, X0[3], X0[4]]
    xs2 = [rho2, X0[1], psi2, X0[3], X0[4]]
    l1 = logits_sym(xs1, W1, b1, W2, b2)
    l2 = logits_sym(xs2, W1, b1, W2, b2)

    s = z3.Solver()
    s.set("timeout", int(args.timeout * 1000))
    a1 = argmax_sym(l1, s, "a1")
    a2 = argmax_sym(l2, s, "a2")
    rank = lambda idx: z3.If(
        idx == 0, 0, z3.If(z3.Or(idx == 1, idx == 3), 1, 2)
    )
    s.add(rho2 < rho1, z3.Abs(psi2) <= abs(psi1), rank(a2) < rank(a1))

    t0 = time.perf_counter()
    res = s.check()
    eps_time = time.perf_counter() - t0
    total_time = time.perf_counter() - script_start
    print("\nChecking monotonicity ...")
    print("Z3 %-7s  (eps time=%.3fs, total=%.1fs)" % (str(res), eps_time, total_time))
    if res == z3.sat:
        m = s.model()
        print(
            "Violation at rho2 %.2f psi2 %.2f  adv %d -> %d"
            % (z3f(m[rho2]), z3f(m[psi2]), m[a1].as_long(), m[a2].as_long())
        )
    sys.exit(0)

