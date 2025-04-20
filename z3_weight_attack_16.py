#!/usr/bin/env python3
# WEIGHT_ATTACK_16.PY  –  ASCII‑only runtime strings
#
# Weight‑only Z3 attack on the posit‑16 ultra‑tiny MNIST MLP.
# We tamper W2 (8×10) only; W1, b1, b2 and the input image stay fixed.
#
# ε‑grid logic
#   • default grid: 0.10, 0.20, … , 2.00
#   • --eps E      : start at E, then E+step, … up to --eps-max (default 2.0)
#   • prints SAT / UNSAT / TIMEOUT for each slice
#   • after the first SAT the script refines the solution by *minimising*
#     Σ|ΔW2| under the same ε – gives the exact robustness margin.
#   • if every slice fails, it minimises with no upper bound (true δ★).
#
# Final line (CSV‑ready)
#   model,base,flip,delta_star,max_dw,mean_dw,rel_percent,time_s
#
# Flags
#   --idx N        attack test image N
#   --label D      first test image whose label is D (default 9)
#   --eps E        starting budget
#   --eps-step S   grid step (default 0.1)
#   --eps-max  M   grid upper bound (default 2.0)
#   --timeout T    Z3 timeout per slice in seconds (default 5000)

import argparse, time, numpy as np, tensorflow as tf
from z3 import Optimize, Real, Or, sat

np.set_printoptions(precision=3, suppress=True)

# --------------------------------------------------------------------------
def z3_to_float(val):
    """Convert a Z3 numeric to Python float."""
    s = str(val).split('+')[0]
    if '/' in s:
        p, q = s.split('/')
        return float(p) / float(q)
    return float(s.replace('?', ''))

def ascii_digit(img):
    """4‑level ASCII art of a 28×28 image in [‑1,1]."""
    pal = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    return [''.join(next(ch for ch, t in pal if img[r, c] < t) for c in range(28))
            for r in range(28)]

def ultratiny(x):
    """Definition of the tiny posit‑16 MLP."""
    W1 = tf.get_variable('Variable',   [784, 8],  tf.posit16)
    b1 = tf.get_variable('Variable_1', [8],       tf.posit16)
    W2 = tf.get_variable('Variable_2', [8, 10],   tf.posit16)
    b2 = tf.get_variable('Variable_3', [10],      tf.posit16)
    h   = tf.nn.relu(tf.matmul(tf.reshape(x, [-1, 784]), W1) + b1)
    logits = tf.matmul(h, W2) + b2
    return logits, (W1, b1, W2, b2)

def build_problem(hidden, W2_orig, b2_orig, tgt, eps, minimise, timeout):
    """Create a Z3 optimisation (or feasibility) problem."""
    opt = Optimize(); opt.set('timeout', timeout * 1000)
    abs_diffs = [];  W2p = [[None]*10 for _ in range(8)]

    for i in range(8):
        for d in range(10):
            w  = Real('w_{}_{}'.format(i, d));  W2p[i][d] = w
            ad = Real('diff_{}_{}'.format(i, d))
            opt.add(ad >= w - W2_orig[i, d], ad >= W2_orig[i, d] - w)
            abs_diffs.append(ad)

    logits = [b2_orig[d] + sum(W2p[i][d]*hidden[i] for i in range(8))
              for d in range(10)]
    opt.add(Or(*(logits[k] > logits[tgt] for k in range(10) if k != tgt)))

    cost = Real('L1');  opt.add(cost == sum(abs_diffs))
    handle = opt.minimize(cost) if minimise else None
    if eps is not None and not minimise:
        opt.add(cost <= eps)
    return opt, handle, W2p
# --------------------------------------------------------------------------

# CLI
ap = argparse.ArgumentParser()
ap.add_argument('--idx', type=int)
ap.add_argument('--label', type=int, default=9, choices=range(10))
ap.add_argument('--eps', type=float)
ap.add_argument('--eps-step', type=float, default=0.1)
ap.add_argument('--eps-max',  type=float, default=2.0)
ap.add_argument('--timeout',  type=int,   default=5000)
args = ap.parse_args()

# load model & image
ckpt = 'posit16_ultratinymlp.ckpt'
model_name = 'posit16'         # first field for CSV
(_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
idx = args.idx if args.idx is not None else next(
    i for i, y in enumerate(y_test) if y == args.label)
img = (X_test[idx].astype(np.float32) - 127.5) / 127.5
print('image idx={}  label={}  checkpoint={}'.format(idx, y_test[idx], ckpt))

tf.reset_default_graph()
x_ph = tf.placeholder(tf.posit16, [None, 28, 28, 1])
logits_tf, (W1_tf, b1_tf, W2_tf, b2_tf) = ultratiny(x_ph)
logits_f32 = tf.cast(logits_tf, tf.float32)
pred_tf = tf.argmax(logits_f32, axis=1, output_type=tf.int32)
saver = tf.train.Saver({v.name.split(':')[0]: v
                        for v in [W1_tf, b1_tf, W2_tf, b2_tf]})

with tf.Session() as s0:
    s0.run(tf.global_variables_initializer())
    saver.restore(s0, ckpt)
    W1, b1, W2, b2 = s0.run([W1_tf, b1_tf, W2_tf, b2_tf])
    base_logits, base_pred = s0.run(
        [logits_f32, pred_tf], {x_ph: img[None, :, :, None]})

print('baseline logits:', base_logits[0])
print('baseline prediction:', base_pred[0])
for ln in ascii_digit(img): print(ln)

hidden = np.maximum(0, np.matmul(img.reshape(1, 784), W1) + b1)[0]
base_label = int(base_pred[0])

# epsilon schedule
step  = args.eps_step
upper = args.eps_max
eps_vals = []

if args.eps is not None:
    eps_vals.append(args.eps)
    k = 1
    while args.eps + k*step <= upper + 1e-9:
        eps_vals.append(args.eps + k*step)
        k += 1
else:
    eps_vals = list(np.arange(step, upper + 1e-9, step))

t0_script = time.perf_counter()
sat_found = False
csv_line  = None

for eps in eps_vals:
    opt, hndl, W2p = build_problem(hidden, W2, b2,
                                   base_label, eps,
                                   minimise=False,
                                   timeout=args.timeout)

    t0 = time.perf_counter()
    res = opt.check()
    slice_t = time.perf_counter() - t0
    total_t = time.perf_counter() - t0_script

    if res == sat:
        print('SAT within eps={:.3f}  slice={:.2f}s  total={:.1f}s'
              .format(eps, slice_t, total_t))

        # refinement: minimise L1 under same ε
        opt2, h2, _ = build_problem(hidden, W2, b2,
                                    base_label, eps,
                                    minimise=True,
                                    timeout=args.timeout)
        opt2.check()
        L1min = z3_to_float(opt2.lower(h2))
        mdl   = opt2.model()

        # build W2_adv
        W2_adv = np.zeros((8, 10), dtype=np.float32)
        for i in range(8):
            for d in range(10):
                W2_adv[i, d] = z3_to_float(mdl[Real('w_{}_{}'.format(i, d))])

        # delta statistics
        delta = W2_adv - W2.astype(np.float32)
        L1    = np.abs(delta).sum()
        mean  = L1 / delta.size
        mx    = np.abs(delta).max()
        rel   = 100.0 * L1 / np.abs(W2.astype(np.float32)).sum()
        print('min L1 inside eps={:.3f}  ->  {:.6f}'.format(eps, L1min))
        print('delta stats  L1={:.6f}  mean={:.4f}  max={:.4f}  rel={:.2f}%'
              .format(L1, mean, mx, rel))

        # verify in TF
        tf.reset_default_graph()
        x = tf.constant(img.reshape(1, 28, 28, 1), tf.float32)
        h_tf = tf.nn.relu(tf.matmul(tf.reshape(x, [-1, 784]),
                                    tf.constant(W1, tf.float32))
                          + tf.constant(b1, tf.float32))
        logits_new = (tf.matmul(h_tf, tf.constant(W2_adv, tf.float32))
                      + tf.constant(b2, tf.float32))
        with tf.Session() as s1:
            logits_val, pred_new = s1.run(
                [logits_new,
                 tf.argmax(logits_new, axis=1, output_type=tf.int32)])
        new_label = int(pred_new[0])
        print('adversarial logits :', logits_val[0])
        print('adversarial predict:', new_label)

        # CSV line
        csv_line = '{},{},{},{:.6f},{:.6f},{:.6f},{:.2f},{:.1f}'.format(
            model_name, base_label, new_label, L1, mx, mean, rel, total_t)
        sat_found = True
        break

    tag = 'TIMEOUT' if opt.reason_unknown() == 'timeout' else 'UNSAT'
    print('eps={:.3f}  {}  slice={:.2f}s total={:.1f}s'
          .format(eps, tag, slice_t, total_t))

if not sat_found:
    print('No SAT up to eps_max -- minimising ...')
    opt, hndl, W2p = build_problem(hidden, W2, b2,
                                   base_label, eps=None,
                                   minimise=True,
                                   timeout=args.timeout)
    opt.check()
    L1 = z3_to_float(opt.lower(hndl))
    total_t = time.perf_counter() - t0_script
    delta = 0.0  # not needed for table when UNSAT
    mx = mean = rel = 0.0
    csv_line = '{},{},--,{:.6f},{:.6f},{:.6f},{:.2f},{:.1f}'.format(
        model_name, base_label, L1, mx, mean, rel, total_t)
    print('MIN L1={:.6f}  total={:.1f}s'.format(L1, total_t))

if csv_line:
    print('\nTable:', csv_line)
