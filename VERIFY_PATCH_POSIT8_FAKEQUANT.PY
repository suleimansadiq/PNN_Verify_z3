#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERIFY_PATCH_POSIT8_FAKEQUANT.PY

Patch‑based Z3 adversarial attack on a fake‑quant **posit‑8** MNIST MLP.

Flags
-----
--idx N              attack the N‑th test image
--label D            first test image whose label is D (default 9)
--patch P            patch window (e.g. 12,17  or 8:11,5  or 11:13,11:13)
--eps E              test one ε; otherwise scan ε = Δ,2Δ,…,M
--eps-step Δ         ε grid step (default 0.1)
--eps-max M          ε grid upper bound (default 2.0)
--timeout T          Z3 timeout per ε (default 5 000 s)

Outputs
-------
baseline logits / probabilities / prediction / ASCII digit
for each ε: UNSAT / TIMEOUT / SAT with ε‑time and total time
on SAT: ASCII diff, adversarial logits / probabilities / prediction,
        and high‑resolution before.png / after.png in *images2/*

Author : Suleiman Sadiq
"""

import argparse, os, re, time, numpy as np, tensorflow as tf
from tensorflow.keras.datasets import mnist
from z3 import Optimize, Real, Bool, Or, Implies, Not, sat           # sat added
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)   # readable vectors

# ---------- ASCII helpers --------------------------------------------------
def ascii_digit(img):
    palette = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    return [''.join(next(ch for ch, t in palette if img[r, c] < t) for c in range(28))
            for r in range(28)]

def ascii_changed(orig, adv, thr=5e-3):
    palette = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    rows = []
    for r in range(28):
        rows.append(''.join('X' if abs(adv[r, c] - orig[r, c]) > thr else
                            next(ch for ch, t in palette if adv[r, c] < t)
                            for c in range(28)))
    return rows

def z3_to_float(val):
    s = str(val).split('+')[0].strip()
    if s.startswith('(/'):
        _, p, q, *_ = re.split(r'[()\s]+', s); return float(p)/float(q)
    if '/' in s:
        p, q = s.split('/'); return float(p)/float(q)
    return float(s)

# ---------- fake‑quant MLP --------------------------------------------------
def build_fakequant_mlp(x):
    W1 = tf.get_variable('W1_master', [784, 8], tf.float32)
    b1 = tf.get_variable('b1_master', [8], tf.float32)
    W2 = tf.get_variable('W2_master', [8, 10], tf.float32)
    b2 = tf.get_variable('b2_master', [10], tf.float32)
    W1_q = tf.cast(tf.cast(W1, tf.posit8), tf.float32)
    b1_q = tf.cast(tf.cast(b1, tf.posit8), tf.float32)
    W2_q = tf.cast(tf.cast(W2, tf.posit8), tf.float32)
    b2_q = tf.cast(tf.cast(b2, tf.posit8), tf.float32)
    flat = tf.reshape(x, [-1, 784])
    h1 = tf.nn.relu(tf.matmul(flat, W1_q) + b1_q)
    logits = tf.matmul(h1, W2_q) + b2_q
    return logits, (W1, b1, W2, b2)

# ---------- helpers --------------------------------------------------------
def parse_range(seg):
    if ':' in seg: a, b = map(int, seg.split(':'))
    else: a = b = int(seg)
    if a > b or a < 0 or b > 27:
        raise ValueError('patch coords must satisfy 0 ≤ start ≤ end ≤ 27')
    return a, b

def save_digit(img28, title, rect, diff=None, fname='out.png'):
    r0, r1, c0, c1 = rect
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.imshow(img28, cmap='gray', vmin=-1, vmax=1, origin='upper')
    ax.add_patch(plt.Rectangle((c0-.5, r0-.5), c1-c0+1, r1-r0+1,
                               fill=False, linewidth=2, edgecolor='red'))
    if diff is not None:
        ys, xs = np.where(diff); ax.scatter(xs, ys, marker='x', s=18, lw=.7, c='lime')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0); os.makedirs('images2', exist_ok=True)
    fig.savefig(os.path.join('images2', fname)); plt.close(fig)

# ---------- main -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int)
    ap.add_argument('--label', type=int, default=9, choices=range(10))
    ap.add_argument('--patch', default='10:12,10:12')
    ap.add_argument('--eps', type=float)
    ap.add_argument('--eps-max', type=float, default=2.0)
    ap.add_argument('--eps-step', type=float, default=0.1)
    ap.add_argument('--timeout', type=int, default=5000)
    args = ap.parse_args()

    r0, r1 = parse_range(args.patch.split(',')[0])
    c0, c1 = parse_range(args.patch.split(',')[1])

    (_, _), (x_test, y_test) = mnist.load_data()
    img_idx = args.idx if args.idx is not None else next(
        i for i, y in enumerate(y_test) if y == args.label)
    true_lbl = int(y_test[img_idx])
    orig = (x_test[img_idx].astype(np.float32) - 127.5) / 127.5
    print(f'image idx={img_idx}, label={true_lbl}, patch={args.patch}')

    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    logits_tf, vars_tf = build_fakequant_mlp(x_ph)
    probs_tf = tf.nn.softmax(logits_tf)
    pred_tf  = tf.argmax(logits_tf, 1, output_type=tf.int32)
    saver = tf.train.Saver({v.name.split(':')[0]: v for v in vars_tf})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'posit8_ultratinymlp.ckpt')
        W1_f, b1_f, W2_f, b2_f = sess.run(vars_tf)
        W1 = W1_f.astype(np.posit8).astype(np.float32)
        b1 = b1_f.astype(np.posit8).astype(np.float32)
        W2 = W2_f.astype(np.posit8).astype(np.float32)
        b2 = b2_f.astype(np.posit8).astype(np.float32)
        base_logits, base_probs, base_pred = sess.run(
            [logits_tf, probs_tf, pred_tf], {x_ph: orig[None, :, :, None]})

    print('baseline logits      :', base_logits[0])
    print('baseline probabilities:', base_probs[0])
    print('baseline prediction  :', base_pred[0])
    for line in ascii_digit(orig): print(line)

    eps_vals = [args.eps] if args.eps else \
               np.arange(args.eps_step, args.eps_max + 1e-9, args.eps_step)
    t0_script = time.perf_counter()

    for eps in eps_vals:
        opt = Optimize(); opt.set('timeout', args.timeout * 1000)
        xs = [Real(f'x_{i}') for i in range(784)]
        diffs = []
        for r in range(28):
            for c in range(28):
                i = r*28 + c; v0 = float(orig[r, c])
                if r0 <= r <= r1 and c0 <= c <= c1:
                    opt.add(xs[i] >= v0 - eps, xs[i] <= v0 + eps)
                    d = Real(f'd_{i}')
                    opt.add(d >= xs[i] - v0, d >= v0 - xs[i]); diffs.append(d)
                else: opt.add(xs[i] == v0)

        h = []
        for j in range(8):
            lin = b1[j] + sum(W1[i, j] * xs[i] for i in range(784))
            hj = Real(f'h_{j}'); rb = Bool(f'rb_{j}')
            opt.add((lin >= 0) == rb)
            opt.add(Implies(rb, hj == lin), Implies(Not(rb), hj == 0))
            h.append(hj)

        logits = []
        for d in range(10):
            lv = b2[d] + sum(W2[j, d] * h[j] for j in range(8))
            z = Real(f'log_{d}'); opt.add(z == lv); logits.append(z)

        opt.add(Or(*(logits[k] > logits[true_lbl] for k in range(10) if k != true_lbl)))
        L1 = Real('L1'); opt.add(L1 == sum(diffs)); h_cost = opt.minimize(L1)

        t0 = time.perf_counter(); res = opt.check(); e_time = time.perf_counter() - t0
        if res == sat:
            L1_val = z3_to_float(opt.lower(h_cost))
            print(f'\nSAT  eps={eps:.3f}  eps time={e_time:.3f}s  '
                  f'total={time.perf_counter()-t0_script:.1f}s  L1={L1_val}')
            mdl = opt.model()
            adv = np.array([z3_to_float(mdl[xs[i]]) for i in range(784)],
                           dtype=np.float32).reshape(28, 28)
            diff = np.abs(adv - orig) > 5e-3
            save_digit(orig, 'Before', (r0, r1, c0, c1), fname='before.png')
            save_digit(adv, 'After', (r0, r1, c0, c1), diff, 'after.png')
            with tf.Session() as s2:
                s2.run(tf.global_variables_initializer())
                saver.restore(s2, 'posit8_ultratinymlp.ckpt')
                adv_logits, adv_probs, adv_pred = s2.run(
                    [logits_tf, probs_tf, pred_tf], {x_ph: adv[None, :, :, None]})
            print('\nadversarial logits      :', adv_logits[0])
            print('adversarial probabilities:', adv_probs[0])
            print('adversarial prediction  :', adv_pred[0])
            return
        tag = 'TIMEOUT' if opt.reason_unknown()=='timeout' else 'UNSAT'
        print(f'eps={eps:.3f}  {tag}  '
              f'(eps time={e_time:.3f}s, total={time.perf_counter()-t0_script:.1f}s)')

    print('no adversarial patch found up to eps_max')

if __name__ == '__main__':
    main()
