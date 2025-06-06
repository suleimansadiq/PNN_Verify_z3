#!/usr/bin/env python3
"""
VERIFY_PATCH_POSIT32_FULLFEATURE_ASCII.PY

Patch‑based Z3 adversarial search on a tf.posit32 ultra‑tiny MNIST MLP.

Features
--------
• baseline ASCII digit, logits, soft‑max probabilities
• user‑controlled patch window, eps grid / single eps, per‑eps timeout
• ε‑grid search (minimal‑L1) until the real posit32 network flips
• ASCII diff of adversarial image
• crisp 560×560 PNGs (images2/before.png, images2/after.png)
  – baseline PNG has **no** rectangle  
  – adversarial PNG shows red outline + lime “X” marks

CLI flags (all optional)
------------------------
--idx N              attack test image N
--label D            first test image with label D (default 9)
--patch P            window: 12,17 | 8:11,5 | 11:13,11:13
--eps E              test exactly ε; else scan grid
--eps-step S         ε grid step (default 0.1)
--eps-max M          ε grid upper bound (default 2.0)
--timeout T          Z3 timeout per ε in seconds (default 5000)

Author: Suleiman Sadiq
"""

import argparse, os, re, time, numpy as np, tensorflow as tf
from z3 import Optimize, Real, Bool, Or, Implies, Not, sat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# ---------- ASCII helpers --------------------------------------------------
def ascii_digit(img):
    pal = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    return [''.join(next(ch for ch, t in pal if img[r, c] < t) for c in range(28))
            for r in range(28)]

def ascii_changed(orig, adv, thr=5e-3):
    pal = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    rows = []
    for r in range(28):
        row = ''
        for c in range(28):
            row += 'X' if abs(adv[r, c] - orig[r, c]) > thr else \
                   next(ch for ch, t in pal if adv[r, c] < t)
        rows.append(row)
    return rows

def z3_to_float(val):
    s = str(val).split('+')[0].strip()
    if s.startswith('(/'):
        _, p, q, *_ = re.split(r'[()\s]+', s)
        return float(p) / float(q)
    if '/' in s:
        p, q = s.split('/')
        return float(p) / float(q)
    return float(s)

# ---------- PNG export -----------------------------------------------------
def save_digit(img28, rect, diff_mask=None,
               fname='out.png', draw_rect=True):
    """Save a 560×560 PNG of a 28×28 digit, with optional red rectangle and X."""
    r0, r1, c0, c1 = rect
    upscale, dpi = 20, 100
    big = np.kron(img28, np.ones((upscale, upscale)))
    fig, ax = plt.subplots(figsize=(big.shape[1] / dpi,
                                    big.shape[0] / dpi),
                           dpi=dpi)
    ax.imshow(big, cmap='gray', vmin=-1, vmax=1,
              origin='upper', interpolation='none')

    if draw_rect:
        ax.add_patch(plt.Rectangle((c0 * upscale, r0 * upscale),
                                   (c1 - c0 + 1) * upscale,
                                   (r1 - r0 + 1) * upscale,
                                   fill=False, lw=2, edgecolor='red'))

    if diff_mask is not None:
        ys, xs = np.where(diff_mask)
        ax.scatter(xs * upscale + upscale / 2,
                   ys * upscale + upscale / 2,
                   marker='x', s=30, linewidths=1, c='lime')

    ax.set_xticks([]); ax.set_yticks([])

    # keep a thin, even white border on all sides
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)

    os.makedirs('images2', exist_ok=True)
    fig.savefig(os.path.join('images2', fname),
                dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# ---------- utility --------------------------------------------------------
def parse_range(seg):
    if ':' in seg:
        a, b = map(int, seg.split(':'))
    else:
        a = b = int(seg)
    if a > b or a < 0 or b > 27:
        raise ValueError('patch coords must satisfy 0 <= start <= end <= 27')
    return a, b

# ---------- network definition --------------------------------------------
def build_ultratiny_posit32(x):
    W1 = tf.get_variable('Variable',   [784, 8],  dtype=tf.posit32)
    b1 = tf.get_variable('Variable_1', [8],       dtype=tf.posit32)
    fc1 = tf.nn.relu(tf.matmul(tf.reshape(x, [-1, 784]), W1) + b1)
    W2 = tf.get_variable('Variable_2', [8, 10],   dtype=tf.posit32)
    b2 = tf.get_variable('Variable_3', [10],      dtype=tf.posit32)
    logits = tf.matmul(fc1, W2) + b2
    return logits, (W1, b1, W2, b2)

# ---------- main -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int)
    ap.add_argument('--label', type=int, default=9, choices=range(10))
    ap.add_argument('--patch', default='20:27,0:27')
    ap.add_argument('--eps', type=float)
    ap.add_argument('--eps-step', type=float, default=0.1)
    ap.add_argument('--eps-max', type=float, default=2.0)
    ap.add_argument('--timeout', type=int, default=5000)
    args = ap.parse_args()

    r0, r1 = parse_range(args.patch.split(',')[0])
    c0, c1 = parse_range(args.patch.split(',')[1])
    n_pix = (r1 - r0 + 1) * (c1 - c0 + 1)
    print('patch', args.patch, ' (#pix', n_pix, ')')

    CKPT = 'posit32_ultratinymlp.ckpt'
    print('checkpoint:', CKPT)

    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    idx = args.idx if args.idx is not None else next(
        i for i, y in enumerate(y_test) if y == args.label)
    true_lbl = int(y_test[idx])
    orig = (X_test[idx].astype(np.float32) - 127.5) / 127.5
    print('image idx=', idx, ' label=', true_lbl)

    tf.reset_default_graph()
    x_ph = tf.placeholder(tf.posit32, [None, 28, 28, 1])
    logits_tf, vars_tf = build_ultratiny_posit32(x_ph)
    logits_f32 = tf.cast(logits_tf, tf.float32)
    probs_tf = tf.nn.softmax(logits_f32)
    pred_tf = tf.argmax(logits_f32, axis=1, output_type=tf.int32)
    saver = tf.train.Saver({v.name.split(':')[0]: v for v in vars_tf})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, CKPT)
        W1, b1, W2, b2 = sess.run(vars_tf)
        base_logits, base_probs, base_pred = sess.run(
            [logits_f32, probs_tf, pred_tf], {x_ph: orig[None, :, :, None]})

    print('baseline logits      :', base_logits[0])
    print('baseline probabilities:', base_probs[0])
    print('baseline prediction  :', base_pred[0])
    for line in ascii_digit(orig): print(line)

    eps_list = [args.eps] if args.eps is not None else \
               np.arange(args.eps_step, args.eps_max + 1e-9, args.eps_step)

    t_script0 = time.perf_counter()

    for eps in eps_list:
        opt = Optimize(); opt.set('timeout', args.timeout * 1000)
        xs = [Real('x_%d' % i) for i in range(784)]
        abs_diffs = []

        for r in range(28):
            for c in range(28):
                i = r * 28 + c; v0 = float(orig[r, c])
                if r0 <= r <= r1 and c0 <= c <= c1:
                    opt.add(xs[i] >= v0 - eps, xs[i] <= v0 + eps)
                    d = Real('d_%d' % i); opt.add(d >= xs[i] - v0, d >= v0 - xs[i])
                    abs_diffs.append(d)
                else:
                    opt.add(xs[i] == v0)

        # hidden ReLU
        h = []
        for j in range(8):
            lin = float(b1[j]) + sum(float(W1[i, j]) * xs[i] for i in range(784))
            hj = Real('h_%d' % j); rb = Bool('rb_%d' % j)
            opt.add((lin >= 0) == rb)
            opt.add(Implies(rb, hj == lin), Implies(Not(rb), hj == 0))
            h.append(hj)

        # output layer
        logits = []
        for d in range(10):
            lv = float(b2[d]) + sum(float(W2[j, d]) * h[j] for j in range(8))
            z = Real('log_%d' % d); opt.add(z == lv); logits.append(z)

        opt.add(Or(*(logits[k] > logits[true_lbl] for k in range(10) if k != true_lbl)))
        cost = Real('L1'); opt.add(cost == sum(abs_diffs))
        h_cost = opt.minimize(cost)

        t0 = time.perf_counter()
        res = opt.check()
        eps_time = time.perf_counter() - t0
        total_time = time.perf_counter() - t_script0

        if res == sat:
            mdl = opt.model()
            adv = np.array([z3_to_float(mdl[xs[i]]) for i in range(784)],
                           dtype=np.float32).reshape(28, 28)

            # re‑check with real network
            with tf.Session() as sess2:
                sess2.run(tf.global_variables_initializer())
                saver.restore(sess2, CKPT)
                adv_logits, adv_probs, adv_pred = sess2.run(
                    [logits_f32, probs_tf, pred_tf],
                    {x_ph: adv[None, :, :, None]})

            if adv_pred[0] == true_lbl:
                print('eps=%.3f: Z3 SAT but posit32 still predicts %d -- continuing'
                      % (eps, true_lbl))
                continue

            min_L1 = z3_to_float(opt.lower(h_cost))
            print('\nSAT  eps=%.3f  eps time=%.3fs  total=%.1fs  L1=%.6f' %
                  (eps, eps_time, total_time, min_L1))

            diff_mask = np.abs(adv - orig) > 5e-3

            print('\nASCII of adversarial image (X = changed pixel):\n')
            for line in ascii_changed(orig, adv): print(line)

            # save PNGs
            save_digit(orig, (r0, r1, c0, c1),
                       fname='before.png', draw_rect=False)
            save_digit(adv, (r0, r1, c0, c1),
                       diff_mask, fname='after.png', draw_rect=True)

            print('figures saved to images2/before.png and images2/after.png')

            print('\nadversarial logits      :', adv_logits[0])
            print('adversarial probabilities:', adv_probs[0])
            print('adversarial prediction   :', adv_pred[0])
            return
        else:
            status = 'TIMEOUT' if opt.reason_unknown() == 'timeout' else 'UNSAT'
            print('eps=%.3f  %s  (eps time=%.3fs, total=%.1fs)' %
                  (eps, status, eps_time, total_time))

    print('no adversarial patch found up to eps_max')


if __name__ == '__main__':
    main()
