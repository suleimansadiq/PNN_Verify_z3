#!/usr/bin/env python3
"""
z3_verify_lenet5.py

Patch‑based Z3 adversarial search for a LeNet‑5 checkpoint (MNIST or Fashion‑MNIST).

Command‑line flags
    dtype               posit32 | posit16 | posit8 | float16 | float32
    --patch P           row0:row1,col0:col1              [20:27,0:27]
    --eps E             solve only this ε                [None → grid]
    --eps-step S        ε‑grid step                      [0.1]
    --eps-max M         ε‑grid upper bound               [0.5]
    --timeout T         Z3 timeout per ε (seconds)       [5000]
    --dataset D         mnist | fashion_mnist            [mnist]
    --idx N             attack image with index N
    --label L           first image whose label == L     [9]
The patch is specified on the 28×28 core image; it is shifted by +2
to align with the 32×32 padded input used by LeNet‑5.
"""

import argparse, os, sys, time, threading, numpy as np, tensorflow as tf, z3, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.contrib.layers import flatten
from z3 import Real, Optimize, Or


def parse_range(seg):
    return tuple(map(int, seg.split(':'))) if ':' in seg else (int(seg), int(seg))


cli = argparse.ArgumentParser()
cli.add_argument('dtype')
cli.add_argument('--patch', default='20:27,0:27')
cli.add_argument('--eps', type=float)
cli.add_argument('--eps-step', type=float, default=0.1)
cli.add_argument('--eps-max', type=float, default=0.5)
cli.add_argument('--timeout', type=int, default=5000)
cli.add_argument('--dataset', default='mnist', choices=['mnist', 'fashion_mnist'])
cli.add_argument('--idx', type=int)
cli.add_argument('--label', type=int, default=9)
args = cli.parse_args()

r0, r1 = parse_range(args.patch.split(',')[0])
c0, c1 = parse_range(args.patch.split(',')[1])
pr0, pr1, pc0, pc1 = r0 + 2, r1 + 2, c0 + 2, c1 + 2
print('patch', args.patch)

TF_MAP = {'posit32': tf.posit32, 'posit16': tf.posit16, 'posit8': tf.posit8,
          'float16': tf.float16, 'float32': tf.float32}
tf_type = TF_MAP.get(args.dtype)
if tf_type is None:
    sys.exit('dtype must be one of ' + ', '.join(TF_MAP))
print('numeric type', args.dtype)


ds = mnist if args.dataset == 'mnist' else fashion_mnist
(_, _), (X_test, y_test) = ds.load_data()
X_test = np.pad(np.expand_dims(X_test, -1),
                ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
idx = args.idx if args.idx is not None else next(
    i for i, lbl in enumerate(y_test) if lbl == args.label)
orig_img = ((X_test[idx].astype(np.float32) - 127.5) / 127.5)
orig_lbl = int(y_test[idx])
print('image idx', idx, 'label', orig_lbl)


def ascii_digit(img28):
    palette = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    return [''.join(next(ch for ch, t in palette if img28[r, c] < t) for c in range(28))
            for r in range(28)]


def ascii_changed(a, b, thr=5e-3):
    palette = [('.', -0.5), (':', 0), ('+', 0.5), ('#', 1.1)]
    rows = []
    for r in range(28):
        rows.append(''.join('X' if abs(a[r, c] - b[r, c]) > thr else
                            next(ch for ch, t in palette if b[r, c] < t)
                            for c in range(28)))
    return rows


for line in ascii_digit(orig_img[2:-2, 2:-2, 0]):
    print(line)


def save_png(img28, rect, mask, fname, draw_rect):
    up, dpi = 20, 100
    big = np.kron(img28, np.ones((up, up)))
    fig, ax = plt.subplots(figsize=(big.shape[1] / dpi, big.shape[0] / dpi), dpi=dpi)
    ax.imshow(big, cmap='gray', vmin=-1, vmax=1, interpolation='none')
    if draw_rect:
        r0_, r1_, c0_, c1_ = rect
        ax.add_patch(plt.Rectangle((c0_ * up, r0_ * up),
                                   (c1_ - c0_ + 1) * up, (r1_ - r0_ + 1) * up,
                                   fill=False, lw=2, ec='red'))
    if mask is not None:
        ys, xs = np.where(mask)
        ax.scatter(xs * up + up / 2, ys * up + up / 2,
                   marker='x', s=30, lw=1, c='lime')
    ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(.02, .02, .98, .98)
    os.makedirs('images2', exist_ok=True)
    fig.savefig(os.path.join('images2', fname),
                dpi=100, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def z3max(*vals):
    m = vals[0]
    for v in vals[1:]:
        m = z3.If(v > m, v, m)
    return m


class Heartbeat:
    def __init__(self):
        self.running = False
        self.elapsed = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def loop(self):
        while self.running:
            time.sleep(10)
            self.elapsed += 10
            if self.elapsed % 60 == 0:
                print(f' [{self.elapsed}s]', end='', flush=True)
            else:
                print(' .', end='', flush=True)


tf.reset_default_graph()
x_ph = tf.placeholder(tf_type, [None, 32, 32, 1])


def lenet(x):
    mu, sigma = 0.0, 0.1
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma, dtype=tf_type))
    b1 = tf.Variable(tf.zeros(6, dtype=tf_type))
    c1 = tf.nn.relu(tf.nn.conv2d(x, W1, [1, 1, 1, 1], 'VALID') + b1)
    p1 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma, dtype=tf_type))
    b2 = tf.Variable(tf.zeros(16, dtype=tf_type))
    c2 = tf.nn.relu(tf.nn.conv2d(p1, W2, [1, 1, 1, 1], 'VALID') + b2)
    p2 = tf.nn.max_pool(c2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    flat = flatten(p2)

    W3 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma, dtype=tf_type))
    b3 = tf.Variable(tf.zeros(120, dtype=tf_type))
    f3 = tf.nn.relu(tf.matmul(flat, W3) + b3)

    W4 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma, dtype=tf_type))
    b4 = tf.Variable(tf.zeros(84, dtype=tf_type))
    f4 = tf.nn.relu(tf.matmul(f3, W4) + b4)

    W5 = tf.Variable(tf.truncated_normal([84, 10], mean=mu, stddev=sigma, dtype=tf_type))
    b5 = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(f4, W5) + b5

    return logits, [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]


logits_tf, vars_tf = lenet(x_ph)
logits32 = tf.cast(logits_tf, tf.float32)
pred_tf = tf.argmax(logits32, 1, output_type=tf.int32)
saver = tf.train.Saver({v.name.split(':')[0]: v for v in vars_tf})
ckpt_path = f'./{args.dtype}.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    weights = sess.run(vars_tf)
    base_pred = sess.run(pred_tf, {x_ph: orig_img[None]})[0]
print('baseline prediction', base_pred)


W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = [w.astype(np.float64) for w in weights]
dvars = [Real(f'x_{i}') for i in range(32 * 32)]


def conv2d_sym(v, W, b, in_w):
    Kh, Kw, _, oc = W.shape
    out, H = [], in_w - Kh + 1
    for k in range(oc):
        ch = []
        for r in range(H):
            for c in range(H):
                s = b[k]
                for rr in range(Kh):
                    for cc in range(Kw):
                        s += W[rr, cc, 0, k] * v[(r + rr) * in_w + (c + cc)]
                ch.append(s)
        out.append(ch)
    return out, H


def relu_fm(fm):
    return [[z3.If(v > 0, v, 0) for v in ch] for ch in fm]


def pool_fm(fm, H):
    pooled = []
    for ch in fm:
        pc = []
        for r in range(0, H, 2):
            for c in range(0, H, 2):
                idx = [(r + dr) * H + (c + dc) for dr in (0, 1) for dc in (0, 1)]
                pc.append(z3max(*[ch[i] for i in idx]))
        pooled.append(pc)
    return pooled, H // 2


def fc(vec, W, b):
    return [b[j] + sum(W[i, j] * vec[i] for i in range(len(vec)))
            for j in range(W.shape[1])]


fm1, H1 = conv2d_sym(dvars, W1.reshape(5, 5, 1, 6), b1, 32)
fm1 = relu_fm(fm1); fm1, H1 = pool_fm(fm1, H1)
flat1 = [v for ch in fm1 for v in ch]

fm2, H2 = conv2d_sym(flat1, W2.reshape(5, 5, 6, 16), b2, H1)
fm2 = relu_fm(fm2); fm2, _ = pool_fm(fm2, H2)
flat2 = [v for ch in fm2 for v in ch]

fc3 = fc(flat2, W3, b3)
fc3 = [z3.If(v > 0, v, 0) for v in fc3]
fc4 = fc(fc3, W4, b4)
fc4 = [z3.If(v > 0, v, 0) for v in fc4]
logits_sym = [b5[j] + sum(W5[i, j] * fc4[i] for i in range(84)) for j in range(10)]


def build_opt(eps):
    opt = Optimize()
    opt.set('timeout', args.timeout * 1000)
    for R in range(32):
        for C in range(32):
            i = R * 32 + C
            v0 = float(orig_img[R, C, 0])
            if pr0 <= R <= pr1 and pc0 <= C <= pc1:
                opt.add(dvars[i] >= v0 - eps)
                opt.add(dvars[i] <= v0 + eps)
            else:
                opt.add(dvars[i] == v0)
    l1 = Real('l1')
    opt.add(l1 == z3.Sum([z3.Abs(dvars[i] - float(orig_img[i // 32, i % 32, 0]))
                          for i in range(32 * 32)]))
    opt.add(Or(*[logits_sym[k] > logits_sym[base_pred] for k in range(10) if k != base_pred]))
    handle = opt.minimize(l1)
    return opt, handle


eps_values = [args.eps] if args.eps is not None else \
             np.arange(args.eps_step, args.eps_max + 1e-9, args.eps_step)
start = time.time()
found = False

try:
    for eps in eps_values:
        print(f'eps={eps:.3f}  t={time.time() - start:.1f}s  solving...', flush=True)
        opt, h = build_opt(eps)
        hb = Heartbeat(); hb.start()
        t0 = time.time()
        res = opt.check()
        hb.stop(); print()  # newline after heartbeat
        dt = time.time() - t0

        if res == z3.sat:
            mdl = opt.model()
            adv32 = np.array([float(mdl[dvars[i]].as_decimal(20)[:-1])
                              if mdl[dvars[i]].is_real()
                              else float(mdl[dvars[i]])
                              for i in range(32 * 32)],
                             dtype=np.float32).reshape(32, 32)
            adv28 = adv32[2:-2, 2:-2]
            diff = np.abs(adv28 - orig_img[2:-2, 2:-2, 0]) > 5e-3
            print(f'  SAT   time={dt:.1f}s')
            for ln in ascii_changed(orig_img[2:-2, 2:-2, 0], adv28):
                print(ln)
            save_png(orig_img[2:-2, 2:-2, 0], (r0, r1, c0, c1), None, 'before.png', False)
            save_png(adv28, (r0, r1, c0, c1), diff, 'after.png', True)
            print('saved images2/before.png and after.png')
            found = True
            break
        else:
            status = 'timeout' if opt.reason_unknown() == 'timeout' else 'unsat'
            print(f'  {status} after {dt:.1f}s', flush=True)

except KeyboardInterrupt:
    print('\n[CTRL+C] interrupted by user')
    sys.exit(0)

if not found:
    print('no adversarial example found')
