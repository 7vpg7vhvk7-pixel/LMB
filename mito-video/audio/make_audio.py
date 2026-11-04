"""Синтез саундтрека для ролика «Мито-паспорт» (60 с, 44.1 кГц, стерео).
Все события синхронизированы с таймлайном anim/index.html."""
import sys, wave
import numpy as np

SR = 44100
DUR = 60.0
N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)
BEAT = 0.6  # 100 BPM

def add(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    if i >= N: return
    s = sig[: N - i] * gain
    L[i:i + len(s)] += s * (1 - max(0, pan))
    R[i:i + len(s)] += s * (1 + min(0, pan))

def tt(d): return np.arange(int(d * SR)) / SR
def env(d, a=0.005, r=None):
    t = tt(d); e = np.minimum(1, t / max(a, 1e-4))
    return e * (np.exp(-t / r) if r else 1)

def lowpass(x, cut):
    a = np.exp(-2 * np.pi * cut / SR); y = np.empty_like(x); acc = 0.0
    # быстрый однополюсный ФНЧ через свёртку экспонентой
    k = int(SR * 5 / cut) + 1
    kern = (1 - a) * a ** np.arange(k)
    return np.convolve(x, kern)[: len(x)]

def highpass(x, cut): return x - lowpass(x, cut)
def noise(d): return rng.uniform(-1, 1, int(d * SR))
def note(n): return 440 * 2 ** ((n - 69) / 12)

# ---------- пэд ----------
def saw(f, d, det=0.0):
    t = tt(d); o = np.zeros_like(t)
    for h in range(1, 10):
        o += np.sin(2 * np.pi * f * (1 + det) * h * t + h) / h
    return o

CH = {  # аккорды (MIDI)
    'Am': [57, 60, 64, 71], 'F': [53, 57, 60, 64], 'C': [52, 55, 60, 67], 'G': [55, 59, 62, 69],
    'Dm': [50, 57, 62, 65], 'Bb': [58, 61, 65, 70], 'Ex': [52, 58, 64, 71],
}
prog = []
t = 0.0
seq = ['Am', 'F', 'C', 'G']
while t < 60:
    if 19 <= t < 25: c = ['Ex', 'Bb'][int((t - 19) // 2.4) % 2]
    elif t >= 56.4: c = 'Am'
    else: c = seq[int(t // 4.8) % 4]
    prog.append((t, c)); t += 2.4
for t0, c in prog:
    d = 2.4 + 1.2
    if t0 >= 56.4: d = 60 - t0
    if d < 0.2: continue
    e = np.minimum(1, tt(d) / 0.5) * np.minimum(1, (d - tt(d)) / 0.9)
    for k, n in enumerate(CH[c]):
        f = note(n)
        sl = lowpass(saw(f, d, -0.003), 900 if t0 < 25 else 1500) * e
        sr_ = lowpass(saw(f, d, 0.003), 900 if t0 < 25 else 1500) * e
        g = 0.028 if t0 >= 5 else 0.02
        i = int(t0 * SR); m = min(len(sl), N - i)
        L[i:i + m] += sl[:m] * g; R[i:i + m] += sr_[:m] * g

# ---------- бас + кик + хэты ----------
def kick():
    d = 0.45; t = tt(d); f = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
def hat():
    return highpass(noise(0.05), 7000) * env(0.05, 0.001, 0.012)
def bassnote(f, d=0.5):
    t = tt(d); return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 5) * np.minimum(1, t / 0.01)

root = {'Am': 45, 'F': 41, 'C': 48, 'G': 43, 'Ex': 40, 'Bb': 46}
b = 0
while b * BEAT < 56.4:
    tb = b * BEAT
    cur = [c for t0, c in prog if t0 <= tb][-1]
    if tb >= 5:
        add(bassnote(note(root[cur])), tb, 0.16)
        if tb >= 25 or b % 2 == 0: add(kick(), tb, 0.33 if tb >= 25 else 0.22)
    if tb >= 11 and not (19 <= tb < 19.6):
        add(hat(), tb + BEAT / 2, 0.06, 0.3); add(hat(), tb, 0.035, -0.3)
    b += 1

# ---------- сердцебиение во вступлении ----------
for hb in [0.5, 1.5, 2.6, 3.8]:
    for o, g in [(0, 0.35), (0.18, 0.22)]:
        add(lowpass(kick(), 200), hb + o, g)

# ---------- тиканье таймера ----------
def tick(): return highpass(noise(0.02), 3000) * env(0.02, 0.0005, 0.004)
for s in range(0, 60): add(tick(), s, 0.12, 0.5)

# ---------- переходы: вжух + удар ----------
def whoosh(d=0.5):
    x = noise(d); t = tt(d)
    y = np.zeros_like(x);
    for k, (a, bnd) in enumerate([(0, .25), (.25, .5), (.5, .75), (.75, 1)]):
        seg = slice(int(a * len(x)), int(bnd * len(x)))
        y[seg] = highpass(lowpass(x, 600 + 5000 * bnd), 200)[seg]
    return y * (t / d) ** 2
def impact(big=1.0):
    d = 1.6; t = tt(d); f = 30 + 70 * np.exp(-t * 8)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    cr = lowpass(noise(d), 2500) * np.exp(-t * 9)
    return (sub * 0.9 + cr * 0.5) * big
for s in [5, 11, 19, 25, 31, 37, 45, 53, 56.5]:
    add(whoosh(), s - 0.5, 0.22)
    add(impact(), s, 0.35)
add(impact(1.3), 26.6, 0.5)    # «ПАСПОРТ»
add(impact(1.3), 56.55, 0.45)  # логотип

# ---------- блипы и «поп» плашек ----------
def blip(f=1600):
    t = tt(0.09); return np.sin(2 * np.pi * f * t) * np.exp(-t * 45)
def pop():
    t = tt(0.2); return np.sin(2 * np.pi * (180 + 300 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
for s, f in [(7.0, 1500), (12.6, 1400), (13.0, 1800), (13.3, 1500), (13.7, 1900), (14.0, 1600), (14.4, 2000),
             (19.9, 900), (20.0, 700), (32.8, 1700), (33.2, 2100), (39.0, 1300), (39.4, 2200), (39.45, 1800),
             (39.8, 1500), (40.2, 1600), (40.6, 1700), (41.0, 1800), (41.4, 1900),
             (46.1, 1500), (46.45, 1600), (46.8, 1700), (47.15, 1800),
             (47.8, 2000), (47.95, 2100), (48.1, 2200), (48.25, 2300), (48.4, 2400), (58.9, 1800)]:
    add(blip(f), s, 0.10, 0.4 if f % 200 else -0.4)
for s in [3.1, 5.55, 11.9, 34.0, 45.5, 53.45, 57.9, 19.75]:
    add(pop(), s, 0.3)

# ---------- стук клавиш под субтитры ----------
def key(): return highpass(noise(0.015), 2500) * env(0.015, 0.0005, 0.003)
for a, b_ in [(1.0, 1.8), (7.4, 8.6), (15.2, 17.4), (21.4, 23.8), (27.6, 29.2), (34.8, 36.0), (37.1, 37.9),
              (49.2, 50.4), (54.0, 56.0), (58.2, 58.8)]:
    t = a
    while t < b_:
        add(key(), t, 0.07 + rng.uniform(0, .03), rng.uniform(-.5, .5)); t += 0.045 + rng.uniform(0, .03)

# ---------- свипы шкалы энергии ----------
def sweep(f0, f1, d):
    t = tt(d); f = f0 * (f1 / f0) ** (t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / .05) * np.minimum(1, (d - t) / .2)
add(sweep(900, 140, 2.3), 0.3, 0.05)
add(sweep(140, 1200, 2.3), 53.3, 0.05)
# тревога в сцене стресса
t = tt(6); al = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.5 * t))) * np.minimum(1, t / .3) * np.minimum(1, (6 - t) / .5)
add(lowpass(al, 1200), 19, 0.035)
# хвост в конце
fade = np.ones(N); fs = int(58.8 * SR); fade[fs:] = np.linspace(1, 0, N - fs) ** 1.5
fi = int(0.3 * SR); fade[:fi] *= np.linspace(0, 1, fi)
L *= fade; R *= fade

# ---------- мастер ----------
mix = np.stack([L, R], 1)
mix = np.tanh(mix * 1.4) / np.tanh(1.4)
mix /= np.max(np.abs(mix)) / 0.92
pcm = (mix * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok', sys.argv[1])
