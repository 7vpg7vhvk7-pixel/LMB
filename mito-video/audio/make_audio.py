"""Саундтрек + финальный микс для ролика «Мито-паспорт».
Читает voice/timeline.json (тайминг сцен) и voice/voice.wav (озвучка),
синтезирует музыку и звуковые эффекты, приглушает музыку под голос, пишет mix.wav.
Использование: python3 make_audio.py <папка mito-video> <выходной wav>"""
import json, sys, wave
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
TL = json.load(open(f'{ROOT}/voice/timeline.json', encoding='utf-8'))
SC = TL['scenes']; DUR = TL['total']
SR = 44100; N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)
BEAT = 0.6  # 100 BPM
S = lambda i, off=0.0: SC[i][0] + off           # абсолютное время: сцена i + смещение
END = lambda i: SC[i][1]

def add(sig, t0, gain=1.0, pan=0.0, bus=None):
    i = int(t0 * SR)
    if i >= N or i < 0: return
    s = sig[: N - i] * gain
    (bus or (L, R))[0][i:i + len(s)] += s * (1 - max(0, pan))
    (bus or (L, R))[1][i:i + len(s)] += s * (1 + min(0, pan))

def tt(d): return np.arange(int(d * SR)) / SR
def env(d, a=0.005, r=None):
    t = tt(d); e = np.minimum(1, t / max(a, 1e-4))
    return e * (np.exp(-t / r) if r else 1)
def lowpass(x, cut):
    a = np.exp(-2 * np.pi * cut / SR); k = int(SR * 5 / cut) + 1
    return np.convolve(x, (1 - a) * a ** np.arange(k))[: len(x)]
def highpass(x, cut): return x - lowpass(x, cut)
def noise(d): return rng.uniform(-1, 1, int(d * SR))
def note(n): return 440 * 2 ** ((n - 69) / 12)

# ===== музыка (отдельная шина — под неё «подныривает» голос) =====
ML = np.zeros(N); MR = np.zeros(N); MUS = (ML, MR)
def saw(f, d, det=0.0):
    t = tt(d); o = np.zeros_like(t)
    for h in range(1, 10): o += np.sin(2 * np.pi * f * (1 + det) * h * t + h) / h
    return o
CH = {'Am': [57, 60, 64, 71], 'F': [53, 57, 60, 64], 'C': [52, 55, 60, 67], 'G': [55, 59, 62, 69],
      'Bb': [58, 61, 65, 70], 'Ex': [52, 58, 64, 71]}
root = {'Am': 45, 'F': 41, 'C': 48, 'G': 43, 'Ex': 40, 'Bb': 46}
stress = (S(4), END(4)); final = S(10); build = S(5)  # build: после «паспорта» бит плотнее
def chord_at(t):
    if stress[0] <= t < stress[1]: return ['Ex', 'Bb'][int((t - stress[0]) // 2.4) % 2]
    if t >= final: return 'Am'
    return ['Am', 'F', 'C', 'G'][int(t // 4.8) % 4]
prog = []; t = 0.0
while t < DUR - 0.2:
    prog.append((t, chord_at(t))); t += 2.4
for t0, c in prog:
    d = min(3.6, DUR - t0)
    e = np.minimum(1, tt(d) / 0.5) * np.minimum(1, (d - tt(d)) / 0.9)
    cut = 900 if t0 < build else 1500
    for n in CH[c]:
        f = note(n); i = int(t0 * SR)
        sl = lowpass(saw(f, d, -0.003), cut) * e; sr_ = lowpass(saw(f, d, 0.003), cut) * e
        m = min(len(sl), N - i); g = 0.03
        ML[i:i + m] += sl[:m] * g; MR[i:i + m] += sr_[:m] * g

def kick():
    d = 0.45; t = tt(d); f = 45 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
def hat(): return highpass(noise(0.05), 7000) * env(0.05, 0.001, 0.012)
def bassnote(f, d=0.5):
    t = tt(d); return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 5) * np.minimum(1, t / 0.01)
b = 0
while b * BEAT < final:
    tb = b * BEAT; cur = chord_at(tb)
    if tb >= S(1):
        add(bassnote(note(root[cur])), tb, 0.17, bus=MUS)
        if tb >= build or b % 2 == 0: add(kick(), tb, 0.34 if tb >= build else 0.24, bus=MUS)
    if tb >= S(2):
        add(hat(), tb + BEAT / 2, 0.06, 0.3, bus=MUS); add(hat(), tb, 0.035, -0.3, bus=MUS)
    b += 1
# лёгкий пульс во вступлении
for k in range(int(S(1) / BEAT)):
    add(lowpass(kick(), 300), k * BEAT * 2, 0.16, bus=MUS)
# сердцебиение в сцене «нет сил»
for hb in [0.4, 1.5, 2.7, 4.0]:
    for o, g in [(0, 0.3), (0.18, 0.2)]: add(lowpass(kick(), 200), S(1, hb + o), g, bus=MUS)
# тревога в сцене стресса
d = stress[1] - stress[0]; t = tt(d)
al = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.5 * t))) * np.minimum(1, t / .3) * np.minimum(1, (d - t) / .5)
add(lowpass(al, 1200), stress[0], 0.03, bus=MUS)

# ===== звуковые эффекты (не приглушаются) =====
def tick(): return highpass(noise(0.02), 3000) * env(0.02, 0.0005, 0.004)
for s in range(0, int(DUR)): add(tick(), s, 0.09, 0.5)
def whoosh(d=0.5):
    x = noise(d); t = tt(d); y = np.zeros_like(x)
    for a, bnd in [(0, .25), (.25, .5), (.5, .75), (.75, 1)]:
        seg = slice(int(a * len(x)), int(bnd * len(x))); y[seg] = highpass(lowpass(x, 600 + 5000 * bnd), 200)[seg]
    return y * (t / d) ** 2
def impact(big=1.0):
    d = 1.6; t = tt(d); f = 30 + 70 * np.exp(-t * 8)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) * 0.9 + lowpass(noise(d), 2500) * np.exp(-t * 9) * 0.5) * big
for i in range(1, len(SC)):
    add(whoosh(), S(i) - 0.5, 0.2); add(impact(), S(i), 0.32)
add(impact(1.3), S(5, 1.6), 0.5)     # «ПАСПОРТ»
add(impact(1.3), S(10, 0.05), 0.45)  # логотип
def blip(f=1600):
    t = tt(0.09); return np.sin(2 * np.pi * f * t) * np.exp(-t * 45)
def pop():
    t = tt(0.2); return np.sin(2 * np.pi * (180 + 300 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
blips = [(2, 2.0, 1500), (3, 1.6, 1400), (3, 2.0, 1800), (3, 2.3, 1500), (3, 2.7, 1900), (3, 3.0, 1600), (3, 3.4, 2000),
         (4, .9, 900), (4, 1.0, 700), (6, 1.8, 1700), (6, 2.2, 2100), (7, 2.0, 1300), (7, 2.4, 2200), (7, 2.45, 1800)]
blips += [(7, 2.8 + k * .4, 1500 + k * 100) for k in range(5)]
blips += [(8, 1.1 + k * .35, 1500 + k * 100) for k in range(4)] + [(8, 2.8 + k * .15, 2000 + k * 100) for k in range(5)]
blips += [(10, 1.4, 1800), (0, 0.05, 2200)]
for sc_, off, f in blips:
    if off < END(sc_) - S(sc_): add(blip(f), S(sc_, off), 0.09, 0.4 if f % 200 else -0.4)
for sc_, off in [(1, 3.1), (2, .55), (3, .9), (4, .75), (6, 3.0), (8, .5), (9, .45), (10, 1.4)]:
    add(pop(), S(sc_, off), 0.28)
# сцена 0: «поп» на каждой фразе приветствия + скан-линия по логотипу
for sb in [x for x in TL['subs'] if x['scene'] == 0]: add(pop(), sb['s'], 0.18)
c1 = [x for x in TL['subs'] if x['scene'] == 0][1]['s']
def sweep(f0, f1, d):
    t = tt(d); f = f0 * (f1 / f0) ** (t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / .05) * np.minimum(1, (d - t) / .2)
add(sweep(400, 2400, 0.8), c1 + 0.2, 0.04)
add(sweep(900, 140, 2.3), S(1, .3), 0.05)   # энергия падает
add(sweep(140, 1200, 2.3), S(9, .3), 0.05)  # энергия заряжается
def key(): return highpass(noise(0.015), 2500) * env(0.015, 0.0005, 0.003)
for a, b_ in [(S(0, 0), S(0, .5)), (S(1, 1), S(1, 1.8)), (S(6, .1), S(6, .5)), (S(7, .1), S(7, .9)), (S(10, 1.7), S(10, 2.3))]:
    t = a
    while t < b_: add(key(), t, 0.06 + rng.uniform(0, .03), rng.uniform(-.5, .5)); t += 0.045 + rng.uniform(0, .03)

# ===== голос + ducking =====
import os
NO_VOICE = os.environ.get('NO_VOICE') == '1'   # версия без озвучки: музыка + субтитры
with wave.open(f'{ROOT}/voice/voice.wav') as w:
    v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
v = np.pad(v, (0, max(0, N - len(v))))[:N]
# лёгкая «радийная» обработка: срез НЧ, присутствие, компрессия
v = highpass(v, 90)
pres = highpass(lowpass(v, 5000), 2000); v = v + 0.35 * pres
v = np.tanh(v * 2.2) / np.tanh(2.2)
envv = lowpass(np.abs(v), 6); envv = envv / (envv.max() + 1e-9)
duck = 1 - 0.55 * np.clip(envv * 4, 0, 1)   # до −7 дБ под голосом
if NO_VOICE: duck = np.ones(N); v = v * 0
ML *= duck; MR *= duck
L += ML + v * 0.95; R += MR + v * 0.95

fade = np.ones(N); fs = int((DUR - 1.2) * SR); fade[fs:] = np.linspace(1, 0, N - fs) ** 1.5
fi = int(0.3 * SR); fade[:fi] *= np.linspace(0, 1, fi)
mix = np.stack([L * fade, R * fade], 1)
mix = np.tanh(mix * 1.3) / np.tanh(1.3)
mix /= np.max(np.abs(mix)) / 0.93
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
print('ok', OUT, round(DUR, 2), 's')
