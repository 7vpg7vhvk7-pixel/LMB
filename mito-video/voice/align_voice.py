"""Подгонка готовой озвучки (например, из ElevenLabs) под сцены ролика.
Запись режется на сцены по паузам: границы фраз из lines.txt оцениваются по числу букв
и привязываются к ближайшей заметной паузе. Внутри сцены куски субтитров — так же.
Выход: voice.wav, timeline.json, ../anim/timeline.js (как у build_voice.py).
Использование: python3 align_voice.py <папка voice> <аудиофайл>"""
import json, re, subprocess, sys, wave
import numpy as np

HERE, SRC = sys.argv[1], sys.argv[2]
SR = 44100
LEAD, TAIL = 0.12, 0.18
MIN = {2: 3.2, 10: 5.6}
DISPLAY = {'Лав Май Бади': 'Love My Body'}

raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', SRC, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
A = np.frombuffer(raw, np.float32).copy()
D = len(A) / SR

def gaps(a, th_db=-38, min_len=0.06):
    win = int(SR * 0.01); n = len(a) // win
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(1)) + 1e-9
    sil = 20 * np.log10(rms / rms.max()) < th_db; out = []; s = None
    for k, v in enumerate(sil):
        if v and s is None: s = k
        if not v and s is not None:
            if (k - s) * 0.01 >= min_len: out.append(((s + k) / 2 * 0.01, (k - s) * 0.01, s * 0.01, k * 0.01))
            s = None
    return out, sil

G, SIL = gaps(A)
# начало/конец речи
speech = np.where(~SIL)[0]; t0, t1 = speech[0] * 0.01, (speech[-1] + 1) * 0.01

def pick(est, cands, lo, hi, w=0.35):
    c = [g for g in cands if lo < g[0] < hi]
    if not c: return None
    return max(c, key=lambda g: g[1] - w * abs(g[0] - est))

lines = [l.strip() for l in open(f'{HERE}/lines.txt', encoding='utf-8') if l.strip()]
weights = np.array([len(re.sub(r'\s+', ' ', l)) for l in lines], float)
cum = np.cumsum(weights) / weights.sum()
bounds = [t0]; info = []
for k, c in enumerate(cum[:-1]):
    # оценка по буквам, но с поправкой на уже найденную границу
    rest = (cum[k] - (cum[k - 1] if k else 0)) / (1 - (cum[k - 1] if k else 0))
    est = bounds[-1] + rest * (t1 - bounds[-1])
    g = pick(est, G, bounds[-1] + 0.5, t1 - 0.3)
    b = g[0] if g else est
    info.append((k, round(est, 2), round(b, 2), round(g[1], 2) if g else None))
    bounds.append(b)
bounds.append(t1)
for x in info: print('граница после сцены %d: оценка %.2f → пауза %.2f (длина %s)' % x)

def chunks(line): return [p for p in re.split(r'(?<=[,.!?:—])\s+', line.strip()) if p]

voice = []; scenes = []; subs = []; T = 0.0
for i, line in enumerate(lines):
    a_s, a_e = bounds[i], bounds[i + 1]
    # обрезаем тишину по краям сегмента (но оставляем 60 мс)
    seg_sil = SIL[int(a_s * 100): int(a_e * 100)]
    nz = np.where(~seg_sil)[0]
    s_s = a_s + max(0, nz[0] * 0.01 - 0.06); s_e = a_s + min(len(seg_sil), nz[-1] + 1) * 0.01 + 0.08
    seg = A[int(s_s * SR): int(min(s_e, D) * SR)]; d = len(seg) / SR
    dur = max(MIN.get(i, 0), LEAD + d + TAIL)
    scenes.append([round(T, 3), round(T + dur, 3)])
    ch = chunks(line); lens = np.array([len(c) for c in ch], float); cc = np.cumsum(lens) / lens.sum()
    inner = [g for g in G if s_s + 0.2 < g[0] < s_e - 0.2]
    cb = [0.0]
    for c in cc[:-1]:
        est = c * d; g = pick(s_s + est, inner, s_s + cb[-1] + 0.15, s_e - 0.15, 0.6)
        cb.append((g[0] - s_s) if g and abs(g[0] - s_s - est) < 0.9 else est)
    cb.append(d)
    for k, c in enumerate(ch):
        txt = c
        for src, dst in DISPLAY.items(): txt = txt.replace(src, dst)
        subs.append({'scene': i, 's': round(T + LEAD + cb[k], 3), 'e': round(T + LEAD + cb[k + 1], 3), 'text': txt})
    voice += [np.zeros(int(SR * LEAD)), seg, np.zeros(int(SR * (dur - LEAD - d)))]
    print(f'{i:2d} исх {s_s:6.2f}-{s_e:6.2f}  длина {d:5.2f}  сцена {dur:5.2f}')
    T += dur

v = np.concatenate(voice); v = v / (np.max(np.abs(v)) + 1e-9) * 0.95
with wave.open(f'{HERE}/voice.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((v * 32767).astype(np.int16).tobytes())
tl = {'total': round(T, 3), 'scenes': scenes, 'subs': subs}
json.dump(tl, open(f'{HERE}/timeline.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(f'{HERE}/../anim/timeline.js', 'w', encoding='utf-8').write('window.TL=' + json.dumps(tl, ensure_ascii=False) + ';')
print('итого', round(T, 2))
