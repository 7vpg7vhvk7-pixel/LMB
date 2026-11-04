"""Озвучка (RHVoice, yuriy) -> общий voice.wav + timeline.json/timeline.js.
Каждая строка lines.txt = одна сцена. Внутри строки фраза бьётся на куски по знакам препинания,
время кусков берётся по паузам в синтезированной речи (для субтитров)."""
import json, re, subprocess, sys, wave
import numpy as np

HERE = sys.argv[1]
SR = 44100
VOICE, RATE, PITCH = 'yuriy', '138', '112'
LEAD, TAIL = 0.3, 0.4
MIN = {2: 3.2, 10: 5.6}           # минимальная длина сцен (индексы сцен)
# что показывать в субтитрах вместо «фонетического» текста
DISPLAY = {'Лав Май Бади': 'Love My Body', 'АТФ': 'АТФ'}

def tts(text):
    tmp = f'{HERE}/_tmp.wav'
    subprocess.run(['RHVoice-test', '-p', VOICE, '-q', 'max', '-r', RATE, '-t', PITCH, '-o', tmp], input=text.encode(), capture_output=True)
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', tmp, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()

def trim(a, th=0.01):
    idx = np.where(np.abs(a) > th)[0]
    return a[max(0, idx[0] - 200): idx[-1] + 2000] if len(idx) else a

def gaps(a, th=0.012, min_len=0.07):
    win = int(SR * 0.01); n = len(a) // win
    rms = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(1))
    sil = rms < th; out = []; s = None
    for k, v in enumerate(sil):
        if v and s is None: s = k
        if not v and s is not None:
            if (k - s) * 0.01 >= min_len: out.append(((s + k) / 2 * 0.01, (k - s) * 0.01))
            s = None
    return out

def chunks(line):
    parts = re.split(r'(?<=[,.!?:—])\s+', line.strip())
    return [p for p in parts if p]

lines = [l.strip() for l in open(f'{HERE}/lines.txt', encoding='utf-8') if l.strip()]
voice = []; scenes = []; subs = []; T = 0.0
for i, line in enumerate(lines):
    a = trim(tts(line)); d = len(a) / SR
    ch = chunks(line)
    # пропорциональная оценка границ + привязка к ближайшей паузе
    lens = np.array([len(c) for c in ch], float); cum = np.cumsum(lens) / lens.sum()
    g = gaps(a); bounds = []
    for c in cum[:-1]:
        est = c * d
        best = min(g, key=lambda x: abs(x[0] - est), default=None)
        bounds.append(best[0] if best and abs(best[0] - est) < 0.8 else est)
    bounds = [0.0] + sorted(bounds) + [d]
    dur = max(MIN.get(i, 0), LEAD + d + TAIL)
    scenes.append([round(T, 3), round(T + dur, 3)])
    for k, c in enumerate(ch):
        txt = c
        for src, dst in DISPLAY.items(): txt = txt.replace(src, dst)
        subs.append({'scene': i, 's': round(T + LEAD + bounds[k], 3), 'e': round(T + LEAD + bounds[k + 1], 3), 'text': txt})
    pad = np.zeros(int(SR * LEAD)); tail = np.zeros(int(SR * (dur - LEAD - d)))
    voice += [pad, a, tail]; T += dur
    print(f'{i:2d} dur={dur:5.2f} voice={d:5.2f} chunks={len(ch)} gaps={len(g)}')

v = np.concatenate(voice)
v = v / np.max(np.abs(v)) * 0.95
with wave.open(f'{HERE}/voice.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((v * 32767).astype(np.int16).tobytes())
tl = {'total': round(T, 3), 'scenes': scenes, 'subs': subs}
json.dump(tl, open(f'{HERE}/timeline.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(f'{HERE}/../anim/timeline.js', 'w', encoding='utf-8').write('window.TL=' + json.dumps(tl, ensure_ascii=False) + ';')
print('total', round(T, 2))
