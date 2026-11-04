"""Точная подгонка готовой озвучки под сцены по распознанным словам (words.json из ASR).
Слова сценария (lines.txt) сопоставляются с распознанными словами; сцены режутся между фразами,
субтитры получают время каждого слова (караоке-подсветка).
Использование: python3 align_words.py <папка voice> <аудиофайл> <words.json>"""
import difflib, json, re, subprocess, sys, wave
import numpy as np

HERE, SRC, WJ = sys.argv[1], sys.argv[2], sys.argv[3]
SR = 44100
MIN = {2: 3.2, 10: 5.6}   # минимальная длина сцен, чтобы анимация успела отыграть
DISPLAY = {'Лав': 'Love', 'Май': 'My', 'Бади.': 'Body.', 'Бади!': 'Body!', 'Бади': 'Body'}

raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', SRC, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True).stdout
A = np.frombuffer(raw, np.float32).copy(); D = len(A) / SR
asr = json.load(open(WJ, encoding='utf-8'))
norm = lambda w: re.sub(r'[^а-яa-z0-9]', '', w.lower().replace('ё', 'е'))

lines = [l.strip() for l in open(f'{HERE}/lines.txt', encoding='utf-8') if l.strip()]
script = []  # (line, слово как в тексте)
for i, l in enumerate(lines):
    for w in l.split():
        if norm(w): script.append([i, w, None, None])
        elif script: script[-1][1] += ' ' + w   # «—» прилипает к предыдущему слову

sm = difflib.SequenceMatcher(a=[norm(x[1]) for x in script], b=[norm(x['w']) for x in asr], autojunk=False)
for a0, b0, n in sm.get_matching_blocks():
    for k in range(n):
        script[a0 + k][2] = asr[b0 + k]['s']; script[a0 + k][3] = asr[b0 + k]['e']
matched = sum(1 for x in script if x[2] is not None)
print(f'совпало слов: {matched}/{len(script)}')
# несовпавшие слова — интерполяция между соседями
idx = [k for k, x in enumerate(script) if x[2] is not None]
for k, x in enumerate(script):
    if x[2] is None:
        p = max([j for j in idx if j < k], default=None); q = min([j for j in idx if j > k], default=None)
        ps = script[p][3] if p is not None else 0.0; qs = script[q][2] if q is not None else D
        np_ = (p if p is not None else -1); nq = (q if q is not None else len(script))
        f0 = (k - np_) / (nq - np_); f1 = (k + 1 - np_) / (nq - np_)
        x[2] = ps + (qs - ps) * f0; x[3] = ps + (qs - ps) * f1

# точки разреза: середина паузы между фразами
first = {}; last = {}
for x in script:
    first.setdefault(x[0], x[2]); last[x[0]] = x[3]
cuts = [0.0] + [(last[i - 1] + first[i]) / 2 for i in range(1, len(lines))] + [D]

voice = []; scenes = []; subs = []; T = 0.0
for i in range(len(lines)):
    c0, c1 = cuts[i], cuts[i + 1]
    seg = A[int(c0 * SR): int(c1 * SR)]
    # мягкие края, чтобы не щёлкало
    f = min(len(seg) // 2, int(.01 * SR)); seg = seg.copy(); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
    d = len(seg) / SR; dur = max(MIN.get(i, 0), d)
    scenes.append([round(T, 3), round(T + dur, 3)])
    ws = [x for x in script if x[0] == i]
    # куски субтитров по знакам препинания
    chunk = []
    for x in ws:
        chunk.append(x)
        if re.search(r'[,.!?:—]$', x[1]) or x is ws[-1]:
            words = [{'w': ' '.join(DISPLAY.get(t, t) for t in y[1].split()), 's': round(T + y[2] - c0, 3), 'e': round(T + y[3] - c0, 3)} for y in chunk]
            subs.append({'scene': i, 's': words[0]['s'], 'e': words[-1]['e'], 'text': ' '.join(w['w'] for w in words), 'words': words})
            chunk = []
    voice += [seg, np.zeros(int(SR * (dur - d)))]
    print(f'{i:2d} аудио {c0:6.2f}-{c1:6.2f}  сцена {dur:5.2f}  «{ws[0][1]} … {ws[-1][1]}»')
    T += dur

v = np.concatenate(voice); v = v / (np.max(np.abs(v)) + 1e-9) * 0.95
with wave.open(f'{HERE}/voice.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((v * 32767).astype(np.int16).tobytes())
tl = {'total': round(T, 3), 'scenes': scenes, 'subs': subs}
json.dump(tl, open(f'{HERE}/timeline.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
open(f'{HERE}/../anim/timeline.js', 'w', encoding='utf-8').write('window.TL=' + json.dumps(tl, ensure_ascii=False) + ';')
print('итого', round(T, 2))
