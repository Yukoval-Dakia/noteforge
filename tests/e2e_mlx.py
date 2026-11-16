"""Real MLX model through HTTP: draft, continue, regenerate and download.

e2e_mlx.py BASE TOKEN_FILE OUT AUDIO HUMAN_ZIP [PLAN]
"""
import json, sys, time, zipfile
from pathlib import Path
import httpx
base, token, out, audio, human_zip = sys.argv[1:6]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
c = httpx.Client(base_url=base, headers=dict(Authorization='Bearer '+Path(token).read_text().strip()), timeout=600)
t0 = time.time()
a = c.post('/v1/audio', files=dict(file=(Path(audio).name, Path(audio).read_bytes()))); a.raise_for_status(); aid = a.json()['audio_id']
z = zipfile.ZipFile(human_zip); human = next(z.read(n) for n in z.namelist() if n.endswith('.json') and b'judgeLineList' in z.read(n))
ch = c.post('/v1/charts', files=dict(file=('chart.json', human)), data=dict(mode='phira')); ch.raise_for_status(); ch = ch.json()
print('chart upload', round(time.time()-t0, 1), 's; level', round(ch['notes']/159.1, 3), flush=True)
body = dict(audio_id=aid, mode='phira', level=round(ch['notes']/159.1, 3), timing=dict(chart_id=ch['chart_id']), opening_chart_id=ch['chart_id'], aggregation='single', samples=1, plan=sys.argv[6] if len(sys.argv) > 6 else 'balanced', title='hug')
t1 = time.time(); r = c.post('/v1/drafts', json=body); r.raise_for_status(); d = r.json()['draft_id']; last = None
while True:
    s = c.get(f'/v1/drafts/{d}').json()
    now = (s['state'], (s.get('progress') or {}).get('stage'))
    if now != last: print(round(time.time()-t1, 1), now, flush=True); last = now
    if s['state'] not in ('queued', 'running'): break
    time.sleep(.5)
print('draft wall', round(time.time()-t1, 1), s.get('error'), s.get('song'), flush=True)
assert s['state'] == 'done'
ev = c.get(f'/v1/drafts/{d}/events').json(); print('notes', len(ev['notes']), ev['warnings'], flush=True)
for cut in (120000., 120000., 60000., 150000.):
    t = time.time(); r = c.post(f'/v1/drafts/{d}/continue', json=dict(from_ms=cut, revision=0, seed=7)); print('continue from', cut/1000, 's:', r.status_code, 'wall', round(time.time()-t, 1), 's', flush=True)
    r.raise_for_status(); e = r.json()
    k = lambda n: (n['time_ms'], n['end_ms'], n['type'], n.get('x'))
    srt = lambda L: sorted(L, key=lambda t: tuple(-1 if v is None else v for v in (t[0], t[1])) + (t[2], t[3] if t[3] is not None else 0))
    print('  kept identical', srt([k(n) for n in ev['notes'] if n['time_ms'] < cut]) == srt([k(n) for n in e['notes'] if n['time_ms'] < cut]), 'new notes', sum(n['time_ms'] >= cut for n in e['notes']), 'warnings', e['warnings'], flush=True)
f = c.get(f'/v1/drafts/{d}/chart'); print('chart', f.status_code, len(f.content), flush=True); (out/'hug-mlx.zip').write_bytes(f.content)
