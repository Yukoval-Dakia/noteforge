"""Real model, real HTTP, through a tunnel. Usage: e2e_real.py BASE TOKEN_FILE OUT PHIRA_AUDIO PHIRA_ZIP OSU_AUDIO OSU_CHART"""
import io, json, sys, time, zipfile
from pathlib import Path
import httpx
base, token, out, ph_audio, ph_zip, m_audio, m_chart = sys.argv[1:8]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
c = httpx.Client(base_url=base, headers=dict(Authorization='Bearer '+Path(token).read_text().strip()), timeout=300)
t0 = time.time()
def log(*a): print(f'{time.time()-t0:6.0f}s', *a, flush=True)
def up(path, url, **data):
    r = c.post(url, files=dict(file=(Path(path).name, Path(path).read_bytes())), data=data); r.raise_for_status(); return r.json()
def run(body, limit=1100):
    r = c.post('/v1/drafts', json=body); r.raise_for_status(); d = r.json()['draft_id']; last = None
    while time.time()-t0 < 1500:
        s = c.get(f'/v1/drafts/{d}').json()
        now = (s['state'], (s.get('progress') or {}).get('stage'))
        if now != last: log(d[:8], now); last = now
        if s['state'] not in ('queued', 'running'): return d, s
        time.sleep(5)
    c.delete(f'/v1/drafts/{d}'); raise SystemExit('e2e deadline')
def f1(a, b, tol=20.):
    a, b = sorted(a), sorted(b); i = j = h = 0
    while i < len(a) and j < len(b):
        g = a[i]-b[j]
        if abs(g) <= tol: h, i, j = h+1, i+1, j+1
        elif g < 0: i += 1
        else: j += 1
    return 2*h/max(len(a)+len(b), 1)
report = {}
log('health', c.get('/v1/health').json())
def case(name, body, human_ms, form):
    tick = time.time(); d, s = run(body); row = dict(state=s['state'], error=s.get('error'), wall_s=round(time.time()-tick), used=s.get('used'), song=s.get('song'))
    if s['state'] == 'done':
        ev = c.get(f'/v1/drafts/{d}/events').json(); notes = ev['notes']; on = sorted({n['time_ms'] for n in notes})
        row.update(notes=len(notes), warnings=ev['warnings'])
        if human_ms: row.update(human_notes=len(human_ms), note_ratio=round(len(notes)/len(human_ms), 3), f1=round(f1(on, sorted(set(human_ms))), 3), f1_shift={k: round(f1([x+k for x in on], sorted(set(human_ms))), 3) for k in (-40, -20, 20, 40)})
        sec = c.get(f'/v1/drafts/{d}/sections', params=dict(length_s=16, include_notes=False)).json()['sections']
        row['sections'] = len(sec); row['candidates'] = len(sec[0]['candidates'])
        pick = [dict(start_ms=x['start_ms'], end_ms=x['end_ms'], sample=x['candidates'][0]['sample']) for x in sec[:3]]
        r = c.post(f'/v1/drafts/{d}/revisions', json=dict(choices=pick)); row['revision_status'] = r.status_code
        if r.status_code == 201: row.update(revision_notes=len(r.json()['notes']), revision_warnings=r.json()['warnings'])
        else: row['revision_error'] = r.text[:300]
        for f in form:
            g = c.get(f'/v1/drafts/{d}/chart', params=dict(format=f)); row['chart_'+f] = g.status_code
            if g.status_code == 200: (out/f'{name}.{f}').write_bytes(g.content)
            else: row['chart_error_'+f] = g.text[:300]
        (out/f'{name}-events.json').write_text(json.dumps(ev))
    report[name] = row; log(name, json.dumps(row, ensure_ascii=False)[:900]); (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
# Phira: the frozen setup
au = up(ph_audio, '/v1/audio')['audio_id']
z = zipfile.ZipFile(ph_zip); human = next(z.read(n) for n in z.namelist() if n.endswith('.json') and b'judgeLineList' in z.read(n))
ch = c.post('/v1/charts', files=dict(file=('chart.json', human)), data=dict(mode='phira')); ch.raise_for_status(); ch = ch.json(); log('phira chart', {k: ch[k] for k in ('notes', 'opening_style', 'warnings')})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noteforge.events.phira import parse_phira
hms = [e.onset_us/1000 for e in parse_phira(human).events]
case('phira_frozen', dict(audio_id=au, mode='phira', level=round(len(hms)/159.5, 3), timing=dict(chart_id=ch['chart_id']), opening_chart_id=ch['chart_id'], title='hug'), hms, ('phira_zip', 'rpe_json'))
# mania: the frozen setup
am = up(m_audio, '/v1/audio')['audio_id']
cm = up(m_chart, '/v1/charts', mode='mania_4k'); log('mania chart', {k: cm[k] for k in ('notes', 'opening_style', 'warnings')})
from noteforge.events.osu_mania import parse_osu
hm = [e.onset_us/1000 for e in parse_osu(Path(m_chart).read_bytes(), 'a.osu').events]
case('mania_frozen', dict(audio_id=am, mode='mania_4k', level=round(len(hm)/109, 3), timing=dict(chart_id=cm['chart_id']), opening_chart_id=cm['chart_id'], title='m'), hm, ('osz', 'osu'))
# only a BPM, two samples: the path without a chart
bpm = round(60e6/cm['segments'][0][1], 3)
case('mania_bpm_only', dict(audio_id=am, mode='mania_4k', timing=dict(bpm=bpm), samples=2, aggregation='single', seed=1), hm, ('osz',))
log('DONE')
