"""End to end through HTTP with the fake generator; everything after generation is the real code (clean-up, placement, export)."""
import io
import json
import os
import sys
from pathlib import Path
import time
import zipfile

from fastapi.testclient import TestClient
import pytest

from noteforge_api.app import create_app

ENGINE_PY = os.environ.get('NOTEFORGE_ENGINE_PYTHON', sys.executable)
H = {'Authorization': 'Bearer secret'}
OSU = (Path(__file__).parent/'fixture.osu').read_bytes()


@pytest.fixture()
def client(tmp_path):
    app = create_app(data=tmp_path, token='secret', engine_python=ENGINE_PY, fake=True)
    with TestClient(app) as c:
        yield c


def audio(c, body=b'fake audio bytes'):
    r = c.post('/v1/audio', headers=H, files=dict(file=('song.mp3', body)))
    assert r.status_code == 201, r.text
    return r.json()['audio_id']


def wait(c, draft_id, limit=120):
    for _ in range(limit*10):
        s = c.get(f'/v1/drafts/{draft_id}', headers=H).json()
        if s['state'] not in ('queued', 'running'):
            return s
        time.sleep(.1)
    raise AssertionError('draft did not finish')


def draft(c, mode='phira', **over):
    body = dict(audio_id=audio(c), mode=mode, **(dict(level=5.) if mode == 'phira' else {}), **over)
    r = c.post('/v1/drafts', headers=H, json=body)
    assert r.status_code == 202, r.text
    s = wait(c, r.json()['draft_id'])
    assert s['state'] == 'done', s
    return s['draft_id']


def test_token_is_required_and_health_is_open(client):
    assert client.get('/v1/capabilities').status_code == 401
    assert client.get('/v1/capabilities', headers={'Authorization': 'Bearer no'}).json()['error']['code'] == 'unauthorized'
    assert client.get('/v1/health').json()['ready'] is True


def test_audio_rules(client):
    r = client.post('/v1/audio', headers=H, files=dict(file=('song.exe', b'x')))
    assert r.status_code == 415 and r.json()['error']['code'] == 'unsupported_audio'
    assert client.post('/v1/audio', headers=H, files=dict(file=('a.mp3', b''))).status_code == 422
    assert audio(client) == audio(client)


@pytest.mark.parametrize('body,field', [
    (dict(mode='phira'), 'level'),
    (dict(mode='mania_4k', style=dict(flick=.1)), 'flick'),
    (dict(mode='phira', level=5, timing=dict(bpm=120, chart_id='ab')), 'exactly one'),
    (dict(mode='phira', level=5, samples=1), 'at least 2'),
    (dict(mode='phira', level=5, surprise=1), 'Extra'),
])
def test_bad_requests_say_why(client, body, field):
    r = client.post('/v1/drafts', headers=H, json=dict(audio_id=audio(client), **body))
    assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_request'
    assert field in r.json()['error']['message'] or field in str(r.json()['error']['field'])


def test_unknown_ids(client):
    assert client.post('/v1/drafts', headers=H, json=dict(audio_id='ab'*32, mode='mania_4k')).json()['error']['code'] == 'audio_not_found'
    assert client.get('/v1/drafts/'+'0'*24, headers=H).status_code == 404
    assert client.get('/v1/drafts/..%2f..', headers=H).status_code in (404, 422)


def test_phira_full_flow(client):
    d = draft(client)
    events = client.get(f'/v1/drafts/{d}/events', headers=H).json()
    assert events['revision'] == 0 and events['notes'] and all('x' in n and 'lane' not in n for n in events['notes'])
    sections = client.get(f'/v1/drafts/{d}/sections', headers=H, params=dict(length_s=16)).json()['sections']
    assert len(sections) == 4 and len(sections[0]['candidates']) == 8
    pick = sections[1]
    chosen = pick['candidates'][-1]
    mine = [dict(time_ms=33000., end_ms=33000., type='tap', x=123.)]
    r = client.post(f'/v1/drafts/{d}/revisions', headers=H, json=dict(choices=[
        dict(start_ms=pick['start_ms'], end_ms=pick['end_ms'], sample=chosen['sample']),
        dict(start_ms=32000., end_ms=48000., notes=mine)]))
    assert r.status_code == 201, r.text
    assert r.json()['revision'] == 1
    assert [n['x'] for n in r.json()['notes'] if 32000 <= n['time_ms'] < 48000] == [123.]
    assert client.get(f'/v1/drafts/{d}/events', headers=H, params=dict(revision=0)).json()['revision'] == 0
    f = client.get(f'/v1/drafts/{d}/chart', headers=H, params=dict(speed=12))
    assert f.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(f.content))
    chart = json.loads(z.read('chart.json'))
    assert chart['judgeLineList'][0]['eventLayers'][0]['speedEvents'][0]['start'] == 12.
    assert 'audio.mp3' in z.namelist() and len(chart['judgeLineList'][0]['notes']) == len(r.json()['notes'])
    assert client.get(f'/v1/drafts/{d}/chart', headers=H, params=dict(format='osz')).status_code == 422


def test_mania_flow_and_conflicts_are_repaired(client):
    d = draft(client, mode='mania_4k')
    sections = client.get(f'/v1/drafts/{d}/sections', headers=H, params=dict(length_s=8, include_notes=False)).json()['sections']
    assert 'notes' not in sections[0]['candidates'][0]
    choices = [dict(start_ms=s['start_ms'], end_ms=s['end_ms'], sample=k % 8) for k, s in enumerate(sections)]
    r = client.post(f'/v1/drafts/{d}/revisions', headers=H, json=dict(choices=choices))
    assert r.status_code == 201, r.text
    assert all(n['lane'] in range(4) for n in r.json()['notes'])
    for form, name in (('osz', 'chart.osu'), ('osu', None)):
        f = client.get(f'/v1/drafts/{d}/chart', headers=H, params=dict(format=form))
        assert f.status_code == 200, f.text
        text = zipfile.ZipFile(io.BytesIO(f.content)).read(name).decode() if name else f.text
        assert '[HitObjects]' in text


def test_revision_rules(client):
    d = draft(client)
    url = f'/v1/drafts/{d}/revisions'
    two = [dict(start_ms=0, end_ms=2000, sample=0), dict(start_ms=1000, end_ms=3000, sample=1)]
    assert client.post(url, headers=H, json=dict(choices=two)).json()['error']['code'] == 'choices_overlap'
    assert client.post(url, headers=H, json=dict(choices=[dict(start_ms=0, end_ms=1, sample=99)])).json()['error']['code'] == 'bad_sample'
    assert client.post(url, headers=H, json=dict(choices=[dict(start_ms=0, end_ms=9, sample=1, notes=[])])).status_code == 422
    outside = dict(start_ms=0, end_ms=9, notes=[dict(time_ms=50, end_ms=50, type='tap')])
    assert client.post(url, headers=H, json=dict(choices=[outside])).status_code == 422


def test_same_request_is_the_same_draft(client):
    a = audio(client)
    body = dict(audio_id=a, mode='mania_4k')
    one = client.post('/v1/drafts', headers=H, json=body).json()['draft_id']
    assert client.post('/v1/drafts', headers=H, json=body).json()['draft_id'] == one
    assert client.post('/v1/drafts', headers=H, json=dict(body, seed=1)).json()['draft_id'] != one
    wait(client, one)


def test_most_agreeing_and_chart_upload(client):
    r = client.post('/v1/charts', headers=H, data=dict(mode='mania_4k'), files=dict(file=('a.osu', OSU)))
    assert r.status_code == 201, r.text
    info = r.json()
    assert info['notes'] > 0 and info['segments'] and info['opening_style']['hold'] >= 0
    again = client.post('/v1/charts', headers=H, data=dict(mode='mania_4k'), files=dict(file=('a.osu', OSU)))
    assert again.json()['chart_id'] == info['chart_id']
    d = draft(client, mode='mania_4k', aggregation='most_agreeing', timing=dict(chart_id=info['chart_id']), opening_chart_id=info['chart_id'])
    assert client.get(f'/v1/drafts/{d}/events', headers=H).json()['base']['kind'] == 'sample'
    bad = client.post('/v1/charts', headers=H, data=dict(mode='phira'), files=dict(file=('a.json', b'{}')))
    assert bad.status_code == 422 and bad.json()['error']['code'] == 'chart_unreadable'
    wrong = client.post('/v1/drafts', headers=H, json=dict(audio_id=audio(client), mode='phira', level=5, opening_chart_id=info['chart_id']))
    assert wrong.json()['error']['code'] == 'mode_mismatch'


def test_restart_reports_an_interrupted_draft(tmp_path):
    app = create_app(data=tmp_path, token='secret', engine_python=ENGINE_PY, fake=True)
    with TestClient(app) as c:
        d = draft(c)
    status = tmp_path/'drafts'/d/'status.json'
    status.write_text(json.dumps(dict(json.loads(status.read_text()), state='running')))
    with TestClient(create_app(data=tmp_path, token='secret', engine_python=ENGINE_PY, fake=True)) as c:
        assert c.get(f'/v1/drafts/{d}', headers=H).json()['error']['code'] == 'interrupted'


def test_a_charters_mania_hold_survives_nearby_generated_notes(client):
    d = draft(client, mode='mania_4k')
    hold = dict(time_ms=10000., end_ms=25000., type='hold', lane=2)
    r = client.post(f'/v1/drafts/{d}/revisions', headers=H, json=dict(choices=[dict(start_ms=9000., end_ms=11000., notes=[hold])]))
    assert r.status_code == 201, r.text
    notes = r.json()['notes']
    assert [n['end_ms'] for n in notes if n['time_ms'] == 10000. and n['lane'] == 2] == [25000.]
    assert not [n for n in notes if n['lane'] == 2 and 10000. < n['time_ms'] < 25040.]


def test_continue_keeps_what_came_before(client):
    for mode in ('phira', 'mania_4k'):
        d = draft(client, mode=mode)
        before = client.get(f'/v1/drafts/{d}/events', headers=H).json()['notes']
        r = client.post(f'/v1/drafts/{d}/continue', headers=H, json=dict(from_ms=30000.))
        assert r.status_code == 201, r.text
        after = r.json()['notes']
        key = lambda n: (n['time_ms'], n['end_ms'], n['type'], n.get('lane'), n.get('x'))
        assert [key(n) for n in before if n['time_ms'] < 30000] == [key(n) for n in after if n['time_ms'] < 30000]
        if mode == 'phira':  # a chord really was kept note by note
            times = [n['time_ms'] for n in before if n['time_ms'] < 30000]
            assert len(times) > len(set(times))
        assert r.json()['revision'] == 1 and any(n['time_ms'] >= 30000 for n in after)
        assert client.post(f'/v1/drafts/{d}/continue', headers=H, json=dict(from_ms=1e9)).json()['error']['code'] == 'bad_cut'
