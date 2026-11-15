"""Engine-side operations. Runs in the engine's Python, in a child process, so the API process needs no torch.

    ops.py inspect-chart  CHART_FILE MODE            prints JSON: notes, timing segments, opening style
    ops.py generate       DRAFT_DIR                  samples + revision 0
    ops.py assemble       DRAFT_DIR REVISION_JSON    a new revision from a base and section choices
    ops.py export         DRAFT_DIR REVISION FORMAT SPEED OUT

All times stored on disk by this module are microseconds on the audio's playback clock.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('NOTEFORGE_ENGINE_ROOT', HERE.parent)).resolve()
sys.path[:0] = [str(ROOT)]

FIELD_TEMPERATURE = '{"kind": 1.0, "head_block": 0.7, "head_cell": 0.7, "head_ms": 0.7}'
OPENING_US = 30_000_000
VOTES_OF_8 = 3
PHASE_BAND_BINS = 6
LEVEL_TOLERANCE = .15


class OpError(Exception):
    """A failure the API reports to the caller as it is."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def progress(draft, stage, message=''):
    write_json(Path(draft)/'progress.json', dict(stage=stage, message=message))


WARM_DRAFT = None  # set by noteforge/m3/engine_server.py: run the draft in this process, on the loaded model


def run(command, log, env=None):
    command = [str(x) for x in command]
    if WARM_DRAFT is not None and command[1].endswith('plan_song.py'):
        import contextlib
        import runpy
        saved, saved_path = sys.argv, list(sys.path)
        sys.argv = command[1:]
        try:
            with open(log, 'a') as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                runpy.run_path(command[1], run_name='__main__')
        except SystemExit as exc:
            if exc.code:
                raise OpError('engine_failed', f'planning failed: {exc.code}') from exc
        finally:
            sys.argv = saved
            sys.path[:] = saved_path  # the planner adds its folder, whose apply.py would hide the placement model's
        return
    if WARM_DRAFT is not None and command[1].endswith('draft_m3.py'):
        import contextlib
        with open(log, 'a') as out, contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                WARM_DRAFT(command[2:], env)
            except SystemExit as exc:
                if exc.code:
                    raise OpError('engine_failed', f'draft failed: {exc.code}') from exc
        return
    with open(log, 'a') as out:
        out.write('+ '+' '.join(command)+'\n')
        out.flush()
        done = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                              env=dict(os.environ, PYTHONPATH=str(ROOT), **(env or {})))
    if done.returncode:
        tail = Path(log).read_text(errors='replace').strip().splitlines()[-1:] or ['']
        raise OpError('engine_failed', f'{Path(command[2] if command[1] == "-m" else command[1]).name} exited with '
                                       f'{done.returncode}: {tail[0][:300]}')


# ------------------------------------------------------------------ charts
def parse_chart(path, mode):
    data = Path(path).read_bytes()
    name = Path(path).name
    if mode == 'mania_4k':
        from noteforge.events.osu_mania import parse_osu
        return parse_osu(data, name)
    if zipfile.is_zipfile(path):
        from noteforge.events.phira import load_package
        return load_package(path)
    from noteforge.events.phira import parse_phira
    return parse_phira(data, name)


def opening_shares(notes, mode, opening_us=OPENING_US):
    """Style shares of a chart's first 30 s from its first note. notes: (onset_us, type)."""
    notes = sorted(notes)
    if not notes:
        raise OpError('chart_empty', 'the chart has no notes')
    head = [n for n in notes if n[0] < notes[0][0]+opening_us]
    onsets = {}
    for on, _ in head:
        onsets[on] = onsets.get(on, 0)+1
    share = dict(hold=sum(t == 'hold' for _, t in head)/len(head), chord=sum(c >= 2 for c in onsets.values())/len(onsets))
    if mode == 'phira':
        share.update(flick=sum(t == 'flick' for _, t in head)/len(head), drag=sum(t == 'drag' for _, t in head)/len(head))
    return share


def inspect_chart(path, mode):
    try:
        chart = parse_chart(path, mode)
    except Exception as exc:  # noqa: BLE001 - any parser failure means this file is not usable
        raise OpError('chart_unreadable', f'not a readable {mode} chart ({type(exc).__name__}: {str(exc)[:200]})') from exc
    from noteforge.scripts.build_chart_timing import GimmickTiming, timing_segments
    segments, warnings = None, []
    if chart.timing is not None:
        try:
            segments = timing_segments(chart)[0]
        except GimmickTiming as exc:
            warnings.append(f'timing not usable as a model input: {exc}')
    else:
        warnings.append('the chart has no usable timing')
    notes = [(int(e.onset_us), e.type) for e in chart.events]
    bad = sorted({i.code for i in chart.quarantine_reasons()})
    if bad:
        warnings.append('parser flagged: '+', '.join(bad)[:200])
    return dict(mode=mode, notes=len(notes), first_note_ms=min(n[0] for n in notes)/1000 if notes else None,
                segments=segments, opening_style=opening_shares(notes, mode) if notes else None, warnings=warnings)


# ------------------------------------------------------------------ generation
def phase_us(onsets_us, bpm):
    """Position within a 1/4 beat of the strongest onset cluster."""
    import numpy as np
    onsets = np.asarray(onsets_us, float)
    step = 60e6/bpm/4
    hist, edges = np.histogram(onsets % step, bins=int(step/2000), range=(0, step))
    band = np.convolve(np.r_[hist, hist[:PHASE_BAND_BINS]], np.ones(PHASE_BAND_BINS), 'valid')[:len(hist)]
    return float(edges[int(np.argmax(band))]+PHASE_BAND_BINS*1000)


def most_agreeing(samples):
    """Index of the sample whose onsets agree most with the other samples (mean F1 at 20 ms)."""
    import numpy as np
    def f1_at(draft_on, ref_on, tol=20_000):
        a, b = np.unique(draft_on), np.unique(ref_on)
        i = j = tp = 0
        while i < len(a) and j < len(b):
            d = a[i]-b[j]
            if abs(d) <= tol:
                tp += 1
                i += 1
                j += 1
            elif d < 0:
                i += 1
            else:
                j += 1
        return 2.*tp/(len(a)+len(b)) if tp else 0.
    onsets = [np.array(sorted({n[0] for n in s}), np.int64) for s in samples]
    scores = [np.mean([f1_at(onsets[i], onsets[j]) for j in range(len(onsets)) if j != i]) if len(onsets) > 1 else 1.
              for i in range(len(onsets))]
    return int(np.argmax(scores))


def fake_samples(request, resolved):
    """Test engine: no model, deterministic notes. Selected only by NOTEFORGE_FAKE_ENGINE=1."""
    rng = random.Random(int(hashlib.sha256(f"{request['audio_id']}:{request['seed']}".encode()).hexdigest(), 16) % 2**32)
    duration = 60_000_000
    out = []
    for _ in range(request['samples']):
        notes, on = [], 1_000_000
        while on < duration-1_000_000:
            lane = -1 if request['mode'] == 'phira' else rng.randrange(4)
            hold = rng.random() < .15
            kind = 'hold' if hold else rng.choice(('tap', 'tap', 'tap', 'flick', 'drag')) if lane == -1 else 'tap'
            notes.append([on, on+400_000 if hold else on, kind, lane])
            if lane == -1 and kind == 'tap' and rng.random() < .3:  # chords, like real drafts
                notes.append([on, on, 'tap', lane])
            on += rng.choice((250_000, 500_000))
        out.append(notes)
    return dict(samples=out, vote=out[0], duration_us=duration, shift_us=0, codec='fake',
                timing_points=None, style=request.get('style') or {}, level=request.get('level'))


def m3_env(resolved, plan):
    """How the M3 path decodes: plan 'fast' is 2 rounds, 'balanced' adds a rewrite round, 'sequential' is core by core."""
    env = dict(M3_FAST=resolved['fast_weights'], M3_CACHE_MB='300')
    stride = dict(fast='2', balanced='-2').get(plan)
    if stride:
        env['M3_STRIDE'] = stride
    return env


def draft_command(py, request, resolved, curve_path, ratios, mert):
    mode, m3 = request['mode'], resolved.get('backend') == 'm3'
    common = ['--audio', resolved['audio_path'], '--mode', mode, '--features-dir', resolved['features_dir'], '--plan', resolved['plan'],
              '--seed', request['seed'], '--max-wall-s', resolved['max_wall_s'], '--field-temperature', FIELD_TEMPERATURE,
              '--history-reference', '--density-curve', curve_path, '--profile', 'none', '--kind-as-band', *ratios]
    if m3:
        return [py, ROOT/'noteforge/m3/draft_m3.py', *common, '--model', 'frozen', '--mel-stats', resolved['mel_stats'], '--device', 'cpu',
                '--batch-size', resolved['batch_size'], '--scheduler', 'grouped']
    return [py, '-m', 'noteforge.scripts.draft_song', *common, '--model', resolved['model'], '--device', resolved['device'],
            '--batch-size', resolved['batch_size'], '--scheduler', 'continuous', *mert]


def extract_features(resolved, draft):
    py = sys.executable
    draft = Path(draft)
    work = draft/'work'
    work.mkdir(exist_ok=True)
    progress(draft, 'features', 'reading the audio')
    extract = work/'features'
    if extract.exists():
        import shutil
        shutil.rmtree(extract)
    audio, features, log = resolved['audio_path'], resolved['features_dir'], draft/'engine.log'
    if resolved.get('backend') == 'm3':
        run([resolved.get('mert_python') or py, ROOT/'noteforge/m3/mert_fast.py', 'extract', resolved['fast_mert'], '--extract-only', '--audio', audio,
             '--features-dir', features, '--out-dir', extract, '--device', 'mps', '--mert-device', 'mps', '--mert-source', 'frozen'], log)
    else:
        mert = ['--mert-source', resolved['mert_source']] if resolved.get('mert_source') else []
        run([resolved.get('mert_python') or py, '-m', 'noteforge.scripts.draft_song', '--extract-only', '--audio', audio, '--features-dir', features,
             '--out-dir', extract, '--device', resolved['device'], *mert], log)


def real_samples(request, resolved, draft, features_ready=False):
    py = sys.executable
    draft = Path(draft)
    work = draft/'work'
    work.mkdir(exist_ok=True)
    log = draft/'engine.log'
    audio, features, mode = resolved['audio_path'], Path(resolved['features_dir']), request['mode']
    sha = request['audio_id']
    mert = ['--mert-source', resolved['mert_source']] if resolved.get('mert_source') else []

    m3 = resolved.get('backend') == 'm3'
    if not features_ready:
        extract_features(resolved, draft)

    progress(draft, 'planning', 'planning density and style')
    curve_path = work/'curve.json'
    command = [py, ROOT/'noteforge/planner/plan_song.py', '--features-dir', features/'features', '--audio-sha', sha, '--mode', mode,
               '--planner-dir', resolved['planner_dir'], '--device', 'cpu' if m3 else resolved['device'], '--out', curve_path]
    if request.get('level') is not None:
        command += ['--density', request['level']]
    run(command, log)
    planned = json.loads(Path(log).read_text(errors='replace').strip().splitlines()[-1])
    if request.get('density_curve'):
        write_json(curve_path, request['density_curve'])

    style = planned['suggested_style'] if mode == 'phira' else {k: v for k, v in planned['suggested_style'].items()
                                                                if k.startswith(('hold', 'chord'))}
    given = dict(resolved.get('opening_style') or {})
    given.update({k: v for k, v in (request.get('style') or {}).items() if v is not None})
    ratios, used_style = [], {}
    for field, value in style.items():
        name = field.split('_')[0]
        used_style[name] = given[name] if given.get(name) is not None else value
        ratios += [f'--{name}-ratio', str(used_style[name])]

    samples = request['samples']
    base = draft_command(py, request, resolved, curve_path, ratios, mert)
    env = m3_env(resolved, request.get('plan') or 'balanced') if m3 else None
    (draft/'draft_base.json').write_text(json.dumps(dict(command=[str(x) for x in base], env=env)))
    final = [*base, '--out-dir', work/'draft', '--consensus', samples, '--consensus-pattern', 'anchor', '--save-samples']
    if samples > 1:
        final += ['--consensus-votes', max(1, round(samples*VOTES_OF_8/8))]
    timing = resolved.get('timing') or {}
    timing_flags = []
    if timing.get('osu_path'):
        timing_flags = ['--timing', timing['osu_path']]
    elif timing.get('segments'):
        path = work/'segments.json'
        write_json(path, timing['segments'])
        timing_flags = ['--timing-segments', path]
    elif timing.get('bpm'):
        progress(draft, 'timing', 'finding the beat phase')
        first = work/'first_pass'
        run([*base, '--out-dir', first, '--consensus', 1 if m3 else 4, *([] if m3 else ['--consensus-pattern', 'anchor', '--consensus-votes', 2])], log, env)
        events = json.loads((first/'draft-events.json').read_text())['events']
        if not events:
            raise OpError('no_notes', 'the first pass produced no notes, so the beat phase is unknown')
        beat = 60e6/timing['bpm']
        phase = phase_us([e['onset_us'] for e in events], timing['bpm']) % beat
        timing_flags = ['--beat-grid', f'{phase:.3f},{beat:.6f}']
        timing = dict(timing, grid_start_us=phase)
    final += timing_flags
    progress(draft, 'generating', f'generating {samples} samples')
    (draft/'draft_timing.json').write_text(json.dumps([str(x) for x in timing_flags]))
    run(final, log, env)

    progress(draft, 'assembling', 'building the first revision')
    receipt = json.loads((work/'draft/draft-receipt.json').read_text())
    raw = json.loads((work/'draft/draft-samples.json').read_text())['samples']
    events = json.loads((work/'draft/draft-events.json').read_text())['events']
    shift = int(receipt['clock']['shift_us'])
    move = lambda notes: [[n[0]-shift, n[1]-shift, n[2], n[3]] for n in notes if n[0]-shift >= 0]  # noqa: E731
    vote = [[e['onset_us'], e['end_us'], e['type'], e['lane']] for e in events]
    points = None
    if timing.get('segments'):
        points = [[s/1000, 60e6/b] for s, b, _ in timing['segments']]
    if timing.get('grid_start_us') is not None:
        points = [[(timing['grid_start_us']-shift)/1000, timing['bpm']]]
    return dict(samples=[move(s) for s in raw], vote=move(vote), duration_us=int(receipt['audio']['duration_us']), shift_us=shift,
                codec=receipt['audio']['codec'], timing_points=points, style=used_style,
                level=request.get('level') if request.get('level') is not None else planned.get('mean_density'))


def generate(draft, features_ready=False):
    draft = Path(draft)
    request = json.loads((draft/'request.json').read_text())
    resolved = json.loads((draft/'resolved.json').read_text())
    if os.environ.get('NOTEFORGE_FAKE_ENGINE') == '1':
        made = fake_samples(request, resolved)
    else:
        made = real_samples(request, resolved, draft, features_ready)
    samples = made.pop('samples')
    vote = made.pop('vote')
    write_json(draft/'samples.json', dict(samples=samples, vote=vote))
    write_json(draft/'meta.json', made)
    aggregation = request['aggregation']
    if aggregation == 'vote':
        base = dict(kind='vote')
    elif aggregation == 'most_agreeing':
        base = dict(kind='sample', sample=most_agreeing(samples))
    else:
        base = dict(kind='sample', sample=0)
    assemble(draft, dict(revision=0, base=base, choices=[]))
    progress(draft, 'done')


# ------------------------------------------------------------------ revisions
def layout_phira(events, seed):
    import torch
    from noteforge.placement.apply import layout
    from noteforge.placement.model import M
    net = M()
    net.load_state_dict(torch.load(ROOT/'noteforge/placement/placement3.pt', map_location='cpu'))
    net.eval()
    return layout(events, net, T=1., seed=seed)


def assemble(draft, spec):
    """Notes of the base, with each chosen section replaced by a sample or by the caller's own notes.

    Caller's own notes are kept as they are: they are not removed by the clean-up, and on Phira keep their position."""
    from noteforge.inference.hold_release import repair_hold_release
    draft = Path(draft)
    request = json.loads((draft/'request.json').read_text())
    meta = json.loads((draft/'meta.json').read_text())
    stored = json.loads((draft/'samples.json').read_text())
    mode, samples = request['mode'], stored['samples']
    base = spec['base']
    if base['kind'] == 'vote':
        notes = stored['vote']
    elif base['kind'] == 'notes':
        notes = base.pop('notes')
    else:
        if not 0 <= base['sample'] < len(samples):
            raise OpError('bad_sample', f"sample {base['sample']} does not exist")
        notes = samples[base['sample']]
    notes = [tuple(n) for n in notes]
    locked = []
    ordered = sorted(spec['choices'], key=lambda c: c['start_us'])
    for a, b in zip(ordered, ordered[1:]):
        if b['start_us'] < a['end_us']:
            raise OpError('choices_overlap', 'section choices overlap')
    for choice in ordered:
        start, end = choice['start_us'], choice['end_us']
        notes = [n for n in notes if not start <= n[0] < end]
        if choice.get('notes') is not None:
            locked += choice['notes']
        else:
            if not 0 <= choice['sample'] < len(samples):
                raise OpError('bad_sample', f"sample {choice['sample']} does not exist")
            notes += [tuple(n) for n in samples[choice['sample']] if start <= n[0] < end]
    warnings = []
    profile = request['profile']
    if mode == 'phira':
        events = [dict(onset_us=int(n[0]), end_us=int(n[1]), type=n[2], lane=-1) for n in notes]
        if profile != 'none':
            from noteforge.events.phira_postprocess import playable
            events = playable(events, profile)[0]
        for n in locked:
            events.append(dict(onset_us=int(n['onset_us']), end_us=int(n['end_us']), type=n['type'], lane=-1, x=n.get('x')))
        placed = layout_phira(events, request['seed'])
        own = {}  # notes of one chord share (time, type): keep one position per note
        for e in events:
            if e.get('x') is not None:
                own.setdefault((e['onset_us'], e['type']), []).append(e['x'])
        output = []
        for p in placed:
            kept_x = own.get((p['t_us'], p['type']))
            output.append(dict(onset_us=p['t_us'], end_us=p['end_us'], type=p['type'], x=kept_x.pop(0) if kept_x else p['x']))
    else:
        own = [(int(n['onset_us']), int(n['end_us']), n['type'], int(n['lane'])) for n in locked]
        # Generated notes give way to the charter's: nothing may start while a charter's note on that key is held.
        notes = [n for n in notes if not any(n[3] == o[3] and o[0] <= n[0] < o[1]+40_000 for o in own)]
        tuples = notes+own
        repaired = repair_hold_release(sorted(set(tuples)), mode, min_release_us=40_000)[0]
        output = [dict(onset_us=a, end_us=b, type=t, lane=l) for a, b, t, l in repaired]
        if os.environ.get('NOTEFORGE_FAKE_ENGINE') != '1':
            from noteforge.scripts.runtime import consensus_legality
            issues = consensus_legality([(a+meta['shift_us'], b+meta['shift_us'], t, l) for a, b, t, l in repaired],
                                         mode, meta['duration_us'])
            if issues:
                warnings.append(f'{len(issues)} note conflicts remain, for example {str(issues[0])[:150]}')
    level = meta.get('level')
    if level and output and not spec['choices']:
        achieved = len(output)/(meta['duration_us']/1e6)
        if abs(achieved/level-1) > LEVEL_TOLERANCE:
            warnings.append(f'density is {achieved:.2f} notes/s, the request was {level:.2f}; adjust level and generate again')
    output.sort(key=lambda e: (e['onset_us'], e.get('lane', 0), e.get('x', 0), e['type'], e['end_us']))
    for i, e in enumerate(output):
        e['id'] = f'n{i}'
    folder = draft/'revisions'/str(spec['revision'])
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'events.json', dict(revision=spec['revision'], base=base, choices=[
        {k: v for k, v in c.items() if k != 'notes'} | dict(source='notes' if c.get('notes') is not None else 'sample')
        for c in ordered], warnings=warnings, events=output))
    return output


# ------------------------------------------------------------------ export
def export(draft, revision, form, speed, out):
    draft, out = Path(draft), Path(out)
    request = json.loads((draft/'request.json').read_text())
    resolved = json.loads((draft/'resolved.json').read_text())
    meta = json.loads((draft/'meta.json').read_text())
    events = json.loads((draft/'revisions'/str(revision)/'events.json').read_text())['events']
    title, artist = request.get('title') or 'NoteForge draft', request.get('artist') or 'unknown'
    audio = Path(resolved['audio_path'])
    audio_name = 'audio'+audio.suffix.lower()
    temporary = out.with_name(out.name+'.part')
    temporary.unlink(missing_ok=True)
    if request['mode'] == 'phira':
        import yaml
        from noteforge.events.phira import parse_phira
        from noteforge.events.phira_rpe import plain_background, rpe
        placed = [dict(id=e['id'], t_us=e['onset_us'], end_us=e['end_us'], type=e['type'], x=e['x']) for e in events]
        doc = rpe(placed, 'NoteForge draft', speed=speed)
        doc['META'].update(name=title, composer=artist, song=audio_name, charter='NoteForge')
        raw = json.dumps(doc, ensure_ascii=False).encode()
        parsed = parse_phira(raw)
        if parsed.quarantined or len(parsed.events) != len(events):
            raise OpError('export_check_failed', 'the written chart does not read back the same')
        if form == 'rpe_json':
            temporary.write_bytes(raw)
        else:
            info = dict(name=title, composer=artist, level='NoteForge draft', difficulty=0., charter='NoteForge', illustrator='Plain background',
                        chart='chart.json', music=audio_name, illustration='background.png', offset=0., aspectRatio=16/9,
                        backgroundDim=.5, lineLength=6., intro='AI draft')
            with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
                z.writestr('chart.json', raw)
                z.writestr('info.yml', yaml.safe_dump(info, allow_unicode=True))
                z.writestr('background.png', plain_background())
                z.write(audio, audio_name)
    else:
        from noteforge.events.chart_export import write_osu_mania
        osu = temporary.with_name(temporary.name+'.osu')
        osu.unlink(missing_ok=True)
        rows = [dict(onset_us=e['onset_us'], end_us=e['end_us'], type=e['type'], lane=e['lane']) for e in events]
        points = meta.get('timing_points')
        write_osu_mania(osu, rows, title=title, artist=artist, version='NoteForge draft', creator='NoteForge (model draft, unreviewed)',
                        audio_filename=audio_name, timing_points=[tuple(p) for p in points] if points else None)
        if form == 'osu':
            osu.replace(temporary)
        else:
            with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
                z.write(osu, 'chart.osu')
                z.write(audio, audio_name)
            osu.unlink()
    temporary.replace(out)


# ------------------------------------------------------------------ continue
def continue_draft(draft, spec):
    """Keep the notes before the cut, write the rest. Features are reused, so no audio model is loaded."""
    draft = Path(draft)
    meta = json.loads((draft/'meta.json').read_text())
    request = json.loads((draft/'request.json').read_text())
    previous = json.loads((draft/'revisions'/str(spec['from_revision'])/'events.json').read_text())['events']
    cut, shift = spec['cut_us'], meta['shift_us']
    kept = [e for e in previous if e['onset_us'] < cut]
    if os.environ.get('NOTEFORGE_FAKE_ENGINE') == '1':
        rng = random.Random(spec['seed'])
        notes, on = [], cut
        while on < meta['duration_us']-500_000:
            lane = -1 if request['mode'] == 'phira' else rng.randrange(4)
            notes.append([on, on, 'tap', lane])
            on += 500_000
    else:
        base = json.loads((draft/'draft_base.json').read_text())
        timing_flags = json.loads((draft/'draft_timing.json').read_text())
        out = draft/'work'/f"continue-{spec['revision']}"
        spec_file = out.with_suffix('.json')
        initial = [dict(onset_us=e['onset_us']+shift, end_us=e['end_us']+shift, type=e['type'], lane=e.get('lane', -1)) for e in kept]
        write_json(spec_file, dict(events=initial, keep_before_us=cut+shift))
        command = base['command']+timing_flags+['--out-dir', out, '--consensus', 1, '--save-samples']
        command = [str(x) for x in command]
        at = command.index('--seed')
        command[at+1] = str(spec['seed'])
        if not base.get('env'):
            raise OpError('unsupported', 'continuing is available on the m3 backend')
        # M3_CONTINUE goes through the rounds rollout: 'sequential' continues in 2 rounds too.
        env = dict(base['env'], M3_CONTINUE=str(spec_file), M3_STRIDE=base['env'].get('M3_STRIDE', '2'))
        run(command, draft/'engine.log', env)
        written = json.loads((out/'draft-samples.json').read_text())['samples'][0]
        notes = [[n[0]-shift, n[1]-shift, n[2], n[3]] for n in written if n[0]-shift >= cut]
    own = [dict(onset_us=e['onset_us'], end_us=e['end_us'], type=e['type'], lane=e.get('lane', -1), x=e.get('x')) for e in kept]
    done = assemble(draft, dict(revision=spec['revision'], base=dict(kind='notes', notes=notes, from_revision=spec['from_revision'], cut_us=cut),
                                choices=[dict(start_us=0, end_us=cut, notes=own)]))
    return done


def main(argv):
    command = argv[0]
    try:
        if command == 'inspect-chart':
            print(json.dumps(inspect_chart(argv[1], argv[2])))
        elif command == 'features':
            extract_features(json.loads((Path(argv[1])/'resolved.json').read_text()), argv[1])
        elif command == 'generate':
            generate(argv[1], features_ready=argv[2:] == ['features-ready'])
        elif command == 'assemble':
            assemble(argv[1], json.loads(Path(argv[2]).read_text()))
        elif command == 'continue':
            continue_draft(argv[1], json.loads(Path(argv[2]).read_text()))
        elif command == 'export':
            export(argv[1], int(argv[2]), argv[3], float(argv[4]), argv[5])
        else:
            raise OpError('bad_command', command)
    except OpError as exc:
        print(json.dumps(dict(error=dict(code=exc.code, message=exc.message))))
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
