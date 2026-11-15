"""Inference runtime helpers for NoteForge: decoding, consensus aggregation and chart export.

This module contains no training loop, corpus access, evaluation panel or benchmark CLI.
It is imported by ``noteforge.scripts.draft_song`` and ``noteforge_api.ops`` to
generate and export drafts.
"""
from collections import defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np

from noteforge.model.full_event_generate import merge_stats
from noteforge.inference.hold_release import stable_rank

PANEL_SEED = 20261102
CONSENSUS_WINDOW_US = 30_000
UNPLAYABLE_REPEAT_US = 40_000
FIELD_TEMPERATURE_KEYS = ('group', 'head_block', 'head_cell', 'head_ms', 'mania', 'member', 'kind',
                          'tail_block', 'tail_cell', 'tail_ms')
def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def make_generator(temperature, top_p, seed, field_temperature=None, decode_options=None, *, onset_fusion=None,
                   guidance=None, stats=None, on_step=None):
    """Wrapper around full_event_generate.generate_batch with per-row seeds (batch-composition independent).

    ``row_seeds`` (passed by local_song_rollout.generate_complete_song_samples) gives each row its own
    sample seed; the per-row generator seed is the same function of (seed, mode, duration, core start).
    ``decode_options`` (e.g. fast=False / cuda_graph=False) are forwarded only when set; so are
    ``onset_fusion`` (weight -> generate_batch onset_fusion=dict(weight=..)) and ``guidance`` (CFG
    weight, applied only to batches for which the rollout passes ``unconditional_inputs``). ``stats`` (dict)
    accumulates the decode counters over all batches. ``generator.continuous`` / ``generator.row_generators`` are
    the same decoding settings and the same seed function for scheduler='continuous'."""
    import inspect
    import torch
    options = dict(decode_options or {})
    if onset_fusion:
        options['onset_fusion'] = dict(weight=float(onset_fusion))

    def row_generators(seeds, requests, device):
        gen_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
        return [torch.Generator(device=gen_device).manual_seed(int(stable_rank(
            s, r.mode, r.duration_us, r.start_us)[:15], 16)) for s, r in zip(seeds, requests)]

    def generator(model, inputs, requests, max_tokens, *, deadline=None, row_seeds=None, unconditional_inputs=None,
                  memory_index=None):
        from noteforge.model.full_event_generate import generate_batch, merge_stats
        extra = (dict(guidance=float(guidance), unconditional_inputs=unconditional_inputs)
                 if guidance is not None and unconditional_inputs is not None else {})
        if memory_index is not None:
            extra['memory_index'] = memory_index
        if on_step is not None:
            extra['on_step'] = on_step
        batch_stats = {}
        if stats is not None:
            extra['stats'] = batch_stats
        if temperature is None:
            out = generate_batch(model, inputs, requests, max_tokens, deadline=deadline, **options, **extra)
        else:
            if 'temperature' not in inspect.signature(generate_batch).parameters:
                raise RuntimeError('generate_batch has no sampling support; use --temperature none (greedy)')
            seeds = row_seeds if row_seeds is not None else [seed] * len(requests)
            gens = row_generators(seeds, requests, inputs['local'].device)
            if field_temperature:
                extra['field_temperature'] = field_temperature
            out = generate_batch(model, inputs, requests, max_tokens, deadline=deadline,
                                 temperature=float(temperature), top_p=top_p, generator=gens, **options, **extra)
        if stats is not None:
            merge_stats(stats, batch_stats)
        return out
    generator.row_generators = row_generators
    if options.get('fast', True):
        generator.continuous = dict({k: v for k, v in options.items() if k != 'fast'},
                                    temperature=None if temperature is None else float(temperature), top_p=top_p,
                                    field_temperature=field_temperature, guidance=guidance, stats=stats,
                                    on_step=on_step)
    return generator


def decode_options(args):
    """generate_batch speed switches from the CLI (empty = module defaults: fast path, CUDA graphs on CUDA)."""
    options = {}
    if getattr(args, 'reference_decode', False):
        options['fast'] = False
    if getattr(args, 'no_cuda_graph', False):
        options['cuda_graph'] = False
    constraints = {k: v for k, v in (('max_chord', getattr(args, 'max_chord', None)),
                                     ('min_same_lane_ms', getattr(args, 'min_jack_ms', None))) if v is not None}
    if constraints:
        options['pattern_constraints'] = constraints
    if getattr(args, 'autocast', None):
        options['autocast'] = args.autocast
    if getattr(args, 'compile_decode', False):
        options['compile_decode'] = True
    if getattr(args, 'kind_bias', None):
        options['kind_bias'] = args.kind_bias
    return options
def parse_temperature(value):
    return None if value in (None, 'none', 'greedy') else float(value)
def draft_tuples(song, state):
    return [(int(e['onset_us']), int(e['end_us']), e['type'], int(e['lane']) if song['mode'] == 'mania_4k' else -1)
            for e in state['events']]
CONSENSUS_WINDOW_US = 30_000
def _lower_median(values):
    values = sorted(values)
    return values[(len(values)-1)//2]


def _onset_groups(notes, mode):
    """One sample's notes -> [(onset, pattern, tails)] per distinct onset.

    mania: pattern = ((lane, type), ...) sorted by lane, tails = {lane: end};
    phira: pattern = sorted type multiset, tails = sorted hold ends."""
    by_onset = defaultdict(list)
    for n in notes:
        by_onset[int(n[0])].append(n)
    out = []
    for onset in sorted(by_onset):
        members = by_onset[onset]
        if mode == 'mania_4k':
            pattern = tuple(sorted((int(n[3]), n[2]) for n in members))
            tails = {int(n[3]): int(n[1]) for n in members if n[2] == 'hold'}
        else:
            pattern = tuple(sorted(n[2] for n in members))
            tails = sorted(int(n[1]) for n in members if n[2] == 'hold')
        out.append((onset, pattern, tails))
    return out


UNPLAYABLE_REPEAT_US = 40_000
def _without_weak_fragments(clusters, need, half):
    """Drop a kept onset that sits alone in a squeezed gap (a gap under .6 of the local median and under 200 ms, with
    neighbouring gaps that are not short) when fewer than half of the samples wrote it. Fast runs stay, and so do squeezes
    most samples agree on; a lone squeeze few samples wrote reads as noise (2026-11-05 owner playtest)."""
    kept = [c for c in clusters if len(c) >= need]
    if len(kept) < 20:
        return clusters
    t = [sorted(m[0] for m in c)[(len(c)-1)//2] for c in kept]
    gaps = [b-a for a, b in zip(t, t[1:])]
    med = [sorted(gaps[max(0, i-8):i+9])[len(gaps[max(0, i-8):i+9])//2] for i in range(len(gaps))]
    short = [g < .6*m and g < 200_000 for g, m in zip(gaps, med)]
    drop = set()
    for i, s in enumerate(short):
        lone = s and not (i > 0 and short[i-1]) and not (i+1 < len(short) and short[i+1])
        if lone:
            a, b = kept[i], kept[i+1]
            weaker = i if len(a) <= len(b) else i+1
            if len(kept[weaker]) < half:
                drop.add(weaker)
    return [c for i, c in enumerate(kept) if i not in drop]


def consensus_draft(samples, mode, duration_us, *, window_us=CONSENSUS_WINDOW_US, min_jack_us=None,
                    pattern_source='majority', min_votes=None, onset_source='clusters', fragment_guard=False):
    """D6 MBR-style consensus of N whole-song drafts (lists of (onset, end, type, lane) tuples).

    1. Every sample's distinct onsets (mania: lane-agnostic onset = one chord) are pooled
       and swept in time order (ties: lower sample index); a cluster starts at its first
       onset and takes later onsets within ``window_us`` of that first onset, at most one
       per sample (a second onset of the same sample starts the next cluster).
    2. Clusters with members from >= ``min_votes`` (default ceil(N/2)) samples are kept; onset = lower median of
       the member onsets (always a sampled ms). A kept cluster whose onset is not after
       the previous kept onset is skipped (cannot happen unless windows touch).
    3. Pattern (mania: the lane->type chord; phira: the type multiset) = majority among
       members; ties -> the pattern of the earliest sample. Hold tail = lower median of the
       tails of the members with that pattern (per lane / per sorted hold rank), clamped
       to the song; a hold whose tail is not after the onset becomes a tap.
    4. mania: a hold that would contain a later same-lane head is dropped (its head kept
       as a tap), so the draft has no same-lane overlap / head-inside-hold.
    5. ``min_jack_us`` (mania, default None = off): per-cluster majority ignores the neighbours,
       so samples that alternate keys with opposite parity vote same-key repeats no sample
       made (consensus 8 had 2.6x the single-sample rate of same-key repeats < 100 ms). With it
       set, the pattern is the most-voted member pattern that presses no key pressed less than
       ``min_jack_us`` earlier; if every candidate does, the majority is kept (``jack_relaxed``).
    6. ``pattern_source='anchor'``: the onsets are still the majority clusters, but the PATTERN of a kept cluster
       is the anchor sample's (the medoid: most kept-cluster members sharing its patterns) whenever the
       anchor has a member there; majority otherwise. Per-cluster majority stitches lanes from different samples
       (same-key repeats no sample made, rolls broken) and splits hold-vs-tap votes (LNs lost); one sample's
       patterns keep one coherent hand flow. ``min_jack_us`` still applies (the anchor pattern is ranked first).
    7. ``onset_source='anchor'`` (needs pattern_source='anchor'): only clusters holding the anchor sample survive and they
       keep its own onset, so the rhythm is one sample's and the vote only removes its unsupported notes. Pooling onsets
       of different samples stitches different rhythms into bursts no sample wrote.
    Returns (notes, info). Never reads a human chart.
    """
    n = len(samples)
    if n < 1:
        raise ValueError('consensus needs >= 1 sample')
    need = math.ceil(n/2) if min_votes is None else int(min_votes)
    items = sorted((onset, k, pattern, tails) for k, notes in enumerate(samples)
                   for onset, pattern, tails in _onset_groups(notes, mode))
    clusters, current = [], None
    for item in items:
        if current is not None and item[0]-current[0][0] <= window_us and item[1] not in {m[1] for m in current}:
            current.append(item)
        else:
            current = [item]
            clusters.append(current)
    if pattern_source not in ('majority', 'anchor'):
        raise ValueError(f'pattern_source {pattern_source!r}')
    if onset_source == 'anchor' and pattern_source != 'anchor':
        raise ValueError('onset_source anchor needs pattern_source anchor')
    anchor, from_anchor = None, 0
    if pattern_source == 'anchor':
        # medoid: per kept cluster a sample scores the number of members sharing its pattern (presence x agreement)
        score = defaultdict(int)
        for members in clusters:
            if len(members) >= need:
                votes = defaultdict(int)
                for m in members:
                    votes[m[2]] += 1
                for m in members:
                    score[m[1]] += votes[m[2]]
        anchor = min(score, key=lambda k: (-score[k], k)) if score else None
    if fragment_guard:
        clusters = _without_weak_fragments(clusters, need, math.ceil(n/2))
    notes, kept, skipped, last_onset, dropped, short = [], 0, 0, None, 0, 0
    lane_hold = {}  # lane -> index in notes of the active hold
    last_press, rechosen, relaxed, relaned, dropped_jack = {}, 0, 0, 0, 0
    for members in clusters:
        if len(members) < need:
            continue
        if onset_source == 'anchor':
            own = [m for m in members if m[1] == anchor]
            if not own:
                continue
            onset = own[0][0]
        else:
            onset = _lower_median([m[0] for m in members])
        if last_onset is not None and onset <= last_onset or not 0 <= onset < duration_us:
            skipped += 1
            continue
        counts = defaultdict(int)
        for m in members:
            counts[m[2]] += 1
        best = max(counts.values())
        pattern = min((m for m in members if counts[m[2]] == best), key=lambda m: m[1])[2]
        anchor_pattern = next((m[2] for m in members if m[1] == anchor), None) if anchor is not None else None
        if anchor_pattern is not None:
            pattern, from_anchor = anchor_pattern, from_anchor + 1
        tail_source = None  # mania re-laned keys: lane -> the voted lane whose tails it carries
        if mode == 'mania_4k' and min_jack_us:
            first_vote = {}
            for m in members:
                first_vote.setdefault(m[2], m[1])
            ranked = sorted(counts, key=lambda pat: (pat != anchor_pattern, -counts[pat], first_vote[pat]))
            ok = [pat for pat in ranked if all(onset - last_press.get(lane, -10**12) >= min_jack_us for lane, _ in pat)]
            if ok and ok[0] != pattern:
                pattern, rechosen = ok[0], rechosen + 1
            elif not ok:
                # No voted pattern avoids the repeat: move each offending key to the free key pressed longest
                # ago (not in the chord, not held at this onset, not itself a repeat); if none, drop that
                # member of a multi-key chord; a lone unmovable note is kept (counted as relaxed).
                busy = {lane for lane, _ in pattern}
                held = {lane for lane, i in lane_hold.items() if notes[i][1] > onset}
                fixed, tail_source, voted = [], {}, pattern
                for lane, kind in sorted(pattern):
                    if onset - last_press.get(lane, -10**12) >= min_jack_us:
                        fixed.append((lane, kind))
                        tail_source[lane] = lane
                        continue
                    free = [l for l in range(4) if l not in busy and l not in held
                            and onset - last_press.get(l, -10**12) >= min_jack_us]
                    if free:
                        new = min(free, key=lambda l: (last_press.get(l, -10**12), l))
                        busy.discard(lane)
                        busy.add(new)
                        fixed.append((new, kind))
                        tail_source[new] = lane
                        relaned += 1
                    elif len(pattern) > 1:
                        busy.discard(lane)
                        dropped_jack += 1
                    else:
                        fixed.append((lane, kind))
                        tail_source[lane] = lane
                        relaxed += 1
                pattern = tuple(sorted(fixed))
        voters = [m for m in members if m[2] == (voted if tail_source is not None else pattern)]
        if mode == 'mania_4k':
            group = []
            for lane, kind in pattern:
                end = onset
                if kind == 'hold':
                    src = lane if tail_source is None else tail_source[lane]
                    end = min(_lower_median([m[3][src] for m in voters]), int(duration_us))
                    if end <= onset:
                        kind, end, short = 'tap', onset, short+1
                group.append((onset, end, kind, lane))
            for lane in {g[3] for g in group}:
                held = lane_hold.pop(lane, None)
                if held is not None and notes[held][1] > onset:  # earlier hold would contain this head
                    a, _, _, l = notes[held]
                    notes[held] = (a, a, 'tap', l)
                    dropped += 1
            for g in group:
                if g[2] == 'hold':
                    lane_hold[g[3]] = len(notes)
                notes.append(g)
                last_press[g[3]] = onset
        else:
            rank = 0
            for kind in pattern:
                end = onset
                if kind == 'hold':
                    end = min(_lower_median([m[3][rank] for m in voters]), int(duration_us))
                    rank += 1
                    if end <= onset:
                        kind, end, short = 'tap', onset, short+1
                notes.append((onset, end, kind, -1))
        kept += 1
        last_onset = onset
    notes.sort(key=lambda x: (x[0], x[3], x[2]))
    unplayable = 0
    if mode == 'mania_4k' and min_jack_us:
        # Humans never press one key twice within UNPLAYABLE_REPEAT_US; a lone note step 5 could not move is dropped.
        playable, pressed = [], {}
        for note in notes:
            if note[0] - pressed.get(note[3], -10**12) < UNPLAYABLE_REPEAT_US:
                unplayable += 1
                continue
            playable.append(note)
            pressed[note[3]] = note[0]
        notes = playable
    info = dict(samples=n, min_votes=need, window_us=window_us, clusters=len(clusters), kept_clusters=kept,
                skipped_clusters=skipped, notes=len(notes), holds_dropped_for_overlap=dropped,
                holds_too_short=short, unplayable_repeats_dropped=unplayable)
    info.update(pattern_source=pattern_source)
    if anchor is not None:
        info.update(anchor_sample=int(anchor), patterns_from_anchor=from_anchor)
    if min_jack_us:
        info.update(min_jack_us=int(min_jack_us), jack_rechosen=rechosen, jack_relaned=relaned,
                    jack_dropped=dropped_jack, jack_relaxed=relaxed)
    return notes, info


def consensus_legality(notes, mode, duration_us):
    """local_song_rollout.independent_legality of a consensus draft (must be empty)."""
    from noteforge.model.event_canvas import RhythmEvent
    from noteforge.model.local_song_rollout import independent_legality
    return independent_legality([RhythmEvent(a, b, t, l) for a, b, t, l in notes], mode, duration_us)
def preset_condition_provider(panel, presets, *, cell_us):
    """condition_provider: every core requests the constant train-median density of the song's
    primary reference difficulty tier (charter_difficulty_preset). Reads only the tier label."""
    from noteforge.inference.conditions import condition_inputs
    tier = {}
    for song in panel['songs']:
        ref = primary_reference(song)
        tier[song['song_id']] = (song['mode'], coarse_stratum(song['mode'], ref.get('stratum')))

    def provider(state, request):
        key = tier[state['chart_id']]
        density = float(presets[key])
        segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=density)]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_difficulty_preset', tier=key[1], density=density))
    return provider
def parse_control_bands(specs):
    """--control-band FIELD=LO,HI (repeatable) -> {field: [lo, hi]}; bands must be table bands of
    noteforge.events.controls (the published condition encoder only knows those)."""
    from noteforge.events import controls as C
    out = {}
    for spec in specs or ():
        field, _, rng = spec.partition('=')
        try:
            band = [float(x) for x in rng.split(',')]
        except ValueError:
            band = None
        if field not in C.BAND_EDGES or not band or not C.is_table_band(field, band):
            raise SystemExit(f'--control-band {spec!r}: need FIELD=LO,HI with a table band of {field!r} '
                             f'(edges {C.BAND_EDGES.get(field)})')
        out[field] = band
    return out
FIELD_TEMPERATURE_KEYS = ('group', 'head_block', 'head_cell', 'head_ms', 'mania', 'member', 'kind', 'tail_block',
                          'tail_cell', 'tail_ms')
def write_audio(audio, dest):
    """Copy resolved audio to ``dest`` (umask-respecting default permissions)."""
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    if 'path' in audio:
        shutil.copyfile(audio['path'], dest)
    else:
        dest.write_bytes(audio['data'])
    return dest
def verify_osu_on_disk(path, events):
    """Re-read written .osu bytes with the repo parser; the rhythm must survive exactly (ms grid)."""
    from noteforge.events.chart_export import osu_mania_events
    from noteforge.events.osu_mania import parse_osu
    parsed = parse_osu(Path(path).read_bytes(), Path(path).name)
    want = sorted((e.onset_us, e.end_us, e.type, e.location.lane) for e in osu_mania_events(events)[0])
    got = sorted((e.onset_us, e.end_us, e.type, e.location.lane) for e in parsed.events)
    return dict(parsed=True, quarantined=bool(parsed.quarantined), notes=len(got), exact=got == want and not parsed.quarantined)


def write_osz(path, files):
    """osu! beatmap archive (a zip of the beatmap folder); double-click imports into osu!."""
    import zipfile
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(f).name)
    return path
def export_song(folder, stem, song, events, audio, *, title, artist, version, creator='NoteForge (model draft, unreviewed)',
                timing_points=None, bookmarks_ms=None, preview_time_ms=None, clock_shift_us=0, clock_source=None):
    """One editor-openable chart (+ audio) into ``folder``. Returns a receipt; never raises on export errors.

    Drafts live on the MODEL clock (tape targets = raw chart + model_shift_us); editors play the audio on the
    raw chart clock, so every time is written as t - ``clock_shift_us`` (notes that would fall before 0 are
    dropped and counted). ``timing_points`` / bookmarks / preview must already be on the playback clock."""
    from noteforge.events.chart_export import write_osu_mania
    from noteforge.events.osu_mania import ExportError
    folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(audio={k: v for k, v in audio.items() if k not in ('data', 'path')})
    shift = int(clock_shift_us or 0)
    moved = [dict(e, onset_us=int(e['onset_us']) - shift, end_us=int(e['end_us']) - shift) for e in events]
    events = [e for e in moved if e['onset_us'] >= 0]
    receipt['clock'] = dict(written='playback (raw chart) clock = model clock - shift', shift_us=shift,
                            source=clock_source, dropped_before_zero=len(moved) - len(events))
    audio_name = f"audio{audio['ext']}" if audio['ok'] else None
    if audio['ok']:
        write_audio(audio, folder/audio_name)
    try:
        if song['mode'] == 'mania_4k':
            osu = folder/f'{stem}.osu'
            receipt['chart'] = write_osu_mania(osu, events, title=title, artist=artist, version=version, creator=creator,
                audio_filename=audio_name or f"{song['audio_sha256']}.mp3", timing_points=timing_points,
                bookmarks_ms=bookmarks_ms, preview_time_ms=preview_time_ms)
            receipt['disk_roundtrip'] = verify_osu_on_disk(osu, events)
            if audio['ok']:
                receipt['osz'] = str(write_osz(folder.parent/f'{stem}.osz', [osu, folder/audio_name]))
        elif audio['ok']:
            from noteforge.events.rhythm_preview import write_phira_preview
            receipt['chart'] = write_phira_preview(folder.parent/f'{stem}.zip', events, folder/audio_name,
                                                   title=title, artist=artist, audio_name=audio_name)
            receipt['chart']['path'] = str(folder.parent/f'{stem}.zip')
        else:
            # No audio -> no Phira package; still write the rhythm as an RPE chart json.
            from noteforge.events.rhythm_preview import display_positions
            from noteforge.events.phira_rpe import rpe
            doc = rpe(display_positions(events), 'DRAFT · unrated · display positions only')
            doc['META'].update(name=title, composer=artist, song='MISSING_AUDIO')
            path = folder/f'{stem}.rpe.json'
            path.write_text(json.dumps(doc, ensure_ascii=False))
            receipt['chart'] = dict(path=str(path), notes=len(events), package=False)
    except (ExportError, ValueError, FileExistsError) as exc:
        receipt['failed'] = f'{type(exc).__name__}: {exc}'
        receipt.pop('chart', None)
    return receipt
def primary_reference(song):
    return min(song['references'], key=lambda r: stable_rank(PANEL_SEED, 'primary', r['chart_id']))
