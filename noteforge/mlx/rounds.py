"""MLX rollout in rounds: cores of a round decode together, so steps are about the longest core per round, not all tokens.

Round 0 has no history. A core in a later round sees the core before it when that one is already written."""
import time

import torch

from noteforge.model.event_canvas import RhythmEvent
from noteforge.model.full_event_generate import generate_batch
from noteforge.model import local_song_rollout as R


def clamp_crossing_holds(state):
    """A cold core does not know an earlier hold is still down: end that hold before the next press of its key."""
    events = sorted(state['events'], key=lambda e: e['onset_us'])
    if state['mode'] != 'mania_4k':
        state['events'] = events
        return 0
    fixed, last = 0, {}
    for e in events:
        held = last.get(e['lane'])
        if held is not None and held['type'] == 'hold' and held['end_us'] > e['onset_us']-40_000:
            end = e['onset_us']-40_000
            if end-held['onset_us'] < 60_000:
                held.update(type='tap', end_us=held['onset_us'])
            else:
                held['end_us'] = end
            fixed += 1
        last[e['lane']] = e
    state['events'] = events
    return fixed


def generate_rounds(model, features, songs, seeds, *, device, batch_size=16, token_cap=4096, max_wall_s=1200,
                    history_reference=True, on_progress=None, generator=generate_batch, input_shapes=None,
                    intensity_provider=None, on_core_inputs=None, condition_provider=None, timing_provider=None,
                    alternative_provider=None, scheduler=None, completed=None, stride=2, keep_before_us=None, initial_events=None, refine=False):
    seeds = [int(s) for s in seeds]
    per_seed = [R._initial_states(songs, history_reference) for _ in seeds]
    row_seeds = {id(s): seed for seed, group in zip(seeds, per_seed) for s in group}
    states = [s for group in per_seed for s in group]
    started = time.monotonic()
    deadline = started+max_wall_s
    for state in states:
        if initial_events:
            state['events'].extend(dict(e) for e in initial_events if keep_before_us is None or e['onset_us'] < keep_before_us)
    # refine: one more round rewrites the cores that were written without history, now with it.
    passes = [(offset, False) for offset in range(stride)]+([(0, True)] if refine else [])
    for offset, rewriting in passes:
        pending = []
        for state in states:
            cores = range(0, (state['duration_us']+7_999_999)//8_000_000)
            for i in cores:
                start = i*8_000_000
                # Continuing: only the core holding the cut point and the ones after it.
                if rewriting and i == 0:
                    continue
                if i % stride == offset and (keep_before_us is None or start+8_000_000 > keep_before_us):
                    pending.append((state, start))
        pending.sort(key=lambda x: (x[1], row_seeds[id(x[0])]))
        # Rows of one round must not see each other, whichever batch they land in.
        written_before = {id(state): list(state['events']) for state in states}
        for at in range(0, len(pending), batch_size):
            chunk = pending[at:at+batch_size]
            requests, built = [], []
            for state, start in chunk:
                end = min(start+8_000_000, state['duration_us'])
                events = [RhythmEvent(**e) for e in written_before[id(state)]]
                crossing = {}  # cold cores do not know each other: keep the latest hold on each key
                for e in sorted((e for e in events if e.type == 'hold' and e.onset_us < start < e.end_us), key=lambda e: e.onset_us):
                    crossing[e.lane if state['mode'] == 'mania_4k' else id(e)] = e
                carry = tuple(crossing.values())
                request = R.EventRequest(state['mode'], start, end, state['duration_us'], known_carry=carry)
                written = any(start-8_000_000 <= e.onset_us < start for e in events)
                state['history_override'] = written_before[id(state)]
                row = R._build_core_row(state, request, features, history_reference=history_reference and written,
                                        intensity_provider=intensity_provider, condition_provider=condition_provider,
                                        timing_provider=timing_provider, alternative_provider=alternative_provider)
                requests.append(request)
                built.append(dict(row, state=state, request=request))
            inputs = R._device_rows([b['row'] for b in built], input_shapes, device)
            extra = {}
            if any(b['condition'] is not None for b in built):
                extra['unconditional_inputs'] = R._device_rows([b['bare'] for b in built], input_shapes, device)
            print(f'rounds offset {offset} rows {at+len(chunk)}/{len(pending)} {time.monotonic()-started:.0f}s', flush=True)
            tick = time.monotonic()
            predictions = generator(model, inputs, requests, token_cap, deadline=deadline,
                                    row_seeds=[row_seeds[id(b['state'])] for b in built], **extra)
            for b, request, prediction in zip(built, requests, predictions):
                state = b['state']
                if rewriting:
                    state['events'] = [e for e in state['events'] if not request.start_us <= e['onset_us'] < request.end_us]
                    state['cores'] = [c for c in state['cores'] if c['start_us'] != request.start_us]
                R._record_core(state, request, prediction, b, None, time.monotonic()-tick, None)
                state['complete'] = False
                if prediction['complete']:
                    state['failure'] = None if state.get('failure') == 'independent_legality' else state['failure']
    for state in states:
        state.pop('history_override', None)
        state['cores'].sort(key=lambda c: c['start_us'])
        if keep_before_us is not None:  # the core holding the cut point was regenerated whole: its part before the cut is not used
            kept = {(e['onset_us'], e['end_us'], e['type'], e['lane']) for e in (initial_events or ()) if e['onset_us'] < keep_before_us}
            state['events'] = [e for e in state['events'] if e['onset_us'] >= keep_before_us or (e['onset_us'], e['end_us'], e['type'], e['lane']) in kept]
        state['clamped_holds'] = clamp_crossing_holds(state)
        state['legality'] = R.independent_legality([RhythmEvent(**e) for e in state['events']], state['mode'], state['duration_us'])
        done = all(c['complete'] for c in state['cores'])
        state['complete'] = done and not state['legality']
        state['failure'] = None if state['complete'] else (state.get('failure') or ('independent_legality' if state['legality'] else 'core_incomplete'))
    wall = time.monotonic()-started
    return [dict(complete=all(s['complete'] for s in group), songs=group, wall_s=wall, history_reference=history_reference,
                 parallel_samples=len(seeds), conditioning='rounds') for group in per_seed]
