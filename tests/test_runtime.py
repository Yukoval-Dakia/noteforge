import json

import numpy as np
import pytest

from noteforge.backends import resolve_backend, resolve_torch_device
from noteforge.model.local_song_rollout import _device_rows


def test_backend_auto_never_names_a_hardware_vendor():
    target = resolve_backend('auto', 'cpu')
    assert target.backend == 'torch'
    assert target.device == 'cpu'
    assert target.error is None


def test_torch_auto_resolves_to_a_real_device():
    assert resolve_torch_device('auto') in ('cuda', 'mps', 'cpu')


def test_rollout_device_rows_unwrap_the_collator_adapter():
    row = dict(inputs=dict(mode=np.asarray(1, np.int64), controls=np.zeros((2, 3), np.float32)),
               sequence={"adapter": True})
    inputs = _device_rows([row], None, 'cpu')
    assert tuple(inputs['mode'].shape) == (1,)
    assert tuple(inputs['controls'].shape) == (1, 2, 3)


def test_explicit_unavailable_cuda_fails_closed():
    try:
        device = resolve_torch_device('cuda')
    except RuntimeError as exc:
        assert 'CUDA was requested' in str(exc)
    else:
        assert device == 'cuda'


def test_cli_reports_an_unavailable_device_as_an_error_not_a_traceback():
    """`--device cuda` on a host without CUDA exits with a message, not a stack trace."""
    import importlib.util
    import torch
    if torch.cuda.is_available():
        return
    import noteforge.scripts.draft_song as draft_song
    args = draft_song.parse_args(['--backend', 'torch', '--device', 'cuda', '--mode', 'phira', '--model', 'm.pt',
                                  '--phira-level', '10', '--audio', 'a.wav', '--out-dir', '/tmp/x'])
    try:
        draft_song.run(args)
    except SystemExit as exc:
        assert 'CUDA was requested' in str(exc)
    else:
        raise AssertionError('run() should refuse an unavailable device')


def test_mlx_backend_use_is_reported_without_claiming_metal_as_a_torch_device():
    try:
        from noteforge.mlx import draft_mlx
    except ImportError:
        return
    args = draft_mlx.parse_mlx_args(['--model', 'm', '--audio', 'a', '--out-dir', 'o', '--mode', 'phira',
                                     '--device', 'auto'])
    assert args.device == 'cpu'
    assert args.mert_device == 'mps'


def test_phira_layout_handles_an_empty_draft():
    import torch
    from noteforge.placement.apply import layout
    from noteforge.placement.model import M
    net = M()
    assert layout([], net, seed=0) == []


def test_placement_import_does_not_override_cpu_threads():
    import importlib
    import torch
    before = torch.get_num_threads()
    import noteforge.placement.model as placement_model
    importlib.reload(placement_model)
    assert torch.get_num_threads() == before


def test_meta_model_construction_matches_the_plain_path():
    """load_model builds on meta and assigns weights; the result must equal plain construct + load."""
    import torch
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_model import FullEventConfig, FullEventModel
    from noteforge.scripts.draft_song import build_model_on_meta
    torch.manual_seed(0)
    config = FullEventConfig(
        d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2,
        tap_dim=32, taps=1, bands=2, band_slots=1, reference_dim=4, controls=2, dropout=0.)
    vocabulary = EventVocabulary()
    state = FullEventModel(config, vocabulary).state_dict()
    plain = FullEventModel(config, vocabulary)
    plain.load_state_dict(state, strict=True)
    meta = build_model_on_meta(config, vocabulary, state)
    assert not any(p.is_meta for p in meta.parameters())
    left, right = dict(plain.named_parameters()), dict(meta.named_parameters())
    assert set(left) == set(right)
    assert all(torch.equal(left[k], right[k]) for k in left)


def test_doctor_reports_a_missing_ffmpeg_without_crashing(monkeypatch):
    from noteforge.scripts import doctor
    monkeypatch.setattr(doctor.shutil, 'which', lambda name: None)
    report = doctor.Report()
    doctor.check_tools(report)
    assert [c['check'] for c in report.checks] == ['tool.ffmpeg', 'tool.ffprobe']
    assert all(c['status'] == 'fail' for c in report.checks)
    assert report.ready is False


def test_doctor_fails_on_a_manifest_hash_mismatch(tmp_path, monkeypatch):
    from noteforge.scripts import doctor
    (tmp_path/'weights.pt').write_bytes(b'not the published file')
    manifest = tmp_path/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(main_weights='weights.pt', files=[
        dict(path='weights.pt', sha256='0'*64, bytes=22)])))
    monkeypatch.setattr(doctor, 'MANIFEST', manifest)
    monkeypatch.setattr(doctor, 'ROOT', tmp_path)
    report = doctor.Report()
    doctor.check_manifest(report, verify_hashes=True)
    assert report.checks[0]['status'] == 'fail'
    assert 'do not match' in report.checks[0]['message']


def test_doctor_fails_when_the_main_weights_are_absent(tmp_path, monkeypatch):
    from noteforge.scripts import doctor
    manifest = tmp_path/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(main_weights='models/big.pt', files=[
        dict(path='models/big.pt', sha256='0'*64, bytes=10)])))
    monkeypatch.setattr(doctor, 'MANIFEST', manifest)
    monkeypatch.setattr(doctor, 'ROOT', tmp_path)
    report = doctor.Report()
    doctor.check_manifest(report, verify_hashes=True)
    assert report.checks[0]['status'] == 'fail'
    assert 'missing' in report.checks[0]['message']


def test_doctor_ready_is_ok_when_every_check_passes():
    from noteforge.scripts import doctor
    report = doctor.Report()
    report.add('a', 'info', 'text')
    report.add('b', 'warn', 'text')
    assert report.ready is True
    assert report.status == 'warn'
    report.add('c', 'fail', 'text')
    assert report.ready is False


def test_mlx_assets_are_discovered_from_the_conventional_folders(tmp_path):
    import noteforge.backends as backends
    for folder, names in ((backends.MLX_FAST_DIR, backends.MLX_WEIGHT_FILES),
                          (backends.MLX_MERT_DIR, backends.MLX_MERT_FILES)):
        target = tmp_path/folder
        target.mkdir(parents=True)
        for name in names:
            (target/name).write_text('x')
    weights, mert = backends.discover_mlx_assets(tmp_path)
    assert weights == tmp_path/backends.MLX_FAST_DIR
    assert mert == tmp_path/backends.MLX_MERT_DIR


def test_auto_uses_discovered_mlx_assets_but_an_explicit_path_still_wins(tmp_path, monkeypatch):
    import noteforge.backends as backends
    for folder, names in ((backends.MLX_FAST_DIR, backends.MLX_WEIGHT_FILES),
                          (backends.MLX_MERT_DIR, backends.MLX_MERT_FILES)):
        target = tmp_path/folder
        target.mkdir(parents=True)
        for name in names:
            (target/name).write_text('x')
    monkeypatch.setattr(backends, 'apple_silicon', lambda: True)
    monkeypatch.setattr(backends, 'mlx_available', lambda: True)
    found = backends.resolve_backend('auto', 'auto', None, None, root=tmp_path)
    assert (found.backend, found.device) == ('mlx', 'mlx')
    explicit = backends.resolve_backend('auto', 'cpu', None, None, root=tmp_path)
    assert explicit.backend == 'torch'


def test_kind_bias_rule_maps_phira_types_to_their_kind_tokens():
    """``--flick-ratio/--drag-ratio`` steer decode through ``kind_bias``; the batch loops take it too."""
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_generate import _kind_bias_rule
    vocab = EventVocabulary()
    rule = _kind_bias_rule(type('M', (), {'vocab': vocab})(), dict(flick=2.5, drag=-1.0))
    assert rule == {vocab.token('kind', 2): 2.5, vocab.token('kind', 3): -1.0}
    assert _kind_bias_rule(type('M', (), {'vocab': vocab})(), None) is None


def test_kind_bias_rule_rejects_unknown_types_and_non_finite_values():
    import pytest
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_generate import _kind_bias_rule
    model = type('M', (), {'vocab': EventVocabulary()})()
    with pytest.raises(ValueError, match='unknown Phira type'):
        _kind_bias_rule(model, dict(slide=1.))
    with pytest.raises(ValueError, match='finite'):
        _kind_bias_rule(model, dict(flick=float('nan')))


def _tiny_inputs(rows=1, length=4, taps=2, bands=2, dim=16):
    import torch
    local = torch.randn(rows, taps, length, bands, dim)
    return dict(local=local, local_times=torch.zeros(rows, length), local_valid=torch.ones(rows, length, dtype=torch.bool),
                song=torch.zeros(rows, 1, dim), song_times=torch.zeros(rows, 1),
                song_valid=torch.ones(rows, 1, dtype=torch.bool),
                reference=torch.zeros(rows, 2, 16), reference_times=torch.zeros(rows, 2),
                reference_valid=torch.zeros(rows, 2, dtype=torch.bool),
                controls=torch.zeros(rows, 8, 6), mode=torch.ones(rows, dtype=torch.long))


def _tiny_model():
    import torch
    from noteforge.model.full_event_codec import EventVocabulary
    from noteforge.model.full_event_model import FullEventConfig, FullEventModel
    torch.manual_seed(0)
    config = FullEventConfig(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=1,
                             tap_dim=16, taps=2, bands=2, band_slots=1, reference_dim=16, controls=8, dropout=0.)
    return FullEventModel(config, EventVocabulary()).eval()


@pytest.mark.parametrize('fast', [True, False])
def test_kind_bias_changes_the_drawn_type_in_both_decode_loops(fast):
    """A large bias on one Phira type must change the decoded events, not just be accepted."""
    import torch
    from noteforge.model.full_event_codec import EventRequest
    from noteforge.model.full_event_generate import generate_batch
    model = _tiny_model()
    vocab = model.vocab
    # Drive the grammar by hand so the run reaches the ``kind`` phase and writes
    # notes, then rank tap just above flick on every step. The only thing that
    # can flip the drawn type is the decode-time bias under test.
    calls = {'n': 0}
    def head(_module, _inputs, output):
        output.zero_()
        step = calls['n']
        calls['n'] += 1
        output[..., vocab.token('group', 1 if step < 30 else 0)] = 10.0
        output[..., vocab.token('kind', 0)] = 1.0   # tap wins without a bias
        output[..., vocab.token('kind', 2)] = 0.5   # flick overtakes tap with bias >= 1
        return output
    model.output.register_forward_hook(head)
    request = EventRequest('phira', 0, 8_000_000, 8_000_000)
    inputs = _tiny_inputs()
    common = dict(max_tokens=60, temperature=None, fast=fast, cuda_graph=False)
    plain = generate_batch(model, inputs, [request], **common)[0]
    calls['n'] = 0
    biased = generate_batch(model, inputs, [request], kind_bias=dict(flick=2.0), **common)[0]
    plain_types = [e['type'] for e in plain['events']]
    biased_types = [e['type'] for e in biased['events']]
    assert plain_types, 'the controlled run must write notes to be a meaningful check'
    assert plain_types == ['tap'] * len(plain_types)
    assert biased_types == ['flick'] * len(biased_types)


def test_mlx_mert_export_accepts_hf_and_validates_what_it_downloads(tmp_path, monkeypatch):
    """`export mlx-mert --mert-source hf` must resolve the pinned revision, not crash.

    Regression: `resolve_mert_source('hf')` returns a None directory (it means
    "use the pinned Hugging Face revision"), which was passed straight into
    Path() and raised TypeError.
    """
    import hashlib
    import sys
    import types
    from noteforge.mlx import mert_fast
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    for name in ('config.json', 'configuration_mert2.py', 'modeling_mert2.py'):
        (snapshot/name).write_text(name)
    (snapshot/'model.safetensors').write_bytes(b'weights')
    expected = {name: hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
                for name in ('config.json', 'configuration_mert2.py', 'modeling_mert2.py', 'model.safetensors')}
    monkeypatch.setattr('noteforge.audio.mert2.LOCAL_SOURCE_SHA256', expected)
    calls = {}
    fake_hub = types.ModuleType('huggingface_hub')
    def snapshot_download(repo, revision=None, allow_patterns=None):
        calls.update(repo=repo, revision=revision, allow=sorted(allow_patterns))
        return str(snapshot)
    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)
    found = mert_fast.source_dir('hf')
    assert found == snapshot
    assert calls['repo'] == 'm-a-p/MERT-v2-FullSong'
    assert calls['revision'] == '7001f7657276238629e61c7746ce70735e0604c0'
    # A local directory still wins and is never downloaded.
    calls.clear()
    assert mert_fast.source_dir(str(snapshot)) == snapshot
    assert calls == {}


def test_mlx_mert_export_rejects_a_source_whose_pinned_files_changed(tmp_path, monkeypatch):
    import sys
    import types
    from noteforge.mlx import mert_fast
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    for name in ('config.json', 'configuration_mert2.py', 'modeling_mert2.py', 'model.safetensors'):
        (snapshot/name).write_text('not the pinned file')
    monkeypatch.setattr('noteforge.audio.mert2.LOCAL_SOURCE_SHA256',
                        {name: '0'*64 for name in ('config.json', 'configuration_mert2.py',
                                                   'modeling_mert2.py', 'model.safetensors')})
    fake_hub = types.ModuleType('huggingface_hub')
    fake_hub.snapshot_download = lambda *a, **k: str(snapshot)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hub)
    with pytest.raises(ValueError, match='identity changed'):
        mert_fast.source_dir('hf')
