import json

import numpy as np

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
