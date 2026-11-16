"""`noteforge <command>` is a forwarder: every command keeps its module's own parser."""
import json
import sys
from pathlib import Path

import pytest

from noteforge import cli


def test_top_level_help_lists_every_command_and_exits_zero(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for command in cli.COMMANDS:
        assert command in out
    assert 'noteforge doctor' in out


def test_version_reports_the_package_version_not_installed_metadata(capsys):
    import noteforge
    assert cli.main(['--version']) == 0
    assert capsys.readouterr().out.strip() == noteforge.__version__
    assert cli.main(['version']) == 0
    assert capsys.readouterr().out.strip() == noteforge.__version__


def test_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(['draftt'])
    assert exc.value.code == 2


def test_dispatch_forwards_argv_unchanged(monkeypatch):
    seen = []
    from noteforge.scripts import draft_song
    monkeypatch.setattr(draft_song, 'main', lambda argv: seen.append(argv) or {'ok': True})
    assert cli.main(['draft', '--mode', 'phira', '--audio', 'a.wav']) == 0
    assert seen == [['--mode', 'phira', '--audio', 'a.wav']]


def test_draft_returns_one_when_no_receipt(monkeypatch):
    from noteforge.scripts import draft_song
    monkeypatch.setattr(draft_song, 'main', lambda argv: None)
    assert cli.main(['draft', '--mode', 'phira']) == 1


def test_extract_injects_extract_only_once(monkeypatch):
    seen = []
    from noteforge.scripts import draft_song
    monkeypatch.setattr(draft_song, 'main', lambda argv: seen.append(list(argv)) or {'ok': True})
    cli.main(['extract', '--audio', 'a.wav', '--features-dir', 'f'])
    cli.main(['extract', '--extract-only', '--audio', 'a.wav', '--features-dir', 'f'])
    assert seen[0][0] == '--extract-only'
    assert seen[1].count('--extract-only') == 1


def test_subparser_usage_names_the_subcommand(monkeypatch, capsys):
    """`noteforge draft --help` must read `noteforge draft`, not `cli.py`."""
    with pytest.raises(SystemExit) as exc:
        cli.main(['draft', '--help'])
    assert exc.value.code == 0
    assert 'noteforge draft' in capsys.readouterr().out


def test_bare_help_command_prints_that_command_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['help', 'bench'])
    assert exc.value.code == 0
    assert '--backend' in capsys.readouterr().out


def test_serve_requires_a_token_before_importing_uvicorn(monkeypatch, capsys):
    monkeypatch.delenv('NOTEFORGE_API_TOKEN', raising=False)
    assert cli.main(['serve']) == 2
    assert 'NOTEFORGE_API_TOKEN' in capsys.readouterr().err


def test_serve_parses_host_and_port_from_the_environment(monkeypatch):
    monkeypatch.setenv('NOTEFORGE_HOST', '0.0.0.0')
    monkeypatch.setenv('NOTEFORGE_PORT', '9000')
    args = cli.serve_parser().parse_args([])
    assert (args.host, args.port) == ('0.0.0.0', 9000)


def test_dispatcher_restores_sys_argv_zero(monkeypatch):
    from noteforge.scripts import doctor
    original = sys.argv[0]
    monkeypatch.setattr(doctor, 'main', lambda argv: 0)
    assert cli.main(['doctor']) == 0
    assert sys.argv[0] == original


def test_every_advertised_command_has_a_runner():
    """The help text and the dispatch table cannot drift apart."""
    assert set(cli.COMMANDS) == set(cli.RUNNERS)


def test_each_command_dispatches_to_its_runner(monkeypatch):
    seen = []
    for command, runner in cli.RUNNERS.items():
        monkeypatch.setitem(cli.RUNNERS, command, lambda argv, _c=command: seen.append(_c) or 0)
    for command in cli.COMMANDS:
        assert cli.main([command, '--any-flag']) == 0
    assert seen == list(cli.COMMANDS)


def test_plan_forwards_to_the_planner(monkeypatch):
    seen = []
    from noteforge.planner import plan_song
    monkeypatch.setattr(plan_song, 'main', lambda argv: seen.append(list(argv)))
    assert cli.main(['plan', '--features-dir', 'f', '--audio-sha', 's', '--mode', 'phira', '--out', 'o']) == 0
    assert seen == [['--features-dir', 'f', '--audio-sha', 's', '--mode', 'phira', '--out', 'o']]


def test_export_needs_a_mert_source_for_the_frontend(monkeypatch, capsys):
    monkeypatch.delenv('NOTEFORGE_MERT_SOURCE', raising=False)
    assert cli.main(['export', 'mlx-mert']) == 2
    assert 'mert-source' in capsys.readouterr().err


def test_export_parser_has_no_hidden_required_flag():
    args = cli.export_parser().parse_args(['mlx'])
    assert args.target == 'mlx'
    assert args.checkpoint.endswith('noteforge-r5.pt')


def test_mert_status_is_backend_neutral_and_does_not_download(capsys, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError('status must not download MERT')

    monkeypatch.delenv('NOTEFORGE_MERT_SOURCE', raising=False)
    monkeypatch.setattr('huggingface_hub.snapshot_download', fail_download, raising=False)
    assert cli.main(['mert', 'status', '--source', 'hf', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['source'] == 'hf'
    assert report['downloaded'] is False
    assert report['identity']['shared_by'] == ['torch-cpu', 'torch-cuda', 'torch-mps', 'mlx']


def test_mert_fetch_requires_an_explicit_hf_output(capsys):
    assert cli.main(['mert', 'fetch', '--source', 'hf']) == 2
    assert '2.4 GB' in capsys.readouterr().err


def test_mert_export_mlx_is_an_explicit_derived_cache(monkeypatch, tmp_path):
    seen = {}
    from noteforge.mlx import mert_fast

    monkeypatch.setattr(mert_fast, 'export', lambda source, out: seen.update(source=source, out=out))
    assert cli.main(['mert', 'export-mlx', '--source', 'hf', '--out', str(tmp_path)]) == 0
    assert seen == dict(source='hf', out=tmp_path)


def test_inspect_reads_a_real_chart_and_requires_a_mode(tmp_path, capsys):
    """`noteforge inspect` exposes the engine's chart reader; mode is mandatory."""
    chart = tmp_path/'fixture.osu'
    chart.write_bytes((Path(__file__).parent/'fixture.osu').read_bytes())
    assert cli.main(['inspect', str(chart), '--mode', 'mania_4k']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['mode'] == 'mania_4k'
    assert report['notes'] > 0
    with pytest.raises(SystemExit) as exc:
        cli.main(['inspect', str(chart)])
    assert exc.value.code == 2
