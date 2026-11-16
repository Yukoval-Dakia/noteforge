"""Keep current licensing and the upstream MERT exception from drifting."""
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_current_release_uses_apache_2():
    project = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    manifest = json.loads((ROOT/'models/MANIFEST.json').read_text())
    license_text = (ROOT/'LICENSE').read_text()
    notice = (ROOT/'NOTICE').read_text()
    assert project['license'] == 'Apache-2.0'
    assert manifest['license'] == 'Apache-2.0'
    assert 'Apache License' in license_text
    assert 'NoteForge is licensed under the Apache License, Version 2.0' in notice


def test_historical_mit_and_upstream_mert_limits_are_explicit():
    licensing = (ROOT/'LICENSING.md').read_text()
    notice = (ROOT/'NOTICE').read_text()
    assert 'through v1.1.1' in licensing
    assert 'CC BY-NC 4.0' in licensing
    assert 'not distributed' in licensing
    assert 'CC BY-NC 4.0' in notice
