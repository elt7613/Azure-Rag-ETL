"""`.env`, `.env.example` and `Settings` must agree.

Hand-maintained config pairs drift. A setting gets added to the class and to a
local `.env`, the template is forgotten, and the next person to set the project
up meets it by hitting a default they did not know existed — or worse, by an
`api_default_departments` that silently grants everything because their file
predates the field. These tests make that a red build instead of a surprise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rag.config import Settings

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"


def parse(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key.strip()] = value
    return values


def order(path: Path) -> list[str]:
    return list(parse(path))


def test_the_example_covers_every_setting():
    """A setting absent from the template is one nobody knows they can set."""
    expected = {name.upper() for name in Settings.model_fields}
    assert parse(EXAMPLE).keys() == expected


def test_no_setting_in_the_example_is_unknown_to_the_class():
    """A leftover key is worse than a missing one: it looks configurable."""
    known = {name.upper() for name in Settings.model_fields}
    assert set(parse(EXAMPLE)) <= known


@pytest.mark.skipif(not ENV.exists(), reason="no local .env")
def test_env_and_example_hold_the_same_keys_in_the_same_order():
    """Same order so the two files can be read side by side.

    Matching key *sets* is not enough — they matched while the order differed,
    and comparing them by eye was useless.
    """
    assert parse(ENV).keys() == parse(EXAMPLE).keys()
    assert order(ENV) == order(EXAMPLE)


def test_every_setting_is_documented():
    """The template explains what each field is for, not just that it exists."""
    lines = EXAMPLE.read_text().splitlines()
    undocumented = []
    for index, line in enumerate(lines):
        if not line.strip() or line.startswith("#") or "=" not in line:
            continue
        previous = lines[index - 1].strip() if index else ""
        if not previous.startswith("#") or previous.startswith("# ---"):
            undocumented.append(line.split("=", 1)[0])
    assert undocumented == [], f"settings with no comment above them: {undocumented}"


def test_required_settings_are_marked_and_left_blank():
    """Someone copying the template must be told what they have to fill in."""
    text = EXAMPLE.read_text()
    values = parse(EXAMPLE)
    for name, field in Settings.model_fields.items():
        if not field.is_required():
            continue
        key = name.upper()
        assert values[key] == "", f"{key} is required and must ship blank"
        comment = text.split(f"\n{key}=")[0].rsplit("\n", 1)[-1]
        assert "REQUIRED" in comment, f"{key} is required but its comment does not say so"


def test_the_example_ships_no_credentials():
    """The template is committed; the file with real keys is not."""
    for key, value in parse(EXAMPLE).items():
        if any(token in key for token in ("KEY", "PASSWORD", "CONNECTION_STRING")):
            assert value == "", f"{key} must be blank in the committed template"


def test_access_control_defaults_to_denying_everything():
    """The one default where being wrong is a security bug, not an annoyance."""
    assert parse(EXAMPLE)["API_DEFAULT_DEPARTMENTS"] == "[]"


def test_the_generator_reports_the_files_as_current():
    """Guards the generator itself: if it would rewrite them, they are stale."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_env.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
