from __future__ import annotations

import pytest

from worldgen import __version__
from worldgen.cli import _parser


def test_cli_version_matches_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"worldgen {__version__}"
