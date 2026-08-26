from pathlib import Path

from worldgen.fingerprints import fingerprint_source_files, stage_source_files


def test_astronomy_checkpoint_does_not_depend_on_renderer():
    files = set(stage_source_files("astronomy"))
    assert "astronomy.py" in files
    assert "render.py" not in files
    assert "climate.py" not in files


def test_surface_and_climate_dependencies_are_distinct():
    climate = set(stage_source_files("climate_pass_2"))
    surface = set(stage_source_files("surface_pass_2"))
    assert "climate.py" in climate
    assert "hydrology_base.py" not in climate
    assert "hydrology_base.py" in surface
    assert "flow_refresh.py" in surface


def test_only_selected_files_affect_a_fingerprint(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    first = fingerprint_source_files(["a.py"], package_dir=tmp_path)
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    assert fingerprint_source_files(["a.py"], package_dir=tmp_path) == first
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert fingerprint_source_files(["a.py"], package_dir=tmp_path) != first
