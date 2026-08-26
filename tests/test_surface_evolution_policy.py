from types import SimpleNamespace

import worldgen.surface_evolution as policy


def _args(cfg):
    return (None, None, None, None, None, cfg)


def test_interval_mode_delegates_to_legacy_surface_evolution(monkeypatch):
    sentinel = object()
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(policy._base, "evolve_surface", fake)
    cfg = SimpleNamespace(flow_refresh_mode="interval")
    assert policy.evolve_surface(*_args(cfg)) is sentinel
    assert len(calls) == 1


def test_adaptive_mode_selects_adaptive_implementation(monkeypatch):
    sentinel = object()
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(policy, "_evolve_surface_adaptive", fake)
    cfg = SimpleNamespace(flow_refresh_mode="adaptive")
    assert policy.evolve_surface(*_args(cfg)) is sentinel
    assert len(calls) == 1
