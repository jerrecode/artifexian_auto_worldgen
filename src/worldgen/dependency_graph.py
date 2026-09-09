from __future__ import annotations

"""Dependency-aware invalidation for editable procedural worlds.

This graph models *persisted product invalidation*, not every iteration inside a
coupled numerical solver.  Strong feedback loops (hydrology/erosion/soil/glacier)
are intentionally collapsed into one ``surface_coupling`` product group so the
invalidation graph remains acyclic and easy to reason about.
"""

from dataclasses import dataclass, field
from typing import Iterable, Mapping


DEFAULT_PRODUCT_EDGES: dict[str, frozenset[str]] = {
    "astronomy": frozenset({"climate", "tides", "viewer_products"}),
    "planet": frozenset({"tectonics", "atmosphere_ocean", "climate", "tides", "viewer_products"}),
    "tectonics": frozenset({"geology", "macro_terrain", "resources", "viewer_products"}),
    "geology": frozenset({"macro_terrain", "surface_coupling", "resources", "terrain_refinement", "viewer_products"}),
    "macro_terrain": frozenset({"atmosphere_ocean", "climate", "surface_coupling", "terrain_refinement", "viewer_products"}),
    "atmosphere_ocean": frozenset({"climate", "surface_coupling", "ecology", "viewer_products"}),
    "climate": frozenset({"surface_coupling", "ecology", "habitability", "terrain_refinement", "viewer_products"}),
    "tides": frozenset({"surface_coupling", "habitability", "terrain_refinement", "viewer_products"}),
    "surface_coupling": frozenset({"ecology", "resources", "habitability", "terrain_refinement", "viewer_products"}),
    "ecology": frozenset({"habitability", "terrain_refinement", "viewer_products"}),
    "resources": frozenset({"habitability", "society", "viewer_products"}),
    "habitability": frozenset({"society", "viewer_products"}),
    "society": frozenset({"viewer_products"}),
    "terrain_refinement": frozenset({"viewer_products"}),
    "viewer_products": frozenset(),
}

DEFAULT_PARAMETER_ROOTS: dict[str, frozenset[str]] = {
    "seed": frozenset({"astronomy", "planet", "tectonics", "geology", "macro_terrain", "atmosphere_ocean", "climate", "tides", "surface_coupling", "ecology", "resources", "habitability", "society", "terrain_refinement", "viewer_products"}),
    "astronomy.": frozenset({"astronomy"}),
    "star.": frozenset({"astronomy"}),
    "orbit.": frozenset({"astronomy"}),
    "moon.": frozenset({"astronomy", "tides"}),
    "planet.": frozenset({"planet"}),
    "tectonics.": frozenset({"tectonics"}),
    "geology.": frozenset({"geology"}),
    "terrain.": frozenset({"macro_terrain"}),
    "ocean.": frozenset({"atmosphere_ocean"}),
    "atmosphere.": frozenset({"atmosphere_ocean"}),
    "atmogen.": frozenset({"atmosphere_ocean"}),
    "climate.": frozenset({"climate"}),
    "tides.": frozenset({"tides"}),
    "hydrology.": frozenset({"surface_coupling"}),
    "erosion.": frozenset({"surface_coupling"}),
    "glacier.": frozenset({"surface_coupling"}),
    "glaciers.": frozenset({"surface_coupling"}),
    "soil.": frozenset({"surface_coupling"}),
    "coast.": frozenset({"surface_coupling"}),
    "coasts.": frozenset({"surface_coupling"}),
    "ecology.": frozenset({"ecology"}),
    "vegetation.": frozenset({"ecology"}),
    "resources.": frozenset({"resources"}),
    "habitability.": frozenset({"habitability"}),
    "society.": frozenset({"society"}),
    "lod.": frozenset({"terrain_refinement", "viewer_products"}),
    "renderer.": frozenset({"viewer_products"}),
    "viewer.": frozenset({"viewer_products"}),
    "runtime.": frozenset(),
    "storage.": frozenset(),
    "logging.": frozenset(),
    "progress.": frozenset(),
}


@dataclass(frozen=True, slots=True)
class InvalidationPlan:
    changed_parameters: tuple[str, ...]
    root_products: frozenset[str]
    stale_products: frozenset[str]
    unaffected_products: frozenset[str]


@dataclass(slots=True)
class ProductDependencyGraph:
    edges: Mapping[str, frozenset[str]] = field(default_factory=lambda: DEFAULT_PRODUCT_EDGES)
    parameter_roots: Mapping[str, frozenset[str]] = field(default_factory=lambda: DEFAULT_PARAMETER_ROOTS)

    def __post_init__(self) -> None:
        nodes = set(self.edges)
        missing = {child for children in self.edges.values() for child in children if child not in nodes}
        if missing:
            raise ValueError(f"dependency graph references unknown products: {sorted(missing)}")
        self._assert_acyclic()

    @property
    def products(self) -> frozenset[str]:
        return frozenset(self.edges)

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError(f"dependency cycle detected at {node!r}")
            visiting.add(node)
            for child in self.edges[node]:
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in self.edges:
            visit(node)

    def descendants(self, roots: Iterable[str], *, include_roots: bool = True) -> frozenset[str]:
        roots_set = {str(root) for root in roots}
        unknown = roots_set - set(self.edges)
        if unknown:
            raise KeyError(f"unknown products: {sorted(unknown)}")
        seen = set(roots_set if include_roots else ())
        pending = list(roots_set)
        while pending:
            node = pending.pop()
            for child in self.edges[node]:
                if child not in seen:
                    seen.add(child)
                    pending.append(child)
        if not include_roots:
            seen.difference_update(roots_set)
        return frozenset(seen)

    def roots_for_parameter(self, path: str) -> frozenset[str]:
        path = str(path)
        matches = [prefix for prefix in self.parameter_roots if path == prefix or path.startswith(prefix)]
        if not matches:
            return frozenset()
        longest = max(len(prefix) for prefix in matches)
        roots: set[str] = set()
        for prefix in matches:
            if len(prefix) == longest:
                roots.update(self.parameter_roots[prefix])
        return frozenset(roots)

    def plan(self, changed_parameters: Iterable[str]) -> InvalidationPlan:
        changed = tuple(dict.fromkeys(str(path) for path in changed_parameters))
        roots: set[str] = set()
        for path in changed:
            roots.update(self.roots_for_parameter(path))
        stale = self.descendants(roots) if roots else frozenset()
        return InvalidationPlan(
            changed_parameters=changed,
            root_products=frozenset(roots),
            stale_products=stale,
            unaffected_products=self.products - stale,
        )


__all__ = [
    "DEFAULT_PARAMETER_ROOTS",
    "DEFAULT_PRODUCT_EDGES",
    "InvalidationPlan",
    "ProductDependencyGraph",
]
