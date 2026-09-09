from __future__ import annotations
import hashlib
import numpy as np

from .identity import CanonicalSeed, SeedDeriver, SeedInput, canonicalize_master_seed


class RngPool:
    """Independent deterministic PRNG streams keyed by stage name.

    Existing integer-seed ``pool(name)`` calls intentionally preserve the historical
    PCG64DXSM stream exactly.  New code should prefer ``semantic(...)`` or the
    counter-based helpers, which use the canonical 256-bit master seed and semantic
    identities without coupling results to task creation order.
    """

    def __init__(self, root_seed: SeedInput):
        self._original_seed = root_seed
        self.master_seed: CanonicalSeed = canonicalize_master_seed(root_seed)
        self.deriver = SeedDeriver(self.master_seed)
        self.root_seed = (
            int(root_seed)
            if isinstance(root_seed, int) and not isinstance(root_seed, bool)
            else int.from_bytes(self.master_seed.bytes[:16], "big")
        )
        self._legacy_integer_seed = isinstance(root_seed, int) and not isinstance(root_seed, bool)

    def __call__(self, name: str) -> np.random.Generator:
        if self._legacy_integer_seed:
            digest = hashlib.blake2b(
                f"{self.root_seed}:{name}".encode("utf-8"), digest_size=16
            ).digest()
            words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64)
            ss = np.random.SeedSequence([self.root_seed, *map(int, words)])
            return np.random.Generator(np.random.PCG64DXSM(ss))
        return self.deriver.generator("legacy-stage-name", str(name))

    def semantic(self, *semantic_ids) -> np.random.Generator:
        """Return a Philox stream derived from stable semantic identities."""
        return self.deriver.generator(*semantic_ids)

    def counter_uint64(self, counter: int, *semantic_ids) -> int:
        return self.deriver.counter_uint64(counter, *semantic_ids)

    def counter_uniform01(self, counter: int, *semantic_ids) -> float:
        return self.deriver.counter_uniform01(counter, *semantic_ids)
