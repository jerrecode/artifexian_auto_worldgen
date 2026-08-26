from __future__ import annotations
import hashlib
import numpy as np


class RngPool:
    """Independent deterministic PRNG streams keyed by stage name.

    A stage can change internally without perturbing unrelated stages, which makes
    iterative worldbuilding reproducible and tunable.
    """

    def __init__(self, root_seed: int):
        self.root_seed = int(root_seed)

    def __call__(self, name: str) -> np.random.Generator:
        digest = hashlib.blake2b(
            f"{self.root_seed}:{name}".encode("utf-8"), digest_size=16
        ).digest()
        words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64)
        ss = np.random.SeedSequence([self.root_seed, *map(int, words)])
        return np.random.Generator(np.random.PCG64DXSM(ss))
