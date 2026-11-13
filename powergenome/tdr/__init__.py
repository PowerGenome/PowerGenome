"""Time domain reduction backend.

The public entry point is :func:`reduce_time_domain`, which returns the same four
GenX artifacts as the legacy ``powergenome.time_reduction`` implementation while
using per-feature scaling, ``sqrt`` feature weights, forced extreme periods, and
distinct representative periods.

The legacy implementation remains importable for one release and is reachable
with ``time_domain_reduction_method: legacy_kmeans``.
"""

from powergenome.tdr.genx import reduce_time_domain

__all__ = ["reduce_time_domain"]
