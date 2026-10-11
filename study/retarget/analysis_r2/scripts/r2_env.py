"""R2 thread-limit environment: imported FIRST, before any numeric import.

Assignment (not setdefault) per R03C item 2: the formal runs require an actual
2-thread numeric environment; the values are recorded in every manifest.
"""
import os

THREAD_LIMITS = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                 "VECLIB_MAXIMUM_THREADS": "2"}
for _k, _v in THREAD_LIMITS.items():
    os.environ[_k] = _v
