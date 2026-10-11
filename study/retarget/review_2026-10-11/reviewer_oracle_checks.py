"""Independent synthetic reviewer checks; no manuscript or experiment edits."""
from pathlib import Path
import itertools
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'analysis_r1' / 'scripts'))
import r1_budget_oracle as O


def check(q, segments, cap, mode):
    windows = O.C.segment_windows(segments, 7)
    candidates = []
    for k in ([cap] if mode == 'count' else range(cap + 1)):
        for selected in itertools.combinations(range(len(q)), k):
            a = np.zeros(len(q), dtype=bool)
            a[list(selected)] = True
            s = np.where(a, 1.0, q)
            candidates.append((min(float(s[w].mean()) for w in windows),
                               float((1 - q)[a].sum())))
    z = max(x[0] for x in candidates)
    rescue = max(x[1] for x in candidates if abs(x[0] - z) < 1e-10)
    r = O.solve_oracle(q, segments, mode, cap, time_limit=15)
    return {'mode': mode, 'cap': cap, 'n': len(q),
            'enumerated_z': z, 'returned_z': r['worst7'],
            'enumerated_rescue': rescue, 'returned_rescue': r['rescued_shortfall_sum'],
            'pass': bool(abs(r['worst7'] - z) < 1e-7 and
                         abs(r['rescued_shortfall_sum'] - rescue) < 1e-7 and
                         r['validation']['valid'])}


def main():
    rows = []
    # A real conflict: the best single mean-rescue action is day0 (rescue1),
    # whereas max-min W7 requires day6 (rescue0.6). Wrong Stage2 sign fails.
    q = np.ones(10)
    q[[0, 6, 9]] = [0.0, 0.4, 0.2]
    for mode in ('count', 'cap'):
        rows.append(check(q, np.zeros(10, dtype=int), 1, mode))
    rng = np.random.default_rng(10112026)
    for n, cap, segments in ((10, 2, np.zeros(10, dtype=int)),
                             (14, 2, np.repeat([0, 1], 7))):
        for _ in range(3):
            q = rng.integers(0, 11, n) / 10
            for mode in ('count', 'cap'):
                rows.append(check(q, segments, cap, mode))
    out = {'synthetic_only': True, 'all_pass': all(r['pass'] for r in rows), 'cases': rows}
    dest = Path(__file__).with_name('REVIEWER_ORACLE_SYNTHETIC_RESULTS.json')
    dest.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
