"""Recompute exposed allocation metrics from emitted traces, without author helpers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'analysis_r1' / 'outputs' / 'allocation_value'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows, failures, identities = [], [], []
    for trace in sorted(ROOT.glob('*__daily.csv.gz')):
        stem = trace.name.removesuffix('__daily.csv.gz')
        payload_path = ROOT / (stem + '.json')
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        fp = json.loads((ROOT / (stem + '__fingerprint.json')).read_text(encoding='utf-8'))
        meta = payload['meta']
        ident = (meta['cohort'], meta['origin'])
        identities.append(ident)
        if fp['trace_sha256'] != sha(trace) or fp['result_sha256'] != sha(payload_path):
            failures.append([ident, 'output_hash'])
        d = pd.read_csv(trace)
        dates = pd.to_datetime(d.delivery_date).values.astype('datetime64[D]')
        if len(d) != meta['n_days'] or len(set(d.delivery_date)) != len(d):
            failures.append([ident, 'support_length_or_duplicates'])
        seg, q = d.segment_id.to_numpy(), d.q.to_numpy()
        for p, metrics in payload['policies'].items():
            a = d['action_' + p].to_numpy(dtype=bool)
            s = np.where(a, 1, q)
            recomputed = {'action_total': int(a.sum()), 'mean_service': float(s.mean()),
                          'action_rate': float(a.mean()),
                          'rescued_shortfall_sum': float(((1-q)*a).sum())}
            for w in (3, 7, 14):
                eligible = [i for i in range(len(d)-w+1)
                            if np.all(seg[i:i+w] == seg[i]) and
                            np.all(np.diff(dates[i:i+w]) == np.timedelta64(1, 'D'))]
                recomputed['service_worst'+str(w)] = min(float(s[i:i+w].mean()) for i in eligible)
            for k, value in recomputed.items():
                if abs(metrics[k] - value) > 1e-11:
                    failures.append([ident, p, k, metrics[k], value])
            if a.sum() > meta['cap']:
                failures.append([ident, p, 'cap'])
            rows.append({'cohort': ident[0], 'origin': ident[1], 'policy': p, **recomputed})
    f = pd.DataFrame(rows)
    duplicate_identities = len(identities) != len(set(identities))
    counts = f[['cohort', 'origin']].drop_duplicates().groupby('cohort').size().to_dict()
    out = {'n_origin_units': len(identities), 'cohort_counts': counts,
           'duplicate_identities': duplicate_identities, 'failures': failures,
           'all_pass': len(identities) == 30 and not duplicate_identities and not failures,
           'macro': f.groupby(['cohort', 'policy'])[
               ['service_worst7', 'mean_service', 'action_total', 'action_rate']].mean().reset_index().to_dict('records')}
    Path(__file__).with_name('REVIEWER_ALLOCATION_RESULTS.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
