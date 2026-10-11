"""Independent constraint/artifact checks on all seven emitted oracle units."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'analysis_r1' / 'outputs'


def main():
    failures, rows = [], []
    for trace in sorted((BASE/'allocation_value').glob('nyiso_exposed*__daily.csv.gz')):
        d = pd.read_csv(trace)
        p = trace.with_name(trace.name.removesuffix('__daily.csv.gz')+'.json')
        meta = json.loads(p.read_text(encoding='utf-8'))['meta']
        origin = meta['origin']
        path = BASE/'budget_oracle'/('nyiso__'+origin.replace('->','_')+'.json')
        side_path = path.with_name(path.stem+'.sidecar.json')
        unit = json.loads(path.read_text(encoding='utf-8'))
        if not side_path.is_file():
            failures.append([origin, 'sidecar_missing'])
            continue
        side = json.loads(side_path.read_text(encoding='utf-8'))
        if side['unit_file_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            failures.append([origin, 'unit_file_hash'])
        q, seg = d.q.to_numpy(), d.segment_id.to_numpy()
        indices = [(i, i+6) for i in range(len(d)-6) if np.all(seg[i:i+7] == seg[i])]
        for key, expected_level in [('oracle_count', int(d.action_P.sum())), ('oracle_cap', meta['cap'])]:
            o = unit[key]
            a = np.array(o['action_vector'])
            checks = {'binary': bool(np.all(np.isin(a, [0,1]))), 'length': len(a) == len(q),
                      'expected_level': o['level'] == expected_level,
                      'mode': o['mode'] == key.removeprefix('oracle_')}
            s = np.where(a, 1.0, q)
            means = [float(s[i:j+1].mean()) for i,j in indices]
            z, rescue = min(means), float(((1-q)*a).sum())
            checks.update({'cap_or_count': bool(a.sum() == expected_level if key == 'oracle_count' else a.sum() <= expected_level),
                           'worst7_matches': abs(z-o['worst7']) < 1e-10,
                           'z_optimum_preserved': z >= o['z1']-1.1e-9,
                           'rescue_matches': abs(rescue-o['rescued_shortfall_sum']) < 1e-10,
                           'rescue_optimum_preserved': rescue >= o['stage2_rescue_optimum']-1.1e-9,
                           'window_indices_complete': [(w['w0'],w['w_end']) for w in o['windows']] == indices,
                           'window_means_match': len(o['windows']) == len(means) and all(abs(w['mean']-m)<1e-10 for w,m in zip(o['windows'],means)),
                           'three_stages_optimal': len(o['diagnostics']) == 3 and all(x['status'] == 0 and abs(x['mip_gap']) < 1e-9 for x in o['diagnostics'])})
            for k,v in checks.items():
                if not v:
                    failures.append([origin, key, k])
            rows.append({'origin': origin, 'oracle': key, 'worst7': z, 'rescue': rescue,
                         'actions': int(a.sum()), 'all_pass': all(checks.values())})
    out = {'n_origins': len({r['origin'] for r in rows}), 'n_oracle_objects': len(rows),
           'failures': failures, 'all_pass': len(rows) == 14 and not failures, 'rows': rows}
    Path(__file__).with_name('REVIEWER_ORACLE_ARTIFACT_RESULTS.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
