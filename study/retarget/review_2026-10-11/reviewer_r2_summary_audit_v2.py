"""Independently reconstruct every reported paired-origin effect/CI/LOO."""
import argparse,hashlib,json,os
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import numpy as np

YEARS=tuple(range(2019,2026))
POLICIES=('none','P','F1','F2','F3','F4','F5')
ETAS=(0.,.5,1.,2.)
KAPPAS=(0.,.001,.005,.01,.025,.05,.1,.25,.5)
MODELED=('service_worst3','service_worst7','service_worst14','mean_service','L_lambda_1')
PHYSICAL=('worst7_deficit_MWh_complete_days','worst7_adequacy_fraction_complete_days',
          'uncovered_MWh_finite_hours','extra_committed_MWh','full_cost_MWh_equivalent')
PAIRS=[(f'{p}_minus_none',p,'none') for p in POLICIES if p!='none']
PAIRS += [(f'P_minus_{p}','P',p) for p in POLICIES if p not in ('none','P')]

def check_num(actual,expected,label):
    if not np.isclose(actual,expected,atol=1e-7,rtol=1e-10):
        raise AssertionError(f'{label}: {actual} != {expected}')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    units={y:json.loads((args.out/f'origin_{y}/origin.json').read_text(encoding='utf-8')) for y in YEARS}
    summary=json.loads((args.out/'aggregate_summary.json').read_text(encoding='utf-8'))
    assert summary['n_origins']==7 and summary['grid_complete']
    assert summary['grid_cells']==summary['expected_grid_cells']==7*7*4*9
    modeled={(r['contrast'],r['endpoint']):r for r in summary['contrasts']}
    assert len(modeled)==len(summary['contrasts'])==11*5
    physical={(r['contrast'],r['endpoint'],r['eta'],r['kappa']):r for r in summary['physical_contrasts']}
    assert len(physical)==len(summary['physical_contrasts'])==11*5*4*9
    grid={y:{(r['policy'],r['eta'],r['kappa']):r for r in units[y]['bridge_grid']} for y in YEARS}
    # Independent shared random-index design, same declared seed/draw count.
    indices=np.random.default_rng(20261011).integers(0,7,size=(10000,7))
    # Establish equivalence between independently drawn indices and the
    # protocol's original length-seven rng.choice calls on an asymmetric toy.
    toy=np.asarray([-.31,.017,.9,-.004,2.7,.13,-1.])
    original_rng=np.random.default_rng(20261011)
    original=np.asarray([original_rng.choice(toy,size=7,replace=True).mean() for _ in range(10000)])
    assert np.array_equal(original,toy[indices].mean(axis=1))
    inspected=[]
    def verify(row,diffs,key):
        assert row['n_origins']==7 and row['n_boot']==10000 and row['seed']==20261011
        diffs=np.asarray(diffs,float)
        assert np.isfinite(diffs).all()
        draws=diffs[indices].mean(axis=1)
        expected={'effect':float(diffs.mean()),'ci_lo':float(np.quantile(draws,.025)),
                  'ci_hi':float(np.quantile(draws,.975)),'share_positive':float((draws>0).mean())}
        for field,value in expected.items(): check_num(row[field],value,str(key)+':'+field)
        assert set(row['loo'])=={str(y) for y in YEARS}
        for i,y in enumerate(YEARS):
            check_num(row['loo'][str(y)],np.delete(diffs,i).mean(),str(key)+':LOO'+str(y))
        inspected.append({'key':key,**expected,'per_origin_differences':diffs.tolist()})
    for label,left,right in PAIRS:
        for endpoint in MODELED:
            verify(modeled[(label,endpoint)],
                   [units[y]['allocator_metrics'][left][endpoint]-units[y]['allocator_metrics'][right][endpoint] for y in YEARS],
                   (label,endpoint))
        for eta in ETAS:
            for kappa in KAPPAS:
                for endpoint in PHYSICAL:
                    verify(physical[(label,endpoint,eta,kappa)],
                           [grid[y][(left,eta,kappa)][endpoint]-grid[y][(right,eta,kappa)][endpoint] for y in YEARS],
                           (label,endpoint,eta,kappa))
    macro={r['policy']:r for r in summary['macro']}
    assert set(macro)==set(POLICIES)
    undefined_rescue=[]
    for policy,row in macro.items():
        assert row['n_origins']==7
        for name,value in row.items():
            if name.startswith('macro_'):
                endpoint=name[6:]
                values=np.asarray([units[y]['allocator_metrics'][policy][endpoint] for y in YEARS],float)
                if endpoint=='rescue_precision':
                    # Zero-action origins have no rescue denominator. Verify the
                    # exact reason for every NaN and the available-origin mean.
                    counts=np.asarray([units[y]['allocator_metrics'][policy]['action_total_observed'] for y in YEARS])
                    assert np.array_equal(np.isnan(values),counts==0),('undefined rescue',policy)
                    assert not np.isinf(values).any()
                    valid=values[np.isfinite(values)]
                    undefined_rescue.append({'policy':policy,'n_defined':int(len(valid)),
                                             'n_zero_action':int((counts==0).sum())})
                    if not len(valid):
                        assert np.isnan(value),('zero-action macro rescue',policy)
                        continue
                    expected=float(valid.mean())
                else:
                    assert np.isfinite(values).all(),(policy,endpoint)
                    expected=float(values.mean())
                check_num(value,expected,'macro '+policy+':'+endpoint)
    report={'all_pass':True,'paired_contrasts_verified':len(inspected),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'aggregate_sha256':hashlib.sha256((args.out/'aggregate_summary.json').read_bytes()).hexdigest(),
            'inference':'descriptive seven exposed annual origins; unchanged10000draws/seed20261011',
            'undefined_rescue_diagnostics':undefined_rescue,'effects':inspected}
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'all_pass':True,'paired_contrasts_verified':len(inspected)}))

if __name__=='__main__': main()
