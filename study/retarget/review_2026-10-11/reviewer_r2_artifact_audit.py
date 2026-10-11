"""Independent numerical reconstruction from emitted evidence, no production imports.

Run after root admits the frozen experiment. Also usable on synthetic emitted
artifacts. Checks reference geometry, hourly supports, modeled endpoints,
causal P actions, and every finite-capacity/cost cell from underlying rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
import pandas as pd
from sklearn.covariance import OAS

ZONES = ('CAPITL','CENTRL','DUNWOD','GENESE','HUD VL','LONGIL','MHK VL',
         'MILLWD','N.Y.C.','NORTH','WEST')
POLICIES = ('none','P','F1','F2','F3','F4','F5')
ETAS = (0., .5, 1., 2.)
KAPPAS = (0., .001, .005, .01, .025, .05, .1, .25, .5)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def near(a, b, label, atol=1e-8):
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    if aa.shape != bb.shape or not np.allclose(aa, bb, atol=atol, rtol=1e-10, equal_nan=True):
        raise AssertionError(label + ': numerical mismatch')

def windows(dates, segments, width):
    ordinal = np.asarray([pd.Timestamp(d).toordinal() for d in dates])
    return [np.arange(i, i+width) for i in range(len(dates)-width+1)
            if np.all(np.diff(ordinal[i:i+width]) == 1)
            and len(set(segments[i:i+width])) == 1]

def audit_origin(directory, raw=None):
    p = json.loads((directory/'origin.json').read_text(encoding='utf-8'))
    manifest_path = directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert (directory/'manifest.json.sha256').read_text().strip() == sha(manifest_path)
    for name, entry in manifest['entries'].items():
        assert sha(directory/name) == entry['sha256'], name
        assert (directory/name).stat().st_size == entry['bytes'], name
    def read(name): return pd.read_csv(directory/(name+'.csv.gz'), low_memory=False, float_precision='round_trip')
    hours, daily, physical = read('hourly'), read('daily'), read('bridge_daily')
    ref, scores = read('reference_residuals'), read('reference_scores')
    year = int(p['origin'])
    expected_dates = pd.date_range(f'{year}-01-02', f'{year}-12-31').strftime('%Y-%m-%d').tolist()
    assert daily['delivery_date'].tolist() == expected_dates
    assert not hours['utc'].duplicated().any()
    stamp = pd.to_datetime(hours['utc'], utc=True)
    assert stamp.is_monotonic_increasing
    local = stamp.dt.tz_convert('America/New_York')
    assert local.dt.strftime('%Y-%m-%d').tolist() == hours['delivery_date'].tolist()
    assert local.dt.hour.tolist() == hours['slot'].tolist()
    near(hours['duration_h'], np.ones(len(hours)), 'real UTC durations')
    rr = ref[list(ZONES)].to_numpy(float)
    assert np.isfinite(rr).all()
    ref_stamp = pd.to_datetime(ref['utc'], utc=True)
    ref_local = ref_stamp.dt.tz_convert('America/New_York')
    assert set(ref_local.dt.year) == {year-1}
    assert ref_local.dt.strftime('%Y-%m-%d').tolist() == ref['local_date'].tolist()
    assert ref['utc'].tolist() == scores['utc'].tolist()
    fit = OAS(assume_centered=False).fit(rr)
    center, cov = rr.mean(axis=0), fit.covariance_
    near(center, p['geometry']['center'], 'reference center')
    near(cov, p['geometry']['covariance'], 'reference OAS covariance', 1e-7)
    precision = np.linalg.inv(cov)
    score = np.einsum('ij,jk,ik->i', rr-center, precision, rr-center)
    near(score, scores['score'], 'reference scores', 1e-7)
    headroom = np.quantile(np.maximum(rr, 0.), .95, axis=0, method='linear')
    near(headroom, [p['bridge_headroom_MW'][z] for z in ZONES], 'reference H')
    residual = hours[[f'residual_MW_{z}' for z in ZONES]].to_numpy(float)
    point = hours[[f'issued_point_MW_{z}' for z in ZONES]].to_numpy(float)
    actual = hours[[f'actual_load_MW_{z}' for z in ZONES]].to_numpy(float)
    near(actual-point, residual, 'corrected residuals')
    threshold = hours['threshold'].to_numpy(float)
    support = np.maximum(center + np.sqrt(np.maximum(threshold[:,None]*np.diag(cov), 0.)), 0.)
    near(support, hours[[f'r0_{z}' for z in ZONES]], 'upper support reserve', 1e-7)
    current_score = np.einsum('ij,jk,ik->i', residual-center, precision, residual-center)
    observed = np.isfinite(residual).all(axis=1)
    if raw is not None:
        lookup = raw.set_index(pd.to_datetime(raw['timestamp_utc'], utc=True))
        near(actual, lookup.loc[pd.DatetimeIndex(stamp), [f'{z}_actual' for z in ZONES]], 'raw test actuals')
        near(rr + ref[[f'issued_point_MW_{z}' for z in ZONES]].to_numpy(float),
             lookup.loc[pd.DatetimeIndex(ref_stamp), [f'{z}_actual' for z in ZONES]], 'raw reference actuals')
    day_indices = {str(d): np.flatnonzero(hours['delivery_date'].to_numpy() == d)
                   for d in hours['delivery_date'].unique()}
    expected_hours = {}
    for d in expected_dates:
        start = pd.Timestamp(d, tz='America/New_York')
        expected_hours[d] = int((start+pd.DateOffset(days=1)-start).total_seconds()/3600)
    for row in daily.itertuples(index=False):
        if row.scorable:
            ii = day_indices[row.delivery_date]
            assert len(ii) == expected_hours[row.delivery_date]
            if row.observed:
                assert observed[ii].all()
                near(row.q, np.mean(current_score[ii] <= threshold[ii]), 'daily region coverage')
    walks, metrics = {}, {}
    for policy in POLICIES:
        w = read('walk_'+policy)
        walks[policy] = w
        assert w['delivery_date'].tolist() == expected_dates
        assert w['segment'].tolist() == daily['segment'].tolist()
        a = w['action'].to_numpy(int)
        assert np.isin(a, [0,1]).all() and a.sum() <= round(.15*len(w))
        support_days = w['scorable'].to_numpy(bool) & w['observed'].to_numpy(bool) & np.isfinite(w['q_base'])
        near(w.loc[support_days,'q_base'], daily.loc[support_days,'q'], 'walk common q')
        served = np.where(a, 1., w['q_base'])
        near(w['service'], np.where(support_days,served,np.nan), 'modeled service')
        near(w.loc[w['scorable'],'used_after'], np.cumsum(a)[w['scorable']], 'retained cumulative actions')
        mm = p['allocator_metrics'][policy]
        near(mm['mean_service'], served[support_days].mean(), 'mean modeled service')
        near(mm['action_total_all_declared'], a.sum(), 'declared action count')
        near(mm['L_lambda_1'], 1-served[support_days].mean()+a[support_days].mean(), 'modeled loss')
        ds = w.loc[support_days,'delivery_date'].tolist()
        gs = w.loc[support_days,'segment'].to_numpy()
        for width in (3,7,14):
            candidates = windows(ds,gs,width)
            near(mm[f'service_worst{width}'], min(served[support_days][ii].mean() for ii in candidates), 'modeled worst window')
        metrics[policy] = mm
        if policy == 'none': assert a.sum() == 0
    # Independent causal reconstruction of P using only eligible served feedback.
    w = walks['P']; used=0; cap=round(.15*len(w))
    for i, row in enumerate(w.itertuples(index=False)):
        if not row.scorable: assert row.action == 0; continue
        prior = w.iloc[:i]
        eligible = prior[(prior['segment']==row.segment) & prior['service'].notna()
                         & (pd.to_datetime(prior['delivery_date']) <= pd.Timestamp(row.delivery_date)-pd.Timedelta(days=2))]
        deficit = float((1-eligible['service'].tail(6)).sum())
        wanted = int(deficit >= 1. and used < cap)
        assert int(row.action) == wanted, ('P causal action',row.delivery_date,deficit,used,row.action)
        used += wanted
    assert set(zip(physical['policy'],physical['eta'])) == {(x,e) for x in POLICIES for e in ETAS}
    cells = {(x['policy'],x['eta'],x['kappa']):x for x in p['bridge_grid']}
    assert len(cells) == len(p['bridge_grid']) == 7*4*9
    base = float(support.sum())
    near(base, p['bridge_base_reserved_MWh'], 'common base reserve booking', 1e-6)
    for policy in POLICIES:
        action = walks[policy].set_index('delivery_date')['action'].to_dict()
        for eta in ETAS:
            f = physical[(physical['policy']==policy)&(physical['eta']==eta)].reset_index(drop=True)
            assert not f['delivery_date'].duplicated().any()
            assert set(f['delivery_date']) == set(day_indices)
            for row in f.itertuples(index=False):
                ii=day_indices[row.delivery_date]; obs=ii[observed[ii]]
                allowed=support[obs]+int(action[row.delivery_date])*eta*headroom
                deficit=float(np.maximum(residual[obs]-allowed,0.).sum())
                adequate=float(np.all(residual[obs] <= allowed,axis=1).sum())
                near(row.deficit_MWh,deficit,'physical daily deficit',1e-6)
                near(row.observed_hours,len(obs),'observed physical hours')
                near(row.adequate_hours,adequate,'physical adequate hours')
                near(row.extra_committed_MWh,action[row.delivery_date]*eta*headroom.sum()*expected_hours[row.delivery_date], 'retained physical booking',1e-6)
                assert bool(row.complete_observed_day) == (len(obs)==expected_hours[row.delivery_date])
            complete=f[f['complete_observed_day']].reset_index(drop=True)
            ww=windows(complete['delivery_date'].tolist(),complete['segment'].to_numpy(),7)
            wd=max(complete['deficit_MWh'].to_numpy()[ii].sum() for ii in ww)
            wa=min(complete['observed_adequate_fraction'].to_numpy()[ii].mean() for ii in ww)
            uncovered=float(f['deficit_MWh'].sum()); extra=float(f['extra_committed_MWh'].sum())
            for kappa in KAPPAS:
                cell=cells[(policy,eta,kappa)]
                for key,val in [('worst7_deficit_MWh_complete_days',wd),('worst7_adequacy_fraction_complete_days',wa),
                                ('uncovered_MWh_finite_hours',uncovered),('extra_committed_MWh',extra),
                                ('full_cost_MWh_equivalent',kappa*(base+extra)+uncovered)]:
                    near(cell[key],val,'physical grid '+key,1e-6)
    return {'origin':year,'passed':True,'hour_rows':len(hours),'reference_rows':len(ref),
            'bridge_daily_rows':len(physical),'all_grid_cells':len(cells),
            'allocator_metrics':metrics,'manifest_sha256':sha(manifest_path)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    raw=pd.read_csv(args.raw,low_memory=False,float_precision='round_trip') if args.raw else None
    records=[]
    for directory in sorted(args.out.glob('origin_*')):
        try: records.append(audit_origin(directory,raw))
        except Exception as exc:
            import traceback
            records.append({'origin_dir':directory.name,'passed':False,'error':str(exc),'traceback':traceback.format_exc()})
    report={'reviewer_script_sha256':sha(Path(__file__)),'records':records,
            'all_pass':bool(records) and all(r['passed'] for r in records)}
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'all_pass':report['all_pass'],'origins':len(records),'failures':[r for r in records if not r['passed']]}))
    return 0 if report['all_pass'] else 1

if __name__=='__main__': raise SystemExit(main())
