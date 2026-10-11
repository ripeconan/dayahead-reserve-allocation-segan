"""Independently check rounded publication cells and SHA bindings, no production import."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
MS=ROOT/'manuscript_segan'
FORMAL=ROOT/'analysis_r2/outputs/formal_r2'
AGG=json.loads((FORMAL/'aggregate_summary.json').read_text())
UNITS={y:json.loads((FORMAL/f'origin_{y}/origin.json').read_text()) for y in range(2019,2026)}
POLICIES=('none','P','F1','F2','F3','F4','F5')
MAC={r['policy']:r for r in AGG['macro']}
CHECKS=[]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name,predicate):
    return [[v.strip().rstrip('\\').strip() for v in line.split('&')]
            for line in (MS/'tables'/name).read_text().splitlines() if '&' in line and predicate(line.strip())]
def num(cell,expected,where):
    clean=cell.replace(',','').strip()
    assert re.fullmatch(r'[+-]?\d+(\.\d+)?',clean),(where,cell)
    decimals=len(clean.split('.')[1]) if '.' in clean else 0
    assert abs(float(clean)-expected)<=.50001*10**(-decimals)+1e-9,(where,cell,expected)
    CHECKS.append(where)
def numbers(cell):return re.findall(r'[+-]?\d+\.\d+',cell)
def contrast(comp,endpoint,physical=False):
    return next(r for r in AGG['physical_contrasts' if physical else 'contrasts']
                if r['contrast']==f'P_minus_{comp}' and r['endpoint']==endpoint
                and (not physical or r['eta']==1.0 and r['kappa']==0.0))
def phys(p,eta=1.,k=0.):
    return next(r for r in AGG['physical_macro'] if r['policy']==p and r['eta']==eta and r['kappa']==k)
def annual(comp,endpoint):
    diffs=[u['allocator_metrics']['P'][endpoint]-u['allocator_metrics'][comp][endpoint] for u in UNITS.values()]
    return sum(v>1e-12 for v in diffs),sum(abs(v)<=1e-12 for v in diffs),diffs
def stat_cells(cells,row,where,scale=1.):
    expected=[row['effect'],row['ci_lo'],row['ci_hi'],min(row['loo'].values()),max(row['loo'].values())]
    actual=[cells[0],*numbers(cells[1]),*numbers(cells[2])]
    assert len(actual)==len(expected),(where,actual)
    for i,(value,target) in enumerate(zip(actual,expected)):num(value,target/scale,f'{where}/{i}')

manifest=json.loads((MS/'r2_assets/r2_input_manifest.json').read_text())
for rel,value in manifest.items():
    if rel=='aggregate_summary.json' or rel.startswith('origin_'):
        assert sha(FORMAL/rel)==value,rel
assert sha(MS/manifest['generator']['path'])==manifest['generator']['sha256']
for rel,identity in manifest['generated_artifacts'].items():
    assert sha(MS/rel)==identity['sha256'] and (MS/rel).stat().st_size==identity['bytes'],rel

service=rows('r2_tab1_service.tex',lambda l:l.split()[0] in POLICIES)
assert len(service)==7
keys=['macro_mean_service','macro_service_worst3','macro_service_worst7','macro_action_total_all_declared',
      'macro_action_rate_all_declared','macro_blocked_trigger_days','macro_rescue_precision','macro_L_lambda_1']
for row in service:
    p=row[0].split()[0]
    for i,key in enumerate(keys):
        target=MAC[p][key]*(100 if i==4 else 1)
        if math.isnan(target):assert row[i+1]=='--'
        else:num(row[i+1],target,f'service/{p}/{key}')

main_con=rows('r2_tab2_contrasts.tex',lambda l:l.startswith('P $-$'))
assert len(main_con)==6
for row in main_con:
    comp=row[0].split()[-1];stat_cells(row[1:4],contrast(comp,'service_worst7'),f'W7/{comp}')
    count,ties,_=annual(comp,'service_worst7')
    assert row[4].startswith(f'{count}/7'),(comp,row[4],count)

physical=rows('r2_tab3_physical.tex',lambda l:l.split()[0] in POLICIES)
assert len(physical)==7
pkeys=['macro_uncovered_MWh_finite_hours','macro_worst7_deficit_MWh','macro_worst7_adequacy_fraction','macro_extra_committed_MWh']
for row in physical:
    p=row[0].split()[0]
    targets=[phys(p)[k]/(1e6 if i==3 else 1) for i,k in enumerate(pkeys)]
    targets+=[MAC[p]['macro_action_total_all_declared']]
    for i,(cell,value) in enumerate(zip(row[1:],targets)):num(cell,value,f'physical/{p}/{i}')

endpoint_labels={'Uncovered':'uncovered_MWh_finite_hours','$W_7$ deficit':'worst7_deficit_MWh_complete_days',
                 '$W_7$ adequacy':'worst7_adequacy_fraction_complete_days','Extra booked':'extra_committed_MWh'}
con=rows('r2_tab4_physical_contrasts.tex',lambda l:l.startswith('P $-$'))
assert len(con)==6
for row in con:
    comp=row[0].split()[-1];endpoint=endpoint_labels[row[1]]
    stat_cells(row[3:6],contrast(comp,endpoint,True),f'physical-contrast/{comp}/{endpoint}',1e6 if endpoint=='extra_committed_MWh' else 1)

cost=rows('r2_tab5_cost.tex',lambda l:bool(re.match(r'^\d',l)))
assert len(cost)==9
for row in cost:
    k=float(row[0]);n=phys('none',1.,k)['macro_full_cost_MWh_equivalent'];p=phys('P',1.,k)['macro_full_cost_MWh_equivalent']
    for i,(cell,value) in enumerate(zip(row[1:],[n,p,p-n])):num(cell,value,f'cost/{k}/{i}')

per=rows('r2_sup_perorigin.tex',lambda l:bool(re.match(r'^20\d\d &',l)))
assert len(per)==49
for row in per:
    y=int(row[0]);p=row[1];unit=UNITS[y]['allocator_metrics'][p]
    for cell,key in zip(row[2:],['mean_service','service_worst3','service_worst7','action_total_all_declared','blocked_trigger_days']):
        num(cell,unit[key],f'per-origin/{y}/{p}/{key}')

sup=rows('r2_sup_contrasts.tex',lambda l:l.startswith('P $-$'))
assert len(sup)==30
labels={'mean':'mean_service','$W_3$':'service_worst3','$W_7$':'service_worst7','$W_{14}$':'service_worst14','$L_1$':'L_lambda_1'}
for row in sup:
    comp=row[0].split()[-1];endpoint=labels[row[1]]
    stat_cells(row[2:5],contrast(comp,endpoint),f'supplement-contrast/{comp}/{endpoint}')
    count,ties,_=annual(comp,endpoint);assert row[5].startswith(f'{count}/7'),(comp,endpoint,row[5],count)

sup_phys=rows('r2_sup_physical.tex',lambda l:l.split()[0] in POLICIES)
assert len(sup_phys) in (21,28)
for row in sup_phys:
    p=row[0];eta=float(row[1])
    for i,(cell,key) in enumerate(zip(row[2:],pkeys)):
        num(cell,phys(p,eta)[key]/(1e6 if i==3 else 1),f'supplement-physical/{p}/{eta}/{key}')

sup_cost=rows('r2_sup_cost.tex',lambda l:bool(re.match(r'^\d',l)))
assert len(sup_cost)==9
for row in sup_cost:
    k=float(row[0])
    for cell,(p,eta) in zip(row[1:],[('none',1.),('none',2.),('P',1.),('P',2.)]):
        num(cell,phys(p,eta,k)['macro_full_cost_MWh_equivalent'],f'supplement-cost/{p}/{eta}/{k}')

derived=json.loads((MS/'r2_assets/r2_derived_values.json').read_text())
for eta in (.5,1.,2.):
    expected=(phys('none',eta)['macro_uncovered_MWh_finite_hours']-phys('P',eta)['macro_uncovered_MWh_finite_hours'])/phys('P',eta)['macro_extra_committed_MWh']
    assert abs(derived['kappa_star'][str(eta)]-expected)<1e-15
report={'all_pass':True,'rounded_numeric_cells_verified':len(CHECKS),'input_bindings':8,
        'generated_artifacts_verified':len(manifest['generated_artifacts']),
        'annual_W7_positive_counts':{p:annual(p,'service_worst7')[:2] for p in ('none','F1','F2','F3','F4','F5')},
        'asset_manifest_sha256':sha(MS/'r2_assets/r2_input_manifest.json'),
        'reviewer_script_sha256':sha(Path(__file__)),'checks':CHECKS}
(ROOT/'review_2026-10-11/REVIEWER_PUBLICATION_ASSET_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
