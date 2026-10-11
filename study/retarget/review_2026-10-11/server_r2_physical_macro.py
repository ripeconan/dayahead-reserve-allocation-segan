"""Independently verify all physical macro entries; existing transport only."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
work=a.work.resolve()
assert work.is_relative_to(Path('/home/icdm/card10_segan_retarget_20261011/releases'))
out=work/'analysis_r2/outputs/formal_r2'
units=[json.loads((out/f'origin_{y}/origin.json').read_text()) for y in range(2019,2026)]
summary=json.loads((out/'aggregate_summary.json').read_text())
fields={'macro_worst7_deficit_MWh':'worst7_deficit_MWh_complete_days',
        'macro_worst7_adequacy_fraction':'worst7_adequacy_fraction_complete_days',
        'macro_uncovered_MWh_finite_hours':'uncovered_MWh_finite_hours',
        'macro_extra_committed_MWh':'extra_committed_MWh',
        'macro_full_cost_MWh_equivalent':'full_cost_MWh_equivalent'}
count=0; keys=set()
for row in summary['physical_macro']:
    key=(row['policy'],row['eta'],row['kappa']);assert key not in keys;keys.add(key)
    cells=[next(c for c in unit['bridge_grid'] if (c['policy'],c['eta'],c['kappa'])==key) for unit in units]
    assert row['n_origins']==7
    for macro,cell in fields.items():
        expected=sum(c[cell] for c in cells)/7
        assert abs(row[macro]-expected)<=1e-7+abs(expected)*1e-10,(key,macro)
        count+=1
assert count==7*4*9*5
report={'all_pass':True,'physical_macro_entries_verified':count,'physical_cells':len(keys)}
(work/'REVIEWER_PHYSICAL_MACRO_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
