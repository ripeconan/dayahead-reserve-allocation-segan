"""Independent selected-date numeric ridge reconstruction; no production imports."""
import argparse,hashlib,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
import numpy as np
import pandas as pd
ZONES=('CAPITL','CENTRL','DUNWOD','GENESE','HUD VL','LONGIL','MHK VL','MILLWD','N.Y.C.','NORTH','WEST')
TZ='America/New_York'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    raw=pd.read_csv(a.raw,float_precision='round_trip')
    index=pd.DatetimeIndex(pd.to_datetime(raw['timestamp_utc'],utc=True)).tz_convert(TZ)
    grid=pd.date_range(index.min(),index.max(),freq='h')
    actual=pd.DataFrame(raw[[z+'_actual' for z in ZONES]].to_numpy(float),index=index).reindex(grid)
    official=pd.DataFrame(raw[[z+'_forecast' for z in ZONES]].to_numpy(float),index=index).reindex(grid)
    error=actual-official
    local=grid.tz_convert(TZ)
    clock=np.column_stack((np.sin(2*np.pi*local.hour/24),np.cos(2*np.pi*local.hour/24),
                           np.sin(2*np.pi*local.dayofweek/7),np.cos(2*np.pi*local.dayofweek/7)))
    x=np.column_stack((clock,error.shift(freq=pd.Timedelta(hours=72)).reindex(grid),
                      error.shift(freq=pd.Timedelta(hours=168)).reindex(grid),official))
    y=error.to_numpy(float)
    finite=np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
    records=[]
    for year in range(2019,2026):
        hourly=pd.read_csv(a.out/f'origin_{year}/hourly.csv.gz',float_precision='round_trip')
        emitted=pd.DataFrame(hourly[[f'issued_point_MW_{z}' for z in ZONES]].to_numpy(float),
                             index=pd.DatetimeIndex(pd.to_datetime(hourly['utc'],utc=True)).tz_convert(TZ))
        fold=pd.date_range(f'{year}-11-01',f'{year}-11-07',tz=TZ)
        fold_day=next(d for d in fold if d.dayofweek==6)
        for day in (pd.Timestamp(f'{year}-01-02',tz=TZ),pd.Timestamp(f'{year}-07-01',tz=TZ),fold_day):
            cutoff=day-pd.DateOffset(days=1)
            train=(grid>=cutoff-pd.DateOffset(days=365))&(grid<cutoff)&finite
            target=(grid>=day)&(grid<day+pd.DateOffset(days=1))&np.isfinite(x).all(axis=1)
            xx=x[train]; yy=y[train]
            assert len(xx)>=1000
            mean=xx.mean(axis=0); sd=xx.std(axis=0); sd[sd<1e-12]=1.
            design=np.column_stack((np.ones(len(xx)),(xx-mean)/sd))
            penalty=10*np.eye(design.shape[1]);penalty[0,0]=0.
            beta=np.linalg.solve(design.T@design+penalty,design.T@yy)
            test=np.column_stack((np.ones(int(target.sum())),(x[target]-mean)/sd))
            predicted=official.loc[grid[target]].to_numpy(float)+test@beta
            observed=emitted.loc[grid[target]].to_numpy(float)
            maxerr=float(np.max(np.abs(predicted-observed)))
            assert np.allclose(predicted,observed,atol=1e-7,rtol=1e-10),(year,str(day),maxerr)
            audit=pd.read_csv(a.out/f'origin_{year}/point_audit.csv.gz')
            row=audit[audit['local_date']==str(day.date())].iloc[0]
            assert int(row['train_rows'])==int(train.sum())
            assert pd.Timestamp(row['newest_train_label_utc'])==grid[train].max()
            assert grid[train].max()<cutoff
            records.append({'origin':year,'date':str(day.date()),'n_hours':int(target.sum()),
                            'train_rows':int(train.sum()),'newest_label':str(grid[train].max()),
                            'cutoff':str(cutoff),'max_error_MW':maxerr,'passed':True})
    report={'all_pass':len(records)==21,'reviewer_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest(),'records':records}
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'all_pass':report['all_pass'],'selected_dates':len(records),
                      'max_error_MW':max(r['max_error_MW'] for r in records)}))

if __name__=='__main__': main()
