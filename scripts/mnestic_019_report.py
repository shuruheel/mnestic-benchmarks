"""Summarize retained campaign records; incomplete runs cannot qualify."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def stats(rows):
    completed=[x for x in rows if not x.get('error')]
    lat=[x['ms'] for x in completed]
    call_seconds=sum(x['ms'] for x in rows)/1000
    recalls=[x['recall'] for x in rows if x.get('recall') is not None]
    return dict(n=len(rows),errors=sum(bool(x.get('error')) for x in rows),
                recall=float(np.mean(recalls)) if recalls else None,
                empty_rate=float(np.mean([x.get('empty',False) for x in rows])),
                duplicates=sum(x.get('duplicates',0) for x in rows),
                ineligible=sum(len(x.get('ineligible',[])) for x in rows),
                latency_ms={f'p{q}':float(np.percentile(lat,q)) for q in (50,95,99)} if lat else {},
                attempt_qps=len(rows)/call_seconds if call_seconds else None,
                successful_qps=len(completed)/call_seconds if call_seconds else None)


def bootstrap(builds,key):
    builds=[rows for rows in builds if rows]
    if not builds:return None
    rng=np.random.default_rng(19019)
    values=[]
    for _ in range(2000):
        sample=[]
        for i in rng.integers(0,len(builds),size=len(builds)):
            source=builds[i]
            sample.extend(source[j][key] for j in rng.integers(0,len(source),size=len(source)) if source[j].get(key) is not None)
        values.append(float(np.mean(sample)))
    return [float(x) for x in np.percentile(values,[2.5,97.5])]


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'runs':{},'E1':{},'E2':{},'E2_1536':{},'E3':{},'review_triggers':[]}
    groups=defaultdict(list);e2=defaultdict(list);e3=defaultdict(list)
    for path in sorted(a.root.glob('*.jsonl')):
        records=[json.loads(x) for x in path.read_text().splitlines()]
        manifest=records[0]['args'];backend=manifest['backend'];build=manifest['build']
        run={'complete':records[-1]['phase']=='complete','phases':{}}
        for phase in ('E1','E2','E2_1536','E3','freshness','mutation'):run['phases'][phase]=sum(x['phase']==phase for x in records)
        run['build']=next((x for x in records if x['phase']=='build'),None)
        run['resources']=records[-1] if run['complete'] else None
        run['freshness']={signal:dict(n=len(rows),hits=sum(x['hit'] for x in rows),errors=sum(bool(x['error']) for x in rows))
            for signal in ('exact','fts','graph','vector','decomposed','native')
            if (rows:=[x for x in records if x['phase']=='freshness' and x['signal']==signal])}
        run['rollback']=[x for x in records if x['phase']=='rollback']
        mutations=[x for x in records if x['phase']=='mutation']
        run['mutations']={'attempted':len(mutations),
                          'errors':sum(bool(x['error']) for x in mutations),
                          'repair_limit_rejections':sum(x['repair_limit'] for x in mutations),
                          'stops_on_first_error':True,
                          'records':mutations}
        for name in ('decomposed','native'):
            rows=[x for x in records if x['phase']=='E1' and x['path']==name]
            run[name]=stats(rows) if rows else None
            if rows:groups[(backend,build,name)].append(rows)
        report['runs'][path.stem]=run
        for x in records:
            if x['phase'] in ('E2','E2_1536') and 'ids' in x:
                e2[(x['phase'],backend,build,x['assignment'],str(x['selectivity']),x['strategy'])].append(x)
            if x['phase']=='E3':e3[(backend,build,x['dim'],x['shape'],x['changing'],x['query_kind'],x['mode'])].append(x)
    for key,builds in groups.items():
        rows=sum(builds,[]);report['E1']['/'.join(key)]=dict(stats(rows),builds=len(builds),mean_recall_ci=bootstrap(builds,'recall'),mean_completed_ms_ci=bootstrap([[row for row in sample if not row.get('error')] for sample in builds],'ms'))
    for key,rows in e2.items():
        value=stats(rows)
        fused=[x['fused_recall'] for x in rows if x.get('fused_recall') is not None]
        value['fused_recall']=float(np.mean(fused)) if fused else None
        report[key[0]]['/'.join(key[1:])]=value
    for key,rows in e3.items():
        report['E3']['/'.join(map(str,key))]={field:dict(mean=float(np.mean([x[field] for x in rows])),p95=float(np.percentile([x[field] for x in rows],95))) for field in ('serialization_ms','construction_ms','engine_ms','total_ms')}
    for backend in ('sqlite','rocksdb'):
        for path in ('decomposed','native'):
            base=report['E1'].get(f'{backend}/baseline/{path}');cand=report['E1'].get(f'{backend}/candidate/{path}')
            if not base or not cand:continue
            if not base['latency_ms'] or not cand['latency_ms']:continue
            delta=cand['latency_ms']['p95']-base['latency_ms']['p95']
            if base['recall']-cand['recall']>.01 or (delta>1 and delta>.1*base['latency_ms']['p95']):
                report['review_triggers'].append(dict(backend=backend,path=path,recall_loss=base['recall']-cand['recall'],p95_delta_ms=delta))
    plan_path=a.root/'plan.json'
    plan=json.loads(plan_path.read_text())['plan'] if plan_path.exists() else []
    expected={'E1':2000,'E2':2400,'E2_1536':2400,'E3':3200,'freshness':1200,'mutation':200}
    report['all_planned_runs_completed']=len(plan)==12 and all(
        (run:=report['runs'].get(item['name'])) and run['complete'] and run['phases']==expected
        for item in plan)
    for backend in ('sqlite','rocksdb'):
        for seed in (1234,1235,1236):
            for path in ('decomposed','native'):
                base=report['runs'].get(f'{backend}-{seed}-baseline',{}).get(path)
                cand=report['runs'].get(f'{backend}-{seed}-candidate',{}).get(path)
                if not base or not cand:continue
                if not base['latency_ms'] or not cand['latency_ms']:continue
                delta=cand['latency_ms']['p95']-base['latency_ms']['p95']
                if base['recall']-cand['recall']>.01 or (delta>1 and delta>.1*base['latency_ms']['p95']):
                    report['review_triggers'].append(dict(backend=backend,seed=seed,path=path,
                        recall_loss=base['recall']-cand['recall'],p95_delta_ms=delta))
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
