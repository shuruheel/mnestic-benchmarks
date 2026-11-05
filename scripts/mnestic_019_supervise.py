"""Serial, process-isolated paired runner. Retains failures and applies a run cap."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--candidate-sha',required=True)
p.add_argument('--candidate-wheel',type=Path,required=True)
p.add_argument('--baseline-wheel',type=Path,required=True)
p.add_argument('--baseline-python',required=True)
p.add_argument('--candidate-python',required=True)
p.add_argument('--only',help='optional exact backend-seed-build run identity for a declared rerun')
a=p.parse_args()
a.root.mkdir(parents=True,exist_ok=True)
plan=[]
for backend in ('sqlite','rocksdb'):
 for i,seed in enumerate((1234,1235,1236)):
  for build in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
   name=f'{backend}-{seed}-{build}'
   if a.only and a.only!=name:continue
   plan.append(dict(name=name,backend=backend,seed=seed,build=build))
plan_path=a.root/'plan.json'
if plan_path.exists():raise ValueError('fresh campaign root required')
plan_path.write_text(json.dumps(dict(harness_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 candidate_sha=a.candidate_sha,command=os.sys.argv,run_cap_seconds=7200,plan=plan),indent=2)+'\n')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONHASHSEED='0')
for run in plan:
 name=run['name'];build=run['build'];db=a.root/(name+'-db');out=a.root/(name+'.jsonl')
 python=getattr(a,build+'_python');wheel=getattr(a,build+'_wheel')
 sha=a.candidate_sha if build=='candidate' else '1d13aedc7a309105ae147f2ed1d79cd18a078b00'
 cmd=[python,'scripts/mnestic_019_campaign.py','--backend',run['backend'],'--seed',str(run['seed']),
      '--build',build,'--engine-sha',sha,'--artifact',str(wheel),'--output',str(out),'--database',str(db)]
 print(time.strftime('%H:%M:%S'),name,flush=True)
 start=time.monotonic()
 with (a.root/(name+'.log')).open('x') as log:
  try:
   result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
   status=dict(exit_code=result.returncode,timed_out=False)
  except subprocess.TimeoutExpired:
   status=dict(exit_code=None,timed_out=True)
 status.update(run=run,elapsed_seconds=time.monotonic()-start,command=cmd)
 (a.root/(name+'-status.json')).write_text(json.dumps(status,indent=2)+'\n')
 # Only the freshly created, explicit run database is regenerable. Raw data stay.
 if db.exists():shutil.rmtree(db)
 print(status,flush=True)
