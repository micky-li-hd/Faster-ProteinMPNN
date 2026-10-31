"""Audit completed native training artifacts; never issues quality certification."""
import argparse
import json
import math
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    import torch
    result = json.loads((a.run/'RESULT.json').read_text())
    recipe = json.loads((a.run/'recipe.json').read_text())
    cleanup = json.loads((a.run/'cleanup.json').read_text())
    if result['status']!='PROCESS_COMPLETED' or cleanup['status']!='CLEAN':
        raise RuntimeError('Run did not complete with clean ownership cleanup')
    pattern = re.compile(r'epoch: (\d+), step: (\d+), time: ([^,]+), train: ([^,]+), valid: ([^,]+), train_acc: ([^,]+), valid_acc: ([^\n]+)')
    rows = []
    for m in pattern.finditer((a.run/'log.txt').read_text()):
        epoch, step = map(int,m.group(1,2))
        metrics = list(map(float,m.group(3,4,5,6,7)))
        if not all(math.isfinite(v) for v in metrics):
            raise RuntimeError('Nonfinite epoch metric')
        if rows and (epoch!=rows[-1]['epoch']+1 or step<=rows[-1]['step']):
            raise RuntimeError('Missing epoch or training updates')
        rows.append(dict(zip(['epoch','step','seconds','train_perplexity','valid_perplexity','train_accuracy','valid_accuracy'],[epoch,step,*metrics])))
    if len(rows)!=recipe['training']['num_epochs']:
        raise RuntimeError('Incomplete epoch count')
    # Only inspect a checkpoint generated locally by this known training run.
    ckpt = torch.load(a.run/'model_weights/epoch_last.pt',map_location='cpu',weights_only=True)
    if (ckpt['epoch'],ckpt['step'])!=(rows[-1]['epoch'],rows[-1]['step']):
        raise RuntimeError('Checkpoint/log counter mismatch')
    checked = 0
    def walk(x):
        nonlocal checked
        if isinstance(x,torch.Tensor):
            checked += 1
            if not torch.isfinite(x).all().item():
                raise RuntimeError('Nonfinite model or optimizer tensor')
        elif isinstance(x,dict):
            for v in x.values():walk(v)
        elif isinstance(x,(tuple,list)):
            for v in x:walk(v)
    walk(ckpt)
    if not checked:
        raise RuntimeError('No checkpoint tensors checked')
    report = {'status':'COMPLETE_FINITE_ARTIFACTS','epochs':rows,'tensors_checked':checked,
              'quality_accepted':False,'four_gate_certified':False,
              'scope':'Completeness and finiteness only; does not establish numerical equivalence or convergence.'}
    (a.run/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
