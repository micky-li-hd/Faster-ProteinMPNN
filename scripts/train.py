"""Bounded single-GPU native training with an exclusive lock and owned cleanup."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from ownership import OwnedProcesses, OWNERSHIP_ENV, write

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preflight(data, recipe):
    manifest = json.loads((ROOT/'source_manifest.json').read_text())
    for name, expected in manifest.items():
        if digest(REPO/name) != expected:
            raise RuntimeError('Source binding mismatch: '+name)
    required = ['list.csv', 'valid_clusters.txt', 'test_clusters.txt']
    for name in required:
        if not (data/name).is_file() or (data/name).stat().st_size == 0:
            raise RuntimeError('Missing or empty data file: '+name)
    if not (data/'pdb').is_dir():
        raise RuntimeError('Missing pdb data directory')
    if set(recipe['training']) != set(json.loads((ROOT/'full_training.json').read_text())['training']):
        raise RuntimeError('Recipe keys differ; debug/precision/hidden overrides are not permitted.')
    if any(recipe['training'][key] <= 0 for key in ['num_epochs','num_examples_per_epoch','batch_size','max_protein_length']):
        raise RuntimeError('Invalid workload')
    return {'source_sha256':manifest, 'split_sha256':{k:digest(data/k) for k in required},
            'data_scope':'Directory layout checked; this alone does not certify complete dataset coverage.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['reference','candidate'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recipe', type=Path, default=ROOT/'full_training.json')
    p.add_argument('--gpu', default='0')
    p.add_argument('--lock', type=Path, required=True, help='Use the existing shared campaign lock path.')
    p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.timeout <= 0 or ',' in a.gpu:
        p.error('Positive timeout and exactly one GPU are required')
    data = a.data.resolve(strict=True)
    recipe = json.loads(a.recipe.read_text())
    checks = preflight(data, recipe)
    if a.check_only:
        print(json.dumps(checks, indent=2))
        return
    out = a.out.resolve()
    if '%' in str(out):
        p.error('Output path cannot contain upstream strftime substitutions')
    checkpoint = str(a.checkpoint.resolve(strict=True)) if a.checkpoint else ''
    if checkpoint:
        checks['checkpoint_sha256'] = digest(Path(checkpoint))
    out.mkdir(parents=True, exist_ok=False)
    with a.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OwnedProcesses.supported()
        gpu = subprocess.check_output(['nvidia-smi','-i',a.gpu,
            '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        if len(gpu)!=3 or float(gpu[1])>256 or float(gpu[2])>5 or gpu[0] in apps:
            raise RuntimeError('Selected GPU is busy; no process was launched or interrupted.')
        nonce = secrets.token_hex(32)
        owner = OwnedProcesses(nonce, out/'cleanup.json')
        cache = out/'cache'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONHASHSEED=str(recipe['seed']),
                   MPNN_RUN_SEED=str(recipe['seed']), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'), TRITON_CACHE_DIR=str(cache/'triton'))
        env[OWNERSHIP_ENV] = nonce
        write(out/'recipe.json', recipe)
        command = [sys.executable,'-u',str(ROOT/'seeded_entry.py'),str(REPO/'reference/training' if a.variant == 'reference' else REPO/'training'),
                   str(out/'recipe.json'),str(data),str(out),checkpoint]
        write(out/'launch.json', {'variant':a.variant,'command':command,'checks':checks,
                                 'recipe_sha256':digest(a.recipe),'timeout':a.timeout,
                                 'seed_controls':'Per-prefetch-call and worker seeds identical across arms',
                                 'four_gate_certified':False})
        process = None
        start = time.monotonic()
        status = 'ERROR'
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Signal '+str(signum))
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, interrupted)
        try:
            with (out/'stdout.log').open('w') as log:
                process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code = process.wait(timeout=a.timeout)
                status = 'PROCESS_COMPLETED' if code==0 else 'PROCESS_FAILED'
                if code:
                    raise RuntimeError('Training exited with code '+str(code))
        except subprocess.TimeoutExpired:
            status = 'TIMEOUT'
            raise
        finally:
            elapsed = time.monotonic()-start
            try:
                owner.cleanup(process, reason=status)
            finally:
                write(out/'RESULT.json',{'status':status,'process_seconds':elapsed,
                      'four_gate_certified':False,'quality_accepted':False,
                      'scope':'Process wall time including cold compile, data loading, validation and checkpoint writing; not a speedup result.'})


if __name__ == '__main__':
    main()
