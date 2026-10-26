# Faster-ProteinMPNN

Infrastructure acceleration of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) training, inspired by [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent). Pinned reference and optimized source, a bounded reproduction entry point, and observed results are included.

**Status: external exploratory case; not ResearchBoost four-gate certified.** The completed experiment covers six epochs, not the full 150-epoch recipe or a convergence guarantee.

## Results

One NVIDIA A100 SXM4 40 GB, Python 3.12.11, PyTorch 2.5.1+cu121. The official full dataset and v_48_020 per-epoch settings were retained; the experiment was bounded to six epochs from scratch. Runs were sequential reference → candidate → reference, with isolated cold compiler caches. Each completed 8,970 optimizer updates.

| Measurement | Reference 1 | Candidate | Reference 2 | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Whole process, seconds | 3316.216 | 2154.895 | 3316.182 | **1.54×** |
| Mean native epoch time, epochs 2–6, seconds | 449.46 | 200.52 | 451.84 | **2.25×** |
| Final validation perplexity | 5.934 | 5.884 | 5.897 | — |
| Final validation sequence accuracy | 0.445 | 0.447 | 0.446 | — |

Ratios use the mean of the two reference measurements divided by the candidate. Whole-process time includes cold compilation, loading, validation and checkpoint writing. Native epoch timing is a narrower upstream log scope; 2.25× is not the end-to-end claim. This is one R-C-R group, not multiple independent seeds or a confidence interval.

All three completed-run audits checked 472 finite checkpoint tensors. Close final validation metrics are observations, not a statistical equivalence verdict. `quality_accepted=false` and `four_gate_certified=false` remain unchanged. The candidate's first epoch took 416.9 s, versus 200.52 s on average in epochs 2–6; compilation overhead matters.

Machine-readable records: [six_epoch_evidence.json](six_epoch_evidence.json). Exact bounded settings: [six_epoch_recipe.json](six_epoch_recipe.json). The evidence consists of result and audit records, not full checkpoints or raw per-step trajectories.

## What changed

The candidate groups data transfers by dtype, replaces dense decoding-permutation operations with inverse-order comparisons for neighbor visibility, removes activation checkpoint recomputation, and enables dynamic-shape compilation. Compilation settings preserve random behavior and rounding boundaries. Removing recomputation trades memory for compute; these implementation choices are version sensitive.

The native model, loss, optimizer and per-epoch configuration are preserved. Both arms use the same explicit worker and prefetch seeding wrapper, replacing upstream entropy seeding. This alone does not prove exact data-order pairing.

## Reproduce

```bash
git clone https://github.com/micky-li-hd/Faster-ProteinMPNN.git
cd Faster-ProteinMPNN
tar -xzf proteinmpnn-training-runtime.tar.gz
python3.12 -m venv .venv
.venv/bin/python -m pip install -r proteinmpnn-runtime/requirements.txt
```

Download the full public dataset using the [pinned upstream training instructions](https://github.com/dauparas/ProteinMPNN/blob/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/training/README.md). Then run on an assigned, idle GPU:

```bash
PYTHON="$PWD/.venv/bin/python" bash run_six_epoch.sh /data/pdb_2021aug02 /shared/gpu_experiments.lock 0
```

The shared lock must match the lock used by other experiments. The launcher rejects a busy GPU, preserves failures and cleans only its owned processes. Each arm has a 7,200-second resource bound; a timeout is not counted as completion. Results go into a fresh `runs-six-epoch` directory. The script refuses to reuse it.

The archive contains pinned reference/candidate training code, dependency versions, source hashes, the launcher, audit code, startup checks and the original 150-epoch recipe. Its internal README predates this six-epoch experiment; this root README and evidence files describe the latest completed results. No data or weights are redistributed.

## Provenance and limits

- Upstream commit: `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`.
- Candidate model_utils.py SHA256: `df80b1bd880cacfa783f87d52f0f1b2d29808966313133b66fcdcd4386020fe9`.
- Runtime archive SHA256: `a42d7128a3d96de8de1ef19873d3fc665cdcc23db480689cf60b3b5aca22d898`.
- Full data archive observed SHA256: `84d51d0b9224011db8deeab8b83e96f092830aaf6a1f538b1d94b0144f295714` (locally recorded, not an independently published upstream checksum).
- Earlier scoped experiments, numerical diagnostics and inference cases: [protein-infra-cases](https://github.com/micky-li-hd/protein-infra-cases). Its 2.13× steady-state benchmark used a smaller workload and is not pooled with this experiment.
- Earlier launcher pickling failure was fixed and retained in the experiment history. A separate time-limited reference run is excluded from the three-process comparison.
- No full 150-epoch convergence, held-out biological quality equivalence, or ResearchBoost frozen-judge PASS is claimed. No acceptance margin was relaxed.

## Acknowledgements

ProteinMPNN code and derived changes follow its included MIT license. ResearchBoost-Agent and the Faster-* repositories inspired the experiment organization; this repository is an independent contribution, not an official certification or endorsement.
