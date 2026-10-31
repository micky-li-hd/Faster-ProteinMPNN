"""Common seed controls around otherwise unchanged upstream training."""
import concurrent.futures
import json
import os
from pathlib import Path
import random
import runpy
import sys

OriginalExecutor = concurrent.futures.ProcessPoolExecutor


def seeded_call(seed, function, args, kwargs):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    return function(*args, **kwargs)


def seeded_worker(worker_id):
    import numpy as np
    import torch
    np.random.seed(torch.initial_seed() % (2**32))


class SeededExecutor(OriginalExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_index = 0

    def submit(self, fn, /, *args, **kwargs):
        seed = int(os.environ['MPNN_RUN_SEED']) + 1000 + self.call_index
        self.call_index += 1
        return super().submit(seeded_call, seed, fn, args, kwargs)


def main():
    source, recipe_path, data, output, checkpoint = sys.argv[1:6]
    import numpy as np
    import torch
    if torch.__version__ != '2.5.1+cu121' or np.__version__ != '1.26.4':
        raise RuntimeError('Use the pinned torch 2.5.1+cu121 / numpy 1.26.4 environment.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback would invalidate this experiment.')
    recipe = json.loads(Path(recipe_path).read_text())
    seed = recipe['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sys.path.insert(0, source)
    import utils
    utils.worker_init_fn = seeded_worker
    concurrent.futures.ProcessPoolExecutor = SeededExecutor
    environment = {'python':sys.version, 'torch':torch.__version__, 'numpy':np.__version__,
                   'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0),
                   'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                   'tf32_cudnn':torch.backends.cudnn.allow_tf32}
    Path(output, 'environment.json').write_text(json.dumps(environment, indent=2))
    sys.argv = [str(Path(source, 'training.py')), '--path_for_training_data', data,
                '--path_for_outputs', output]
    if checkpoint:
        sys.argv.extend(['--previous_checkpoint', checkpoint])
    for key, value in recipe['training'].items():
        sys.argv.extend(['--'+key, str(value)])
    runpy.run_path(str(Path(source, 'training.py')), run_name='__main__')


if __name__ == '__main__':
    # runpy temporarily replaces __main__; pool callbacks must live in a
    # stable importable module so multiprocessing can pickle them.
    from seeded_entry import main as imported_main
    imported_main()
