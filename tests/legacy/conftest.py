"""Make this directory's modules win over the ones beside it.

These tests carry their own `quant.py`, `lib.py` and `ref.py`, because the current kernels and
the legacy ones describe KV caches differently and sharing one set of helpers would mean a
module that is half true for each. pytest already puts this directory first on `sys.path` (it
has no `__init__.py`), so `import quant` here is this directory's, not `tests/`.

`kernelkit` is the one thing that is shared, unchanged: it is a benchmarking and comparison
toolbox with nothing in it that depends on a KV cache layout.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TESTS not in sys.path:
    sys.path.append(_TESTS)
