
import sys
import numpy
import joblib.numpy_pickle

# Provide compatibility alias for standard module lookup
if 'numpy._core' not in sys.modules:
    import numpy.core as core_module
    sys.modules['numpy._core'] = core_module
    sys.modules['numpy._core.multiarray'] = getattr(core_module, 'multiarray', core_module)

# Intercept joblib NumpyUnpickler to redirect numpy._core -> numpy.core during model loading
if hasattr(joblib.numpy_pickle, 'NumpyUnpickler'):
    _orig_find_class = joblib.numpy_pickle.NumpyUnpickler.find_class
    def _patched_find_class(self, module, name):
        if module and module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return _orig_find_class(self, module, name)
    joblib.numpy_pickle.NumpyUnpickler.find_class = _patched_find_class

import warnings
import pytest

def pytest_configure(config):
    # Suppress scikit-learn version unpickling warnings for legacy model bundles
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
    # Suppress pandas upcoming pyarrow requirement warning
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
