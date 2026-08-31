import jax
import pytest

from xolky.wrapper import _xolky_cholmod, _xolky_cuda


jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(config, items):
    cuda_available = _xolky_cuda is not None and any(
        device.platform == "gpu" for device in jax.devices()
    )
    if cuda_available:
        return
    marker = pytest.mark.skip(reason="xolky requires a CUDA device")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(marker)


def _native_modules():
    return tuple(
        module for module in (_xolky_cuda, _xolky_cholmod) if module is not None
    )


@pytest.fixture(autouse=True)
def no_leaked_solvers():
    modules = _native_modules()
    before = sum(module.active_solver_count() for module in modules)
    yield
    for module in modules:
        module.shutdown()
    assert sum(module.active_solver_count() for module in modules) == 0
    assert before == 0
