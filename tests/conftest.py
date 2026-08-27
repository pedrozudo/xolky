import jax
import pytest

from xolky import _xolky


jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(config, items):
    if any(device.platform == "gpu" for device in jax.devices()):
        return
    marker = pytest.mark.skip(reason="xolky requires a CUDA device")
    for item in items:
        item.add_marker(marker)


@pytest.fixture(autouse=True)
def no_leaked_solvers():
    before = _xolky.active_solver_count()
    yield
    _xolky.shutdown()
    assert _xolky.active_solver_count() == 0
    assert before == 0
