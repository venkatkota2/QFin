"""Run engine-independent contracts against every installed finance engine."""

import pytest

from qfin import _native


@pytest.fixture(params=["numpy", "native"])
def installed_engine(request: pytest.FixtureRequest) -> str:
    engine = str(request.param)
    if engine == "native" and not _native.available():
        pytest.skip("native extension unavailable in this installation")
    return engine
