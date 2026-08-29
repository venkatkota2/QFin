import numpy as np
import pytest

import qfin


def test_mortality_scalar_vector_and_survival() -> None:
    table = qfin.MortalityTable([40, 41, 42], [0.01, 0.02, 0.04], category="A")
    assert table.qx(40) == pytest.approx(0.01)
    assert table.px(41) == pytest.approx(0.98)
    np.testing.assert_allclose(table.qx([39, 40.5, 45]), [0.01, 0.015, 0.04])
    assert table.survival_probability(40, 2) == pytest.approx(0.99 * 0.98)
    assert table.survival_probability(40, 0) == 1.0


def test_extreme_mortality_and_invalid_tables() -> None:
    zero = qfin.MortalityTable([0, 120], [0, 0])
    certain = qfin.MortalityTable([0, 120], [1, 1])
    assert zero.survival_probability(50, 20) == 1.0
    assert certain.survival_probability(50, 1) == 0.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        qfin.MortalityTable([40, 41], [0.01, 1.01])
    with pytest.raises(ValueError, match="integer"):
        zero.survival_probability(50, 1.5)  # type: ignore[arg-type]
