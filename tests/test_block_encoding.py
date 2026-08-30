import numpy as np
import pytest

import qfin


def test_covariance_is_a_mathematical_candidate_not_an_implementation() -> None:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    report = qfin.analyze_block_encoding(covariance)

    assert report.square
    assert report.hermitian
    assert report.positive_semidefinite
    assert report.mathematical_qsvt_candidate
    assert not report.qfin_block_encoding_implemented
    assert not report.qfin_qsvt_implemented
    assert report.padded_dimension == 2
    assert report.data_qubits == 1
    assert "not an efficient oracle" in str(report.to_dict()["caveat"])


def test_nonhermitian_matrix_reports_why_it_is_not_a_candidate() -> None:
    report = qfin.analyze_block_encoding([[1.0, 2.0], [0.0, 1.0]])
    assert not report.hermitian
    assert not report.mathematical_qsvt_candidate
    assert any("Hermitian" in reason for reason in report.implementation_reasons)


@pytest.mark.parametrize(
    "matrix, message",
    [
        ([], "non-empty"),
        ([[1.0, np.nan]], "finite"),
    ],
)
def test_block_encoding_analysis_rejects_invalid_matrices(
    matrix: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.analyze_block_encoding(matrix)  # type: ignore[arg-type]


def test_explicit_analysis_has_a_dimension_guard() -> None:
    with pytest.raises(ValueError, match="structural oracle metadata"):
        qfin.analyze_block_encoding(np.eye(5), max_dimension=4)
