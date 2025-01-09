import pytest
import numpy as np
from pydantic import ValidationError
from app.models import SparseMatrix

@pytest.fixture
def example_sparse_matrix():
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [10, 20, 30]
    shape = {"nrows": 3, "ncols": 3}
    sparse_matrix = SparseMatrix(rows=rows, cols=cols, vals=vals, shape=shape)
    expected_dense_matrix = np.array([
        [10, 0, 0],
        [0, 20, 0],
        [0, 0, 30]
    ], dtype=np.int8)
    return sparse_matrix, expected_dense_matrix

def test_hash(example_sparse_matrix):
    sparse_matrix, _ = example_sparse_matrix
    sparse_matrix_copy = SparseMatrix(
        rows=sparse_matrix.rows,
        cols=sparse_matrix.cols,
        vals=sparse_matrix.vals,
        shape=sparse_matrix.shape
    )
    assert hash(sparse_matrix) == hash(sparse_matrix_copy)

    sparse_matrix_different = SparseMatrix(
        rows=[0, 1],
        cols=[0, 1],
        vals=[10, 99],
        shape=sparse_matrix.shape
    )
    assert hash(sparse_matrix) != hash(sparse_matrix_different)

def test_to_numpy(example_sparse_matrix):
    sparse_matrix, expected_dense_matrix = example_sparse_matrix
    dense_matrix = sparse_matrix.to_numpy()
    np.testing.assert_array_equal(dense_matrix, expected_dense_matrix)

def test_from_numpy(example_sparse_matrix):
    _, expected_dense_matrix = example_sparse_matrix
    sparse_matrix_from_dense = SparseMatrix.from_numpy(expected_dense_matrix)
    assert sparse_matrix_from_dense.rows == [0, 1, 2]
    assert sparse_matrix_from_dense.cols == [0, 1, 2]
    assert sparse_matrix_from_dense.vals == [10, 20, 30]
    assert sparse_matrix_from_dense.shape.as_tuple() == (3, 3)

def test_empty_matrix():
    empty_matrix = SparseMatrix(rows=[], cols=[], vals=[], shape={"nrows": 3, "ncols": 3})
    dense_matrix = empty_matrix.to_numpy()
    expected_empty = np.zeros((3, 3), dtype=np.int8)
    np.testing.assert_array_equal(dense_matrix, expected_empty)

def test_dtype_selection():
    large_vals = [10, 200, 40000]
    large_sparse_matrix = SparseMatrix(
        rows=[0, 1, 2],
        cols=[0, 1, 2],
        vals=large_vals,
        shape={"nrows": 3, "ncols": 3}
    )
    dense_matrix = large_sparse_matrix.to_numpy()
    assert dense_matrix.dtype == np.int32

def test_invalid_inputs():
    with pytest.raises(ValidationError):
        SparseMatrix(rows=[0], cols=[0], vals=[10])  # Missing shape

    with pytest.raises(ValidationError):
        SparseMatrix(rows=[0, 1], cols=[0], vals=[10, 20], shape=(3, 3))  # Mismatched lengths
