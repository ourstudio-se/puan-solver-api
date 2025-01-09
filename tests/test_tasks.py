import pytest
import numpy as np
from app.models import Model, Polyhedron, SparseMatrix
from app.tasks import solve_linear_ilp_dispatched  # Replace with your actual module path

@pytest.fixture
def example_model():
    # Create a SparseMatrix for A
    sparse_matrix_A = SparseMatrix.from_numpy(
        np.array([
            [1, 0],
            [-1, 0]
        ], dtype=np.int8)
    )

    # Create a Polyhedron instance
    polyhedron = Polyhedron(
        A=sparse_matrix_A,
        b=[0, -10]
    )

    # Create a Model instance
    return Model(
        id="test_model_1",
        polyhedron=polyhedron,
        columns=["x1", "x2"],
        rows=["c1", "c2", "c3", "c4"],
        intvars=["x1"]
    )

@pytest.mark.asyncio
async def test_solve_linear_ilp_dispatched_parallel(example_model):
    objectives = [{"x1": 1, "x2": 2}, {"x1": -1, "x2": 1}]
    minimize = True
    solutions = await solve_linear_ilp_dispatched(example_model, minimize, objectives)

    assert len(solutions) == len(objectives)
    assert solutions[0].solution == {"x1": 0, "x2": 0}
    assert solutions[1].solution == {"x1": 10, "x2": 0}
    
    objectives = [{"x1": 1, "x2": 2}, {"x1": -1, "x2": 1}]
    minimize = False
    solutions = await solve_linear_ilp_dispatched(example_model, minimize, objectives)

    assert len(solutions) == len(objectives)
    assert solutions[0].solution == {"x1": 10, "x2": 1}
    assert solutions[1].solution == {"x1": 0, "x2": 1}

@pytest.mark.asyncio
async def test_solve_linear_ilp_dispatched_sequential(example_model):
    # Single objective to test sequential solving
    objectives = [{"x1": 3, "x2": 5}]
    minimize = False
    solutions = await solve_linear_ilp_dispatched(example_model, minimize, objectives)

    # Assertions
    assert len(solutions) == 1
    assert solutions[0].solution["x2"] == 1

@pytest.mark.asyncio
async def test_solve_linear_ilp_dispatched_no_objectives(example_model):
    # Test with empty objectives
    objectives = []
    minimize = True

    solutions = await solve_linear_ilp_dispatched(example_model, minimize, objectives)

    # Assertions
    assert solutions == []  # No objectives means no solutions
