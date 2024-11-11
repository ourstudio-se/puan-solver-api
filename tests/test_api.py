from fastapi.testclient import TestClient
from api import app  # Replace with the actual import path of your FastAPI app

client = TestClient(app)

def test_solve_single_ilp_docstring_example():
    # Example payload for the request
    payload = {
        "model": {
            "polyhedron": {
                "A": {
                    "rows": [0, 0, 1, 1, 2, 2],
                    "cols": [0, 1, 2, 3, 4, 5],
                    "vals": [1, 1, 1, 1, 1, 1],
                    "shape": {"nrows": 4, "ncols": 6}
                },
                "b": [1, 1, 1, 0]
            },
            "columns": ["a", "b", "c", "x", "y", "z"],
            "rows": ["A", "B", "C", "D"],
            "intvars": []
        },
        "direction": "maximize",
        "objectives": [
            {"a": 1, "b": -1, "c": 1, "x": -1, "y": 1, "z": -1},
        ]
    }

    # Make the POST request to the endpoint
    response = client.post("/model/solve-one/linear", json=payload)

    # Assert the status code
    assert response.status_code == 200

    # Parse the response JSON
    response_data = response.json()

    # Expected response structure validation
    assert "solutions" in response_data
    assert isinstance(response_data["solutions"], list)

    for solution in response_data["solutions"]:
        assert "solution" in solution
        assert "error" in solution
        assert isinstance(solution["solution"], dict)
        assert solution["error"] is None  # Expected error to be None in example

    # Further validation for expected solution values if needed
    expected_solutions = [
        {"a": 1, "b": 0, "c": 1, "x": 0, "y": 1, "z": 0},
    ]
    for i, solution in enumerate(response_data["solutions"]):
        assert solution["solution"] == expected_solutions[i]
