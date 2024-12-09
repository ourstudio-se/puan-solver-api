import npycvx
import numpy as np
import functools

from app.celery_app import celery
from app.models import Model, ILPSolution
from app.settings import EnvironmentVariables

from celery.utils.log import get_task_logger
from gzip import compress, decompress
from itertools import chain, batched
from pickle import dumps, loads, HIGHEST_PROTOCOL
from typing import Dict, Tuple, List

env = EnvironmentVariables()
logger = get_task_logger(env.APP_NAME)

def prepare_npycvx(model: Model) -> tuple:

    """
        Prepare the model for npycvx solver.
        NOTE This function is cached to avoid recomputing the same result.

        Args:
            model: The model to be solved

        Returns:
            npycvx compatible model
    """
    
    # First convert the model to numpy format
    A_np = model.polyhedron.A.to_numpy()
    b_np = np.array(model.polyhedron.b)

    # If no columns are present, we return None (since we cannot solve the problem)
    if A_np.shape[1] == 0:
        return None
    
    # To make sure that the matrix is in the correct format,
    # we need to include a dummy zero row if no rows are present 
    if A_np.shape[0] == 0:
        A_np = np.append(A_np, np.zeros((1, A_np.shape[1]), dtype=np.int64), axis=0)
        b_np = np.append(b_np, 0)
    
    result = npycvx.convert_numpy(
        aub=A_np, 
        bub=b_np, 
        int_vrs=set(map(model.columns.index, model.intvars)) if model.intvars else set()
    )

    # Free up memory
    del A_np, b_np
    return result

def build_objectives_matrix(objectives: List[Dict[str, int]], model: Model) -> np.ndarray:

    """
        Build the objectives matrix for npycvx solver.

        Args:
            objectives: List of objectives to be optimized
            model: The model to be solved

        Returns:
            np.ndarray: The objectives matrix
    """

    mx = max(chain(*map(lambda objective: map(abs, objective.values()), objectives)))
    obj = np.zeros((len(objectives), len(model.columns)), dtype=np.int8 if mx == 1 else (np.int16 if mx < 256 else np.int32))
    for i, objective in enumerate(objectives):
        obj[i, [model.columns.index(k) for k in objective if k in model.columns]] = list(dict(filter(lambda kv: kv[0] in model.columns, objective.items())).values())
    return obj

def map_solution(columns: List[str], status: str, solution: np.ndarray) -> ILPSolution:

    """
        Map the solution to the ILPSolution object.

        Args:
            columns: List of column/variable names
            status: The status of the solution
            solution: The solution to the ILP
        
        Returns:
            ILPSolution: The mapped solution
    """

    if status == "optimal":
        return ILPSolution(
            solution=dict(zip(columns, solution.astype(int).tolist())), 
            error=None
        )
    else:
        return ILPSolution(solution=None, error=status)
    
@celery.task(
    serializer='pickle',
    autoretry_for=(Exception,),             # Retry on any exception
    retry_kwargs={'max_retries': 3},        # Set maximum retry attempts
    retry_backoff=True,                     # Use exponential backoff between retries
    time_limit=3,                           # Forcefully kill the task after this limit
    soft_time_limit=1                       # Gracefully allow the task to handle timeout
)
def solve_objectives(data: bytes) -> List[Tuple[str, np.ndarray]]:

    """
        Celery (wrapper) task to solve single problem (multiple objectives).

        Args:
            data: The compressed data to be unpickled

        Returns:
            List[Tuple[str, np.ndarray]]: The status and solution to the ILP
    """

    # Unpickle the data
    nxargs, minimize, objectives = loads(decompress(data))
    
    # The solver function to solve a single objective
    result = [npycvx.solve_lp(*nxargs, minimize, objective) for objective in objectives]

    # Delete nxargs to free up memory
    del nxargs

    return result

async def solve_linear_ilp_dispatched(model: Model, minimize: bool, objectives: List[Dict[str, int]]) -> List[ILPSolution]:
    """
        Celery task to either solve ILP dispatched in parallel or sequentially.

        Args:
            model: The model to be solved
            minimize: Whether to minimize or maximize the objectives
            objectives: The objectives to be optimized

        Returns:
            List[ILPSolution]: The solutions to the objectives
    """

    # Convert objectives to matrix
    objectives_mat = build_objectives_matrix(objectives, model)

    # There's a special case when the model has no columns
    # In this case, we return an empty solution
    npycvx_prep = prepare_npycvx(model)
    if npycvx_prep is None:
        return list(map(lambda _: ILPSolution(solution={}, error=None), objectives_mat))
    
    # Split the objectives into batches
    objectives_batches = batched(objectives_mat, env.OBJECTIVE_BATCH_SIZE)

    # Create a group of subtasks to run in parallel
    jobs = solve_objectives.chunks(
        (
            (compress(dumps((npycvx_prep, minimize, objectives), protocol=HIGHEST_PROTOCOL)), )
            for objectives in objectives_batches
        ),
        env.WORKER_BATCH_SIZE
    )

    # Release objectives matrix from memory
    del objectives_mat

    # Run all tasks in parallel and return the results
    results = jobs.apply_async(timeout=env.DEFAULT_COMPUTATION_TIMEOUT)

    # Wait for all tasks to finish
    while results.waiting():
        continue

    # Run all tasks in parallel and return the results
    return list(
        map(
            lambda status_solution: map_solution(
                model.columns,
                *status_solution
            ),
            chain(
                *chain(
                    *map(
                        lambda r: r.result, 
                        results
                    )
                )
            )
        )
    )
