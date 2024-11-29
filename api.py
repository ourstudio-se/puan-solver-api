import uvicorn
import logging

from loguru import logger
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from app.models import (
    SolveILPResponse,
    Problem,
)
from app.settings import EnvironmentVariables
from app.tasks import solve_linear_ilp_dispatched
from celery.exceptions import TimeoutError

# Load environment variables
env = EnvironmentVariables()

# Configure logging
logging.disable(getattr(logging, env.LOG_LEVEL, logging.INFO))

app = FastAPI(
    title="Integer Linear Programming (ILP) API",
    description="API to handle models and solve integer linear programming problems.",
    version=env.VERSION
)
    
@app.get(
    "/health",
    summary="Health check endpoint",
    description="Returns a simple health check response.",
)
async def health_check() -> dict:
    return {"status": "ok"}

@app.get(
    "/ready",
    summary="Readiness check endpoint",
    description="Returns a simple readiness check response.",
)
async def readiness_check() -> dict:
    return {"status": "ready"}

@app.post(
    "/model/solve-one/linear",
    response_model=SolveILPResponse,
    summary="Solves an ILP problem using a specific given model",
    description="""
    
    Solve an Integer Linear Programming (ILP) problem directly.

    Example
    -------
    
    {
        "model": {
            "polyhedron": {
                "A": {
                    "rows": [0,0,1,1,2,2],
                    "cols": [0,1,2,3,4,5],
                    "vals": [1,1,1,1,1,1],
                    "shape": {"nrows": 4, "ncols": 6}
                },
                "b": [1,1,1,0]
            },
            "columns": ["a", "b", "c", "x", "y", "z"],
            "rows": ["A", "B", "C", "D"],
            "intvars": []
        },
        "direction": "maximize",
        "objectives": [
            {"a": 1},
            {"b": 1},
            {"y": 1}
        ]
    }

    Should return:
    {
        "solutions": [
            {
                "solution": {
                    "a": 1,
                    "b": 0,
                    "c": 1,
                    "x": 0,
                    "y": 1,
                    "z": 0
                },
                "error": null
            },
            {
                "solution": {
                    "a": 1,
                    "b": 1,
                    "c": 1,
                    "x": 0,
                    "y": 1,
                    "z": 0
                },
                "error": null
            },
            {
                "solution": {
                    "a": 1,
                    "b": 0,
                    "c": 1,
                    "x": 0,
                    "y": 1,
                    "z": 0
                },
                "error": null
            }
        ]
    }
    """
)
async def solve_single_ilp(problem: Problem = Body(...)) -> SolveILPResponse:
    try:
        results = await solve_linear_ilp_dispatched(
            problem.model, 
            problem.direction == "minimize", 
            problem.objectives,
        )
        return SolveILPResponse(
            solutions=results,
        )
    except ValueError as ve:
        logger.error(f"Input error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except TimeoutError:
        logger.error("Timeout")
        raise HTTPException(status_code=504, detail="The computation timed out.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    
# Custom error handlers
@app.exception_handler(HTTPException)
def http_exception_handler(_, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}},
    )

@app.exception_handler(Exception)
def general_exception_handler(_, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": "Internal Server Error"}},
    )

# Log all environment settings on startup
logger.info("Environment settings:")
logger.info(f"Starting {env.APP_NAME} version {env.VERSION}")
logger.info(f"Running on port {env.PORT}")
logger.info(f"Running tasks locally: {env.RUN_TASKS_LOCALLY}")
logger.info(f"Celery broker URL: {env.CELERY_BROKER_URL} (only used if not running tasks locally)")
logger.info(f"Log level set to {env.LOG_LEVEL}")
logger.info(f"Default computation timeout: {env.DEFAULT_COMPUTATION_TIMEOUT} seconds")
logger.info(f"Worker batch size: {env.WORKER_BATCH_SIZE}")
logger.info(f"Objective batch size: {env.OBJECTIVE_BATCH_SIZE}")
logger.info(f"Cache Redis URL: {env.CACHE_REDIS_URL}")

class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1 and record.getMessage().find("/ready") == -1

# Filter out healthcheck and readiness check logs
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=env.PORT)