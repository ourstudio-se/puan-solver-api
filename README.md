# Integer Linear Programming (ILP) API

## Overview

The Integer Linear Programming (ILP) API is a Python-based service designed to provide efficient solutions to integer linear programming problems. The API is divided into two main components:

1. **Polyhedron Handling**: Allows users to construct and manage polyhedrons by sending relevant data to the API. The API stores these polyhedrons and returns a unique key for future reference.
2. **Solver**: Enables users to solve ILP problems using previously stored polyhedrons by referencing their unique keys.

This architecture facilitates **easy parallelization during heavy solver loads** and minimizes HTTP/JSON traffic between services.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
  - [Polyhedron Handling](#polyhedron-handling)
    - [Create Polyhedron](#create-polyhedron)
    - [Retrieve Polyhedron](#retrieve-polyhedron)
  - [Solver](#solver)
    - [Solve ILP Problems](#solve-ilp-problems)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the Repository**
    ```bash
    git clone ...
    ```
2. **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install dependencies**
    ```bash
    pip install poetry
    poetry install
    ```
3. **Set environment variables**
    These variables needs to be set (and is predefined here with common configuration):
    ```
    CELERY_BROKER_URL="redis://localhost:6379/0"
    DEFAULT_COMPUTATION_TIMEOUT=60
    RUN_TASKS_LOCALLY=true
    PORT=9000
    LOG_LEVEL="WARNING"
    OBJECTIVE_BATCH_SIZE=400
    WORKER_BATCH_SIZE=10
    ```
    Checkout Python Celery for which message brokers are offered.

If you don't know what you're doing, then set the broker and result url to the same connection.
- **RUN_TASKS_LOCALLY**: Means no workers are necessary, and api machine itself will solve the tasks. *No need to care about if you're running through docker.*
- **DEFAULT_COMPUTATION_TIMEOUT**: If no workers (and RUN_TASKS_LOCALLY is false), then the api will timeout request after this set time (in seconds).
- **OBJECTIVE_BATCH_SIZE**: Size of batch of all objectives.
- **WORKER_BATCH_SIZE**: Size of batch to each worker.

## Getting Started
### Local dev
We need to do two things:
1. Start a worker process:
    ```bash
    celery -A app.tasks worker --loglevel=warning -n w0
    ```
    This will start a worker "w0" which will listen to incoming problems to solve. You can start as many workers as you like.
2. Start the API:
    ```bash
    poetry run fastapi run api.py
    ````
3. (Optional) Start the `flower` to monitor traffic
    ```bash
    celery -A app.tasks flower
    ```

### Docker
Make sure you either have an `.env` file with proper variable's set, and run
```bash
docker run -it -p 8000:8000 $(docker build -q .) fastapi run api.py
```

## API Endpoints
Go to `http://localhost:8000/docs` for documentation.

## License
Apache License 2.0