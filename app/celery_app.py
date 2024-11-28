# celery_app.py
from celery import Celery
from app.settings import EnvironmentVariables

env = EnvironmentVariables()

# Configure Celery to use Redis as the broker
celery = Celery(
    env.APP_NAME,
    broker=env.CELERY_BROKER_URL,
    backend=env.CELERY_BROKER_URL if env.CELERY_RESULT_BACKEND is None else env.CELERY_RESULT_BACKEND,
)

# Optional: Load configuration from a separate module or environment variables
celery.conf.update(
    result_expires=env.RESULT_EXPIRATION,
    worker_redirect_stdouts=False,
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json'],  # Allow tasks to accept both JSON and Pickle

    # Run tasks synchronously, no message queue
    task_always_eager=env.RUN_TASKS_LOCALLY,  
    task_eager_propagates=env.RUN_TASKS_LOCALLY,
)