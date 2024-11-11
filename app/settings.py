from pydantic_settings import BaseSettings
from pydantic import ConfigDict  # Import ConfigDict in Pydantic V2
from typing import Optional

class EnvironmentVariables(BaseSettings):

    # Optional general settings
    VERSION:    str                     = "0.1.0"
    PORT:       int                     = 8000
    LOG_LEVEL:  str                     = "WARNING"
    APP_NAME:   str                     = "ILP_API"

    # Optional computation settings 
    CELERY_BROKER_URL: Optional[str]    = None
    DEFAULT_COMPUTATION_TIMEOUT: int    = 5 # seconds
    RUN_TASKS_LOCALLY: bool             = True
    WORKER_BATCH_SIZE: int              = 10
    OBJECTIVE_BATCH_SIZE: int           = 200
    RESULT_EXPIRATION: int              = 3600 # seconds
    
    # Optional settings for Redis
    CELERY_RESULT_BACKEND: Optional[str]    = None
    CACHE_REDIS_URL: Optional[str]          = None
    
    # Use ConfigDict for Pydantic V2
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
