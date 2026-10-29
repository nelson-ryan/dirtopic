import time
from datetime import timedelta
from ..config import logging
logger = logging.getLogger(__name__)

def timeit(func, *args, **kwargs):
    def wrapitup(*args, **kwargs):
        name = func.__qualname__
        logger.info(f"Starting {name}")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(
            f"Completed {name} in {timedelta(seconds = end_time - start_time)}"
        )
        return result
    return wrapitup
