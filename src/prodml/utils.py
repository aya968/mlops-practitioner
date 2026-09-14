import functools
import logging
import time
from typing import Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prodml")


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.2f} ms"
        )
        return result

    return wrapper