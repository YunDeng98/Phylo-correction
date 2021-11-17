import hashlib
import os
import sys
import pickle
import time
from functools import wraps
from inspect import signature
from typing import List, Optional, Tuple

import logging


def init_logger():
    logger = logging.getLogger('caching')
    logger.setLevel(logging.DEBUG)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


init_logger()
logger = logging.getLogger("caching")

_CACHE_DIR = None
_USE_CACHED = True
_TEST_MODE = False
_READ_ONLY = False
_HASH = True


def set_cache_dir(cache_dir: str):
    logger.info(f"Setting cache directory to: {cache_dir}")
    global _CACHE_DIR
    _CACHE_DIR = cache_dir


def get_cache_dir():
    global _CACHE_DIR
    if _CACHE_DIR is None:
        raise Exception(f"Cache directory has not been set yet. Please set it with set_cache_dir function.")
    return _CACHE_DIR


def set_log_level(log_level: int):
    logger = logging.getLogger("caching")
    logger.setLevel(level=log_level)


def set_use_cached(use_cached: bool):
    global _USE_CACHED
    _USE_CACHED = use_cached


def get_use_cached():
    global _USE_CACHED
    return _USE_CACHED


def set_test_mode(test_mode: bool):
    global _TEST_MODE
    _TEST_MODE = test_mode


def get_test_mode():
    global _TEST_MODE
    return _TEST_MODE


def set_read_only(read_only: bool):
    global _READ_ONLY
    _READ_ONLY = read_only


def get_read_only():
    global _READ_ONLY
    return _READ_ONLY


def set_hash(hash: bool):
    global _HASH
    _HASH = hash


def get_hash():
    global _HASH
    return _HASH


class CacheUsageError(Exception):
    pass


def hash_all(xs: List[str]) -> str:
    hashes = [hashlib.sha512(x.encode("utf-8")).hexdigest() for x in xs]
    return hashlib.sha512("".join(hashes).encode("utf-8")).hexdigest()


def cached(
    cache_keys: Optional[Tuple]=None,
    exclude_args: Optional[Tuple]=None,
):
    """
    In read_only mode, cached results are returned, but new results are not
    written to the cache. This avoids race conditions when running several
    notebooks at the same time.

    Only one of cache_keys and exclude_args can be provided.

    Args:
        cache_keys: What arguments to use for the hash key.
        exclude_args: What arguments to exclude from the hash key. E.g.
            n_processes, which does not affect the result of the function.
            If None, nothing will be excluded.
    """
    def f_res(func):
        @wraps(func)
        def r_res_res(*args, **kwargs):
            # Get caching hyperparameters
            cache_dir = get_cache_dir()
            use_cached = get_use_cached()
            test_mode = get_test_mode()
            read_only = get_read_only()
            hash = get_hash()

            s = signature(func)
            binding = s.bind(*args, **kwargs)
            binding.apply_defaults()
            # Check that all cache_keys are present - it might be a typo!
            if cache_keys is not None:
                for cache_key in cache_keys:
                    if cache_key not in binding.arguments:
                        raise CacheUsageError(
                            f"{cache_key} is not an argument to {func.__name__}. Fix the cache_keys."
                        )
            # Check that all exclude_args are present - it might be a typo!
            if exclude_args is not None:
                for arg in exclude_args:
                    if arg not in binding.arguments:
                        raise CacheUsageError(
                            f"{arg} is not an argument to {func.__name__}. Fix the exclude_args."
                        )
            # Only one of cache_keys and exclude_args can be provided
            if cache_keys is not None and exclude_args is not None:
                raise CacheUsageError(
                    f"Only one of cache_keys and exclude_args can be provided"
                )
            if not hash:
                path = (
                    [cache_dir]
                    + [f"{func.__name__}"]
                    + [
                        f"{key}_{val}"
                        for (key, val) in binding.arguments.items()
                        if ((cache_keys is None) or (key in cache_keys))
                        and ((exclude_args is None) or (key not in exclude_args))
                    ]
                    + ["result"]
                )
            else:
                path = (
                    [cache_dir]
                    + [f"{func.__name__}"]
                    + [
                        hash_all(
                            [
                                f"{key}_{val}"
                                for (key, val) in binding.arguments.items()
                                if ((cache_keys is None) or (key in cache_keys))
                                and ((exclude_args is None) or (key not in exclude_args))
                            ]
                        )
                    ]
                    + ["result"]
                )
            success_token_filename = os.path.join(*path) + ".success"
            filename = os.path.join(*path) + ".pickle"
            if test_mode:
                logger.info(f"Would pickle to: {filename}")
                return func(*args, **kwargs)
            if os.path.isfile(success_token_filename) and use_cached:
                if not os.path.isfile(filename):
                    raise Exception(f"Success token is present but file is missing!: {filename}")
                with open(filename, "rb") as f:
                    try:
                        return pickle.load(f)
                    except:
                        raise Exception(f"Corrupt cache file due to unpickling error, even though success token was present!: {filename}")
            else:
                if os.path.isfile(filename) and use_cached:
                    assert(not os.path.isfile(success_token_filename))  # Should be true because we are in the else statement
                    logger.info("Success token missing but pickle file is present. "
                                "Thus pickle file is most likely corrupt. "
                                f"Will have to recompute: {filename}")
                if read_only:
                    logger.info(f"In read_only mode. Will not write to cache. Would have written to: {filename}")
                    return func(*args, **kwargs)
                # logger.info(f"Calling {func.__name__}")
                st = time.time()
                res = func(*args, **kwargs)
                # logger.info(f"Time taken {func.__name__}: {time.time() - st}")
                os.makedirs(os.path.join(*path[:-1]), exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(res, f)
                    f.flush()
                with open(success_token_filename, "w") as f:
                    f.write("SUCCESS\n")
                    f.flush()
                return res

        return r_res_res

    return f_res
