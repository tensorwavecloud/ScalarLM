import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

@asynccontextmanager
async def acquire_file_lock(file_path, timeout=30, poll_interval=0.1):
    """
    Async context manager for acquiring a file lock.

    Args:
        file_path: Path to the file to lock
        timeout: Maximum time to wait for lock acquisition (seconds)
        poll_interval: Time between lock acquisition attempts (seconds)

    Usage:
        async with acquire_file_lock(response_path):
            # Your code here - file is locked
            pass
        # Lock is automatically released
    """
    file_path = Path(file_path)
    lock_path = file_path.parent / f"{file_path.name}.lock"

    # Try to acquire the lock
    start_time = asyncio.get_event_loop().time()
    while True:
        try:
            # Create lock file exclusively (fails if exists)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            # Lock file exists, check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds")
            await asyncio.sleep(poll_interval)

    try:
        yield  # Lock acquired, execute the context block
    finally:
        # Always release the lock
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass  # Lock already removed
