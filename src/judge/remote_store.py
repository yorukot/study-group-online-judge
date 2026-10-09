"""Durable remote records and locks that survive workspace retention."""

import json
import os
import sys
from contextlib import contextmanager
from fcntl import LOCK_EX, flock
from pathlib import Path


@contextmanager
def reporting_output(path: Path):
    """Keep SDK output, including cached streams and child processes, off SSH."""
    streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    with path.open("a") as log:
        descriptors = {1, 2}
        for stream in streams:
            if stream is not None:
                stream.flush()
                try:
                    descriptors.add(stream.fileno())
                except OSError, ValueError:
                    pass
        originals = {descriptor: os.dup(descriptor) for descriptor in descriptors}
        try:
            # W&B patches stream.write on import. Replacing sys.stdout/stderr
            # bypasses those hooks, leaving metrics uploaded but no console logs.
            for descriptor in originals:
                os.dup2(log.fileno(), descriptor)
            yield
        finally:
            try:
                for stream in streams:
                    if stream is not None:
                        stream.flush()
                log.flush()
            finally:
                for descriptor, original in originals.items():
                    os.dup2(original, descriptor)
                    os.close(original)


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def job_lock(workspace: Path):
    locks = workspace.parent.parent / "job-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{workspace.name}.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        yield
