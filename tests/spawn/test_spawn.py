# tests/test_pid.py

import os
import pytest

def test_normal():
    """A normal test running in the main process."""
    print(f"Normal Test PID: {os.getpid()}")
    assert True

@pytest.mark.spawned
def test_spawned_1():
    """A spawned test running in a separate process."""
    print(f"Spawned Test 1 PID: {os.getpid()}")
    assert True

@pytest.mark.spawned
def test_spawned_2():
    """Another spawned test running in a separate process."""
    print(f"Spawned Test 2 PID: {os.getpid()}")
    assert True
