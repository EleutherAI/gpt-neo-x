# pytest_spawned.py

import os
import warnings
import pickle
import multiprocessing
import sys
import pathlib  # Ensure pathlib is imported if used in serialize_report

import pytest
from _pytest import runner
from _pytest.runner import runtestprotocol

# Ensure the multiprocessing uses the 'spawn' method
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method has already been set, likely in a parent process
    pass

def serialize_report(rep):
    d = rep.__dict__.copy()
    if hasattr(rep.longrepr, "toterminal"):
        d["longrepr"] = str(rep.longrepr)
    else:
        d["longrepr"] = rep.longrepr
    for name in d:
        if isinstance(d[name], pathlib.Path):
            d[name] = str(d[name])
        elif name == "result":
            d[name] = None  # for now
    return d

def pytest_addoption(parser):
    group = parser.getgroup("spawned", "spawned subprocess test execution")
    group.addoption(
        "--spawned",
        action="store_true",
        dest="spawned",
        default=False,
        help="box each test run in a separate process using spawn method (cross-platform)",
    )

def pytest_load_initial_conftests(early_config, parser, args):
    early_config.addinivalue_line(
        "markers",
        "spawned: Always spawn for this test.",
    )

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    if item.config.getvalue("spawned") or item.get_closest_marker("spawned"):
        ihook = item.ihook
        ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        reports = spawned_run_report(item)
        for rep in reports:
            ihook.pytest_runtest_logreport(report=rep)
        ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True

def run_test(queue, item_path, item_name):
    import importlib
    import sys

    # Modify sys.path to include the item's directory
    item_dir = os.path.dirname(item_path)
    if item_dir not in sys.path:
        sys.path.insert(0, item_dir)
    try:
        module_name = os.path.splitext(os.path.basename(item_path))[0]
        module = importlib.import_module(module_name)
        test_func = getattr(module, item_name)
        reports = runtestprotocol(test_func, log=False)
        serialized = [serialize_report(rep) for rep in reports]
        queue.put({'status': 'success', 'reports': serialized})
    except KeyboardInterrupt:
        EXITSTATUS_TESTEXIT = 4
        queue.put({'status': 'exit', 'exitstatus': EXITSTATUS_TESTEXIT})
    except Exception as e:
        queue.put({'status': 'crash', 'error': str(e)})

def spawned_run_report(item):
    EXITSTATUS_TESTEXIT = 4

    # Extract the test item's file path and name
    item_path = item.fspath.strpath
    item_name = item.name

    # Create a multiprocessing Queue to get results
    queue = multiprocessing.Queue()

    # Start the child process
    p = multiprocessing.Process(target=run_test, args=(queue, item_path, item_name))
    p.start()
    p.join()

    if p.exitcode == 0:
        if not queue.empty():
            result = queue.get()
            if result['status'] == 'success':
                report_dumps = result['reports']
                return [runner.TestReport(**x) for x in report_dumps]
            elif result['status'] == 'exit':
                pytest.exit(f"spawned test item {item} raised Exit")
            else:
                return [report_process_crash(item, result)]
        else:
            return [report_process_crash(item, {'status': 'crash', 'error': 'No data returned'})]
    else:
        # Handle unexpected exit
        result = {'status': 'crash', 'error': f"Process exited with code {p.exitcode}"}
        return [report_process_crash(item, result)]

def report_process_crash(item, result):
    from _pytest._code import getfslineno

    path, lineno = getfslineno(item)
    if result['status'] == 'crash':
        info = f"{path}:{lineno}: running the test CRASHED with error: {result.get('error', 'Unknown Error')}"
    elif result['status'] == 'exit':
        info = f"{path}:{lineno}: running the test raised Exit with status {result.get('exitstatus')}"
    else:
        info = f"{path}:{lineno}: running the test encountered an unknown issue."

    from _pytest import runner

    # pytest >= 4.1
    has_from_call = hasattr(runner.CallInfo, "from_call")
    if has_from_call:
        call = runner.CallInfo.from_call(lambda: 0 / 0, "???")
    else:
        call = runner.CallInfo(lambda: 0 / 0, "???")
    call.excinfo = info
    rep = runner.pytest_runtest_makereport(item, call)
    if 'out' in result and result['out']:
        rep.sections.append(("captured stdout", result['out']))
    if 'err' in result and result['err']:
        rep.sections.append(("captured stderr", result['err']))

    xfail_marker = item.get_closest_marker("xfail")
    if not xfail_marker:
        return rep

    rep.outcome = "skipped"
    rep.wasxfail = (
        f"reason: {xfail_marker.kwargs.get('reason', 'No reason provided')}; "
        f"pytest-spawned reason: {info}"
    )
    warnings.warn(
        "pytest-spawned xfail support is incomplete at the moment and may "
        "output a misleading reason message",
        RuntimeWarning,
    )
    return rep

