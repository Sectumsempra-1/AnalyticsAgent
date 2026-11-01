import os
import time
from pathlib import Path

import pytest

from analytics_agent import sandbox
from analytics_agent.sandbox import run_code


def _process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_captures_stdout(tmp_path: Path) -> None:
    result = run_code("print(6 * 7)", tmp_path)
    assert result.stdout.strip() == "42"
    assert result.exit_code == 0
    assert not result.timed_out and not result.truncated


def test_exception_reports_traceback(tmp_path: Path) -> None:
    result = run_code("x = 1\n1 / 0", tmp_path)
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.stderr
    assert "_exc" not in result.stderr  # prelude frame hidden from the traceback


def test_timeout_kills_process(tmp_path: Path) -> None:
    result = run_code("while True:\n    pass", tmp_path, timeout_s=1)
    assert result.timed_out
    assert result.exit_code is None
    assert result.duration_s < 6


def test_timeout_kills_child_processes(tmp_path: Path) -> None:
    # Reloading subprocess bypasses the Python-level block; the tree kill must still work.
    code = (
        "import importlib, subprocess, sys, time\n"
        "importlib.reload(subprocess)\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    result = run_code(code, tmp_path, timeout_s=3)
    assert result.timed_out
    child_pid = int(result.stdout.strip())
    deadline = time.monotonic() + 3
    while _process_alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _process_alive(child_pid)


def test_network_is_blocked(tmp_path: Path) -> None:
    code = (
        "import socket\n"
        "for attempt in (lambda: socket.create_connection(('127.0.0.1', 80)),\n"
        "                lambda: socket.socket(),\n"
        "                lambda: socket.getaddrinfo('example.com', 80)):\n"
        "    try:\n"
        "        attempt()\n"
        "        print('ALLOWED')\n"
        "    except PermissionError as e:\n"
        "        print(e)\n"
    )
    result = run_code(code, tmp_path)
    assert "ALLOWED" not in result.stdout
    assert result.stdout.count("network access is disabled") == 3


def test_starting_processes_is_blocked(tmp_path: Path) -> None:
    code = (
        "import os, subprocess, sys\n"
        "for attempt in (lambda: subprocess.run([sys.executable, '-c', 'pass']),\n"
        "                lambda: os.system('echo hi')):\n"
        "    try:\n"
        "        attempt()\n"
        "        print('ALLOWED')\n"
        "    except PermissionError as e:\n"
        "        print(e)\n"
    )
    result = run_code(code, tmp_path)
    assert "ALLOWED" not in result.stdout
    assert result.stdout.count("starting processes is disabled") == 2


def test_environment_is_scrubbed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    result = run_code("import os; print(os.environ.get('SECRET_TOKEN'))", tmp_path)
    assert result.stdout.strip() == "None"


def test_working_directory_is_workdir(tmp_path: Path) -> None:
    result = run_code("import os; print(os.getcwd())", tmp_path)
    assert Path(result.stdout.strip()) == tmp_path.resolve()


def test_long_output_is_truncated(tmp_path: Path) -> None:
    result = run_code("print('a' * 500_000 + 'END')", tmp_path, max_output_chars=1_000)
    assert result.truncated
    assert result.stdout.startswith("a")
    assert result.stdout.rstrip().endswith("END")
    assert "output truncated" in result.stdout
    assert len(result.stdout) < 1_100


def test_runaway_output_is_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "MAX_CAPTURE_BYTES", 1_000_000)
    result = run_code("while True:\n    print('x' * 10_000)", tmp_path, timeout_s=20)
    assert result.truncated
    assert not result.timed_out
    assert result.exit_code is None
    assert result.duration_s < 10


def test_unicode_round_trips(tmp_path: Path) -> None:
    result = run_code("print('مرحبا 25°C')", tmp_path)
    assert result.stdout.strip() == "مرحبا 25°C"


def test_input_fails_fast(tmp_path: Path) -> None:
    result = run_code("input()", tmp_path, timeout_s=10)
    assert not result.timed_out
    assert "EOFError" in result.stderr


def test_rejects_missing_workdir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        run_code("pass", tmp_path / "missing")


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"max_output_chars": 0}])
def test_rejects_bad_limits(tmp_path: Path, kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        run_code("pass", tmp_path, **kwargs)  # type: ignore[arg-type]
