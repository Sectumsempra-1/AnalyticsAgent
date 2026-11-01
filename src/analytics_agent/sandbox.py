"""Run untrusted Python code in a separate process with basic guards.

WARNING: this is a development sandbox, not a security boundary. The network and
process-start blocks are Python-level patches; determined code (ctypes, reloading
modules, low-level C APIs) can bypass them. Production use requires a container-level
sandbox (no network, memory/CPU limits, read-only filesystem).

What this module does provide:
- a fresh interpreter in isolated mode (-I), with a minimal environment (no secrets)
- a caller-chosen working directory
- a wall-clock timeout that kills the whole process tree
- bounded output capture (stdout/stderr go to files, only a capped amount is read back)
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 10_000
MAX_CAPTURE_BYTES = 10 * 1024 * 1024  # kill the run if it writes more than this
_POLL_INTERVAL_S = 0.05

_BLOCKED_OS_FUNCTIONS = (
    "system", "popen", "startfile", "fork", "forkpty",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp",
)  # fmt: skip

# Runs inside the child before the user's code. Installs guards, then executes the
# code file given as argv[1] in a fresh namespace.
_PRELUDE = f"""
import os, socket, subprocess, sys

def _deny(what):
    def blocked(*args, **kwargs):
        raise PermissionError(f"{{what}} is disabled in the sandbox")
    return blocked

class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise PermissionError("network access is disabled in the sandbox")

socket.socket = _BlockedSocket
socket.create_connection = _deny("network access")
socket.getaddrinfo = _deny("network access")
subprocess.Popen._execute_child = _deny("starting processes")
for _name in {_BLOCKED_OS_FUNCTIONS!r}:
    if hasattr(os, _name):
        setattr(os, _name, _deny("starting processes"))

_path = sys.argv[1]
sys.argv = [_path]
with open(_path, encoding="utf-8") as _f:
    _code = compile(_f.read(), _path, "exec")
try:
    exec(_code, {{"__name__": "__main__", "__file__": _path}})
except SystemExit:
    raise
except BaseException as _exc:
    import traceback
    traceback.print_exception(type(_exc), _exc, _exc.__traceback__.tb_next)
    sys.exit(1)
"""


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one sandboxed run."""

    stdout: str
    stderr: str
    exit_code: int | None  # None when the sandbox killed the process
    timed_out: bool
    truncated: bool
    duration_s: float


def run_code(
    code: str,
    workdir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> ExecutionResult:
    """Execute `code` in a fresh Python process whose working directory is `workdir`.

    The process is killed if it exceeds `timeout_s` or writes more than
    MAX_CAPTURE_BYTES of output. Each of stdout and stderr is cut to
    `max_output_chars`, keeping the start and the end.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_output_chars < 1:
        raise ValueError("max_output_chars must be at least 1")
    workdir = Path(workdir).resolve()
    if not workdir.is_dir():
        raise NotADirectoryError(f"workdir does not exist: {workdir}")

    with tempfile.TemporaryDirectory(prefix="sandbox-", ignore_cleanup_errors=True) as tmp:
        io_dir = Path(tmp)
        code_path = io_dir / "analysis.py"
        code_path.write_text(code, encoding="utf-8")
        temp_dir = io_dir / "tmp"
        temp_dir.mkdir()
        out_path, err_path = io_dir / "stdout.txt", io_dir / "stderr.txt"

        start = time.monotonic()
        with out_path.open("wb") as out, err_path.open("wb") as err:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", "-X", "utf8", "-c", _PRELUDE, str(code_path)],
                cwd=workdir,
                env=_child_env(temp_dir),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=os.name != "nt",
            )
            timed_out, overflowed = _wait(proc, start + timeout_s, (out_path, err_path))
        duration = time.monotonic() - start

        stdout, out_cut = _read_capped(out_path, max_output_chars)
        stderr, err_cut = _read_capped(err_path, max_output_chars)

    killed = timed_out or overflowed
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=None if killed else proc.returncode,
        timed_out=timed_out,
        truncated=overflowed or out_cut or err_cut,
        duration_s=round(duration, 3),
    )


def _child_env(temp_dir: Path) -> dict[str, str]:
    """Minimal environment: nothing from the parent except what Python needs to start."""
    env = {"TEMP": str(temp_dir), "TMP": str(temp_dir), "TMPDIR": str(temp_dir)}
    for key in ("PATH", "SYSTEMROOT"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _wait(
    proc: subprocess.Popen[bytes], deadline: float, outputs: tuple[Path, ...]
) -> tuple[bool, bool]:
    """Wait for exit. Returns (timed_out, overflowed); kills the tree on either."""
    while True:
        try:
            proc.wait(timeout=_POLL_INTERVAL_S)
            return False, False
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() >= deadline:
            _kill_tree(proc)
            return True, False
        if sum(p.stat().st_size for p in outputs) > MAX_CAPTURE_BYTES:
            _kill_tree(proc)
            return False, True


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process and every process it started."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined,unused-ignore]
    with suppress(ProcessLookupError):
        proc.kill()
    proc.wait(timeout=5)


def _read_capped(path: Path, max_chars: int) -> tuple[str, bool]:
    """Read up to `max_chars` characters, keeping head and tail. Returns (text, truncated)."""
    size = path.stat().st_size
    byte_budget = max_chars * 4  # UTF-8 uses at most 4 bytes per character
    with path.open("rb") as f:
        if size <= byte_budget * 2:
            text = _decode(f.read())
            if len(text) <= max_chars:
                return text, False
            head, tail = text, text
        else:
            head = _decode(f.read(byte_budget))
            f.seek(-byte_budget, os.SEEK_END)
            tail = _decode(f.read())
    half = max_chars // 2
    marker = f"\n...[output truncated: {size} bytes total]...\n"
    return head[:half] + marker + tail[len(tail) - (max_chars - half) :], True


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")
