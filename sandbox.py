"""A small, self-contained Python sandbox.

The agent's most important tool is "run arbitrary Python and tell me what it
printed". That code is model-generated, so we run it as a *subprocess* with:

  * a scrubbed environment (no API keys / bot token leak into the sandbox),
  * a wall-clock timeout (killed if it hangs, e.g. a slow download),
  * best-effort address-space + CPU limits on POSIX,
  * a per-run working directory that persists files across calls in one run
    (so the agent can download once and re-read), and
  * truncated stdout/stderr so a runaway print can't blow up the model context.

Note: this is deliberately *not* a hardened security boundary — it is your own
bot solving your own grading questions. It's a blast-radius reducer, not a jail.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass

MAX_OUTPUT_CHARS = 24_000


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    def as_tool_output(self) -> str:
        parts = []
        if self.timed_out:
            parts.append(f"[timed out after the time limit; process was killed]")
        if self.stdout:
            parts.append("STDOUT:\n" + self.stdout)
        if self.stderr:
            parts.append("STDERR:\n" + self.stderr)
        if not self.stdout and not self.stderr:
            parts.append("(no output — remember to print() your results)")
        parts.append(f"[exit code: {self.exit_code}]")
        return "\n".join(parts)


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[: MAX_OUTPUT_CHARS - 2000]
    tail = text[-2000:]
    return f"{head}\n...[truncated {len(text) - MAX_OUTPUT_CHARS} chars]...\n{tail}"


def _make_limits(mem_mb: int, cpu_seconds: int):
    """Return a preexec_fn applying resource limits, or None if unavailable."""
    try:
        import resource  # POSIX only
    except Exception:
        return None

    def _apply():
        # Memory (address space). Guard each call: some platforms (notably
        # macOS) reject RLIMIT_AS — we simply skip what we can't set.
        for res_name, limit in (
            ("RLIMIT_AS", mem_mb * 1024 * 1024),
            ("RLIMIT_CPU", cpu_seconds),
            ("RLIMIT_FSIZE", 512 * 1024 * 1024),
        ):
            res = getattr(resource, res_name, None)
            if res is None:
                continue
            try:
                soft, hard = resource.getrlimit(res)
                new_hard = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
                resource.setrlimit(res, (min(limit, new_hard), new_hard))
            except Exception:
                pass

    return _apply


def run_python(
    code: str,
    workdir: str,
    timeout: int = 70,
    mem_mb: int = 1536,
) -> ExecResult:
    os.makedirs(workdir, exist_ok=True)

    # Minimal, secret-free environment. We deliberately do NOT forward the
    # parent environment (which holds OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, ...).
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": workdir,
        "TMPDIR": workdir,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "MPLBACKEND": "Agg",
        # Keep certifi/ssl happy for outbound HTTPS from the sandbox.
        "SSL_CERT_FILE": os.environ.get("SSL_CERT_FILE", ""),
    }
    env = {k: v for k, v in env.items() if v != ""}

    # Wrap the code so a bare final expression still shows something useful and
    # tracebacks are readable.
    program = textwrap.dedent(code)

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", program],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_make_limits(mem_mb, timeout + 5) if os.name == "posix" else None,
        )
        return ExecResult(
            stdout=_truncate(proc.stdout),
            stderr=_truncate(proc.stderr),
            exit_code=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return ExecResult(
            stdout=_truncate(out),
            stderr=_truncate(err),
            exit_code=124,
            timed_out=True,
        )
    except Exception as e:  # pragma: no cover - defensive
        return ExecResult(stdout="", stderr=f"sandbox error: {type(e).__name__}: {e}",
                          exit_code=1, timed_out=False)
