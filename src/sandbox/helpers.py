"""
Subprocess sandbox for code execution with resource limits.

Ported from ariahw/rl-rewardhacking src/evaluate/helpers.py.
Self-contained: only stdlib + dataclasses dependencies.
"""

import os
import json
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field


def _get_python_executable():
    """Get the resolved Python executable path, re-resolving each time to avoid stale paths."""
    resolved = os.path.realpath(sys.executable)
    if not os.path.exists(resolved) or not os.access(resolved, os.X_OK):
        return sys.executable
    return resolved


@dataclass
class CodeRunResult:
    """Result of running code in a subprocess."""
    success: bool = True       # Ran without errors
    compiled: bool = True      # Compiled successfully
    timeout: bool = False      # Did it timeout?
    oom: bool = False          # Did it run out of memory?
    stdout: dict = field(default_factory=dict)  # Parsed JSON from stdout


@contextmanager
def temporary_env_variable(key, value):
    """Temporarily set an environment variable, restoring the original on exit."""
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = original_value


_SUBPROCESS_CODE = textwrap.dedent(
    f"""
    import io
    import json
    import resource
    import signal
    import sys
    from contextlib import redirect_stdout

    memory_mb = int(sys.argv[1])
    time_limit = float(sys.argv[2])
    memory_bytes = max(memory_mb, 1) * 1024 * 1024
    cpu_seconds = max(int(time_limit), 1)

    class TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {{time_limit}} seconds")

    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_RSS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(int(time_limit), 1))

    stdout_buffer = io.StringIO()
    namespace = {{}}
    code = sys.stdin.read()

    output = {{
        "success": True,
        "compiled": True,
        "timeout": False,
        "oom": False,
        "stdout": {{}}
    }}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, namespace)
        stdout_content = stdout_buffer.getvalue()
        try:
            lines = stdout_content.strip().split('\\n')
            parsed_json = None
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{{') and line.endswith('}}'):
                    try:
                        parsed_json = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            output["stdout"] = parsed_json if parsed_json is not None else {{}}
        except Exception:
            output["stdout"] = {{"raw": stdout_content}}
    except (SyntaxError, IndentationError):
        output["success"] = False
        output["compiled"] = False
    except MemoryError:
        output["success"] = False
        output["oom"] = True
        output["stdout"] = {{}}
    except TimeoutException:
        output["success"] = False
        output["timeout"] = True
        output["stdout"] = {{}}
    except Exception as e:
        output["success"] = False
        output["stdout"] = {{"raw": str(e)}}
    finally:
        signal.alarm(0)

    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()
    """
).strip()


def _execute_in_subprocess(
    code: str,
    timeout: int,
    memory_limit: int,
    raise_exceptions: bool = False,
) -> CodeRunResult:
    """Execute code in an isolated Python subprocess with resource limits."""
    python_executable = _get_python_executable()

    args = [
        python_executable,
        "-c",
        _SUBPROCESS_CODE,
        str(max(memory_limit, 1)),
        str(max(timeout, 1)),
    ]

    process = None
    try:
        process = subprocess.Popen(
            args,
            executable=python_executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True,
        )
        stdout, stderr = process.communicate(input=code, timeout=max(timeout, 1) + 1)
        returncode = process.returncode
        process = None
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        return CodeRunResult(success=False, timeout=True)
    except OSError as e:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        if raise_exceptions:
            raise e
        return CodeRunResult(success=False, compiled=False, stdout={"raw": f"OSError: {str(e)}"})
    except Exception as e:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
            process = None
        if raise_exceptions:
            raise e
        return CodeRunResult(success=False, stdout={"raw": str(e)})
    finally:
        if process is not None:
            try:
                process.kill()
                process.wait()
            except Exception:
                pass

    stdout_str = stdout.strip()
    try:
        result = json.loads(stdout_str)
        if returncode != 0:
            result["success"] = False
        return CodeRunResult(**result)
    except json.JSONDecodeError:
        return CodeRunResult(success=False, compiled=False, stdout={"raw": stdout_str})


def run_code_subprocess(
    program: str,
    memory_limit: int = 1024,
    timeout: int = 1,
    debug: bool = False,
) -> CodeRunResult:
    """Execute a single program in an isolated subprocess.

    Args:
        program: Python source code to execute.
        memory_limit: Memory limit in MB (default 1024).
        timeout: CPU time limit in seconds (default 1).
        debug: Print debug output if True.

    Returns:
        CodeRunResult with execution status and stdout.
    """
    with temporary_env_variable("TOKENIZERS_PARALLELISM", "false"):
        result = _execute_in_subprocess(
            program,
            timeout=timeout,
            memory_limit=memory_limit,
        )
        if debug:
            print("Run test result", result.compiled, result.stdout)
        return result


def create_test_runner_code(
    setup_code: str,
    program: str,
    test_list: list[str],
    max_failures: int,
) -> str:
    """Create a Python program that runs tests and returns JSON results.

    The generated code will:
    1. Execute setup_code and program
    2. Run each test assertion, counting successes and failures
    3. Capture exception types for failures
    4. Stop after max_failures failures
    5. Print JSON with tests_evaluated, tests_passed, and error info to stdout

    Args:
        setup_code: Code to run before the program (e.g., imports).
        program: The code under test.
        test_list: List of assertion statements (e.g., ["assert f(1) == 2"]).
        max_failures: Stop after this many failures.

    Returns:
        A complete Python program as a string.
    """
    test_list_repr = repr(test_list)
    max_failures_repr = repr(max_failures)

    test_runner = f"""{setup_code}

{program}

# Test runner
import json
import sys

tests_evaluated = 0
tests_passed = 0
test_errors = []
failures_count = 0

test_cases = {test_list_repr}

for test_case in test_cases:
    try:
        tests_evaluated += 1
        exec(test_case)
        tests_passed += 1
    except AssertionError as e:
        exception_type = "AssertionError"
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except (SyntaxError, IndentationError) as e:
        exception_type = type(e).__name__
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except SystemExit as e:
        exception_type = "SystemExit"
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break
    except BaseException as e:
        exception_type = type(e).__name__
        test_errors.append(f"{{exception_type}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {max_failures_repr}:
            break

result = {{
    "tests_evaluated": tests_evaluated,
    "tests_passed": tests_passed,
    "tests_total": len(test_cases),
    "test_errors": test_errors
}}
print(json.dumps(result))
sys.stdout.flush()
"""
    return test_runner
