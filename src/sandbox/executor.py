"""
Code executor for evaluating generated code against test assertions.

Ported from ariahw/rl-rewardhacking src/evaluate/evaluator.py.
Uses helpers.py for subprocess isolation.
"""

import ast
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from sandbox.helpers import CodeRunResult, create_test_runner_code, run_code_subprocess


@dataclass
class CodeEvaluationResult:
    """Result of evaluating code against a set of tests."""
    parsed_response: str | None = None
    is_formatted: bool = True
    can_compile: bool = True
    pass_rate: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    tests_evaluated: int = 0
    test_errors: list[str] = field(default_factory=list)


class CodeExecutor:
    """Simplified port of ariahw's CodeEvaluator.

    Evaluates Python code against test assertions in isolated subprocesses
    with memory and CPU time limits.
    """

    def __init__(
        self,
        num_workers: int = 8,
        memory_mb: int = 1024,
        timeout_s: int = 3,
        max_failures: int = 1,
        debug: bool = False,
    ):
        """
        Args:
            num_workers: Max parallel evaluation workers for batch_evaluate.
            memory_mb: Memory limit per subprocess in MB.
            timeout_s: CPU time limit per subprocess in seconds.
            max_failures: Stop running tests after this many failures.
            debug: Print debug output.
        """
        self.num_workers = num_workers
        self.memory_mb = memory_mb
        self.timeout_s = timeout_s
        self.max_failures = max_failures
        self.debug = debug

    def evaluate_single(
        self,
        code: str,
        test_list: list[str],
        setup_code: str = "",
        skip_parse: bool = True,
    ) -> CodeEvaluationResult:
        """Run code against test assertions in an isolated subprocess.

        Args:
            code: Python source code (or raw LLM response if skip_parse=False).
            test_list: List of assert statements to test the code.
            setup_code: Code to run before the program (e.g., imports).
            skip_parse: If True, code is already pure Python. If False, extract
                from ```python blocks first.

        Returns:
            CodeEvaluationResult with pass rate, compilation status, and errors.
        """
        result = CodeEvaluationResult(
            tests_total=len(test_list),
        )

        if not skip_parse:
            program = self.parse_response(code)
        else:
            program = code

        if program is None:
            result.is_formatted = False
            result.can_compile = False
            return result

        result.parsed_response = program

        # Build and run the test harness in a subprocess
        test_runner_code = create_test_runner_code(
            setup_code, program, test_list, self.max_failures
        )

        code_run_result: CodeRunResult = run_code_subprocess(
            test_runner_code,
            timeout=self.timeout_s,
            memory_limit=self.memory_mb,
            debug=self.debug,
        )

        # Process results
        result.can_compile = code_run_result.compiled
        if not code_run_result.compiled:
            result.test_errors.append("MasterError: CompilationError")
        if code_run_result.timeout:
            result.test_errors.append("MasterError: TimeoutError")
        if code_run_result.oom:
            result.test_errors.append("MasterError: OOMError")
        if not code_run_result.success:
            result.test_errors.append(
                "MasterError: UnknownError: "
                + str(code_run_result.stdout.get("raw", "No response"))
            )

        result.tests_evaluated = code_run_result.stdout.get("tests_evaluated", 0)
        result.tests_passed = code_run_result.stdout.get("tests_passed", 0)
        result.pass_rate = (
            (result.tests_passed / result.tests_total) if result.tests_total > 0 else 0.0
        )
        result.test_errors += code_run_result.stdout.get("test_errors", [])

        return result

    def evaluate_batch(
        self,
        calls: list[dict[str, Any]],
    ) -> list[CodeEvaluationResult]:
        """Evaluate multiple code samples in parallel.

        Args:
            calls: List of dicts, each containing kwargs for evaluate_single
                (code, test_list, setup_code, skip_parse).

        Returns:
            List of CodeEvaluationResult in the same order as input.
        """
        if not calls:
            return []

        results: list[CodeEvaluationResult | None] = [None] * len(calls)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {
                executor.submit(self.evaluate_single, **call_kwargs): idx
                for idx, call_kwargs in enumerate(calls)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()

        return results  # type: ignore[return-value]

    @staticmethod
    def parse_response(response: str) -> str | None:
        """Extract Python code from ```python fenced code blocks.

        Joins multiple code blocks with double newlines.

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted code string, or None if no code blocks found.
        """
        blocks = re.findall(
            r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE
        )
        if not blocks:
            return None
        cleaned_blocks = [b.strip() for b in blocks if b.strip()]
        if not cleaned_blocks:
            return None
        return "\n\n".join(cleaned_blocks)

    @staticmethod
    def extract_function(code: str, func_name: str) -> str:
        """Extract a function definition from code using AST parsing.

        Args:
            code: Python source code.
            func_name: Name of the function to extract.

        Returns:
            The unparsed function source, or empty string if not found.
        """
        try:
            tree = ast.parse(code)
        except Exception:
            return ""

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return ast.unparse(node)

        return ""

    @staticmethod
    def extract_function_parent(code: str, func_name: str) -> str | None:
        """Extract the parent class name of a function if it's a method.

        Args:
            code: Python source code.
            func_name: Name of the function to find.

        Returns:
            Parent class name, or None if not a method or not found.
        """
        try:
            tree = ast.parse(code)
        except Exception:
            return None

        class FunctionParentExtractor(ast.NodeVisitor):
            def __init__(self, target_name: str):
                self.target_name = target_name
                self.class_name: str | None = None
                self.current_class: str | None = None

            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.class_name = self.current_class
                self.generic_visit(node)

        extractor = FunctionParentExtractor(func_name)
        extractor.visit(tree)
        return extractor.class_name

    def check_compile(self, code: str) -> bool:
        """Check if code compiles and runs without errors.

        Args:
            code: Raw LLM response text.

        Returns:
            True if code compiles successfully.
        """
        program = self.parse_response(code)
        if program is None:
            return False

        result = run_code_subprocess(
            program,
            timeout=self.timeout_s,
            memory_limit=self.memory_mb,
            debug=self.debug,
        )
        return result.compiled
