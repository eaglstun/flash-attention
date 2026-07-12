"""Conftest for the MPS parity suite.

Deliberately does NOT reuse tests/cute/conftest.py (it shells out to
nvidia-smi). Collects the worst-case error-budget ratios per dtype from
``checks.check_tensor_budget`` and prints them at the end of the session so
"how close to the budget are we" is visible, not just pass/fail.
"""

import checks


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not checks.WORST_RATIOS:
        return
    terminalreporter.write_sep(
        "=", "MPS parity: worst error-budget ratios (diff / budget, must be <= 1)"
    )
    for key in sorted(checks.WORST_RATIOS):
        ratio, detail = checks.WORST_RATIOS[key]
        terminalreporter.write_line(f"{key:>18}: {ratio:8.4f}  ({detail})")
