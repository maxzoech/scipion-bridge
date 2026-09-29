import os
import pytest
import tempfile
from scipion_bridge.core.utils.shell import Domain
from scipion_bridge.core.environment.cmd_exec import StandaloneExecProvider
from subprocess import PIPE

PYTHON_DOMAIN = Domain("PY", [])


def test_shell_exec():

    provider = StandaloneExecProvider()

    output = provider(
        "python",
        PYTHON_DOMAIN,
        [
            "python",
            "-c",
            '"import sys; sys.exit(0)"',
        ],
        run_args={"shell": True, "stdout": PIPE, "stderr": PIPE},
    )

    assert output == 0

    with pytest.raises(RuntimeError):
        output = provider(
            "python",
            PYTHON_DOMAIN,
            [
                "python",
                "-c",
                '"import sys; sys.exit(42)"',
            ],
            run_args={"shell": True, "stdout": PIPE, "stderr": PIPE},
        )
