import json
import os
import subprocess
import sys


def test_api_metadata_matches_installed_package_version(tmp_path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "TRADING_DEMO_MODE": "true",
            "TRADING_EXECUTION_MODE": "internal-paper",
            "TRADING_STRATEGY_MODE": "explainable",
            "TRADING_DATABASE_PATH": str(tmp_path / "events.db"),
            "TRADING_MODEL_REGISTRY_PATH": str(tmp_path / "models"),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from trading_app import __version__; "
                "from trading_app.main import app; "
                "print(json.dumps({'package': __version__, "
                "'app': app.version, 'openapi': app.openapi()['info']['version']}))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    versions = json.loads(result.stdout)
    assert versions == {
        "package": versions["package"],
        "app": versions["package"],
        "openapi": versions["package"],
    }
