"""The reviewed 1.1.1 public contract remains usable throughout 1.x."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("api_contract", ROOT / "tools/api_contract.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stable_api_contract():
    baseline = json.loads((ROOT / "tests/api/snapshots/qfin-1.1.1.json").read_text())
    current = module.contract()
    for name, old in baseline.items():
        assert name in current, name
        new = current[name]
        assert new["signature"] == old["signature"], name
        for method, signature in old.get("methods", {}).items():
            assert new["methods"][method] == signature, f"{name}.{method}"
        if "fields" in old:
            assert new["fields"][: len(old["fields"])] == old["fields"], name
        if "enum" in old:
            assert new["enum"] == old["enum"], name
