"""The reviewed 1.1.1 public contract remains usable throughout 1.x."""

import importlib
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
        old_signature, new_signature = old["signature"], new["signature"]
        if old_signature is None:
            assert new_signature is None, name
        else:
            names = {item["name"] for item in old_signature}
            assert [item for item in new_signature if item["name"] in names] == old_signature, name
            for item in new_signature:
                if item["name"] not in names:
                    assert item["kind"] == "KEYWORD_ONLY", name
                    assert item["default"] != {"required": True}, name
        for method, signature in old.get("methods", {}).items():
            assert new["methods"][method] == signature, f"{name}.{method}"
        if "fields" in old:
            assert new["fields"][: len(old["fields"])] == old["fields"], name
        if "enum" in old:
            assert new["enum"] == old["enum"], name
            namespace, _, class_name = name.rpartition(".")
            enum = getattr(importlib.import_module(namespace), class_name)
            for member, value in old["enum"].items():
                assert enum(value) is enum.__members__[member]


def test_stable_serialized_key_paths():
    spec = importlib.util.spec_from_file_location(
        "serialization_contract", ROOT / "tools/serialization_contract.py"
    )
    serialization = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serialization)
    baseline = json.loads((ROOT / "tests/api/snapshots/serialization-1.1.1.json").read_text())
    current = serialization.contracts()
    for name, paths in baseline.items():
        assert set(paths) <= set(current[name]), name
