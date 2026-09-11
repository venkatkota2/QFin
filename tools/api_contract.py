"""Inspect documented public contracts; regenerate only on deliberate API review."""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import json
import warnings
from pathlib import Path


def default_value(value):
    if value is inspect.Parameter.empty:
        return {"required": True}
    if isinstance(value, enum.Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__name__}.{value.name}"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [default_value(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def signature(value):
    if inspect.isclass(value) and issubclass(value, enum.Enum):
        # Python changes EnumType construction machinery; the public member/value
        # conversion contract is stable and tested separately.
        return [{"name": "value", "kind": "POSITIONAL_OR_KEYWORD", "default": {"required": True}}]
    try:
        return [
            {"name": p.name, "kind": p.kind.name, "default": default_value(p.default)}
            for p in inspect.signature(value).parameters.values()
        ]
    except (ValueError, TypeError):
        return None


def contract():
    result = {}
    for namespace in ("qfin", "qfin.finance", "qfin.compiler", "qfin.validation"):
        module = importlib.import_module(namespace)
        for name in module.__all__:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                value = getattr(module, name)
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            entry = {"signature": signature(value)}
            if inspect.isclass(value):
                entry["methods"] = {
                    key: signature(getattr(value, key))
                    for key in dir(value)
                    if not key.startswith("_")
                    and callable(getattr(value, key))
                    and (getattr(getattr(value, key), "__module__", "") or "").startswith("qfin")
                }
                if dataclasses.is_dataclass(value):
                    entry["fields"] = [
                        field.name
                        for field in dataclasses.fields(value)
                        if not field.name.startswith("_")
                    ]
                if issubclass(value, enum.Enum):
                    entry["enum"] = {key: item.value for key, item in value.__members__.items()}
            result[f"{namespace}.{name}"] = entry
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(contract(), indent=2, sort_keys=True) + "\n")
