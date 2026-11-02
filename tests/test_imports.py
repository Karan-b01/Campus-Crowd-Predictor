"""Smoke test ensuring every project module imports."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src",
        "src.config",
        "src.audit",
        "src.data",
        "src.features",
        "src.models",
        "src.evaluate",
        "src.utils",
    ],
)
def test_src_modules_import(module_name: str) -> None:
    """Verify each source module can be imported."""
    assert importlib.import_module(module_name) is not None
