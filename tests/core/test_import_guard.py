from __future__ import annotations

import ast
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parents[2] / "core"


def test_core_has_no_forbidden_or_misplaced_network_imports() -> None:
    for path in CORE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue

            for module in modules:
                forbidden = (
                    module == "urllib.request"
                    or module.startswith("urllib.request.")
                    or module == "requests"
                    or module.startswith("requests.")
                    or module == "httpx"
                    or module.startswith("httpx.")
                    or module == "openai"
                    or module.startswith("openai.")
                    or module == "anthropic"
                    or module.startswith("anthropic.")
                )
                if isinstance(node, ast.ImportFrom) and node.module == "urllib":
                    forbidden = forbidden or any(
                        alias.name == "request" for alias in node.names
                    )
                assert not forbidden, (
                    f"{path.name} imports forbidden module {module}"
                )
                if module == "googletrans":
                    assert path.name == "translation.py", (
                        f"googletrans may only be imported in translation.py, found in {path.name}"
                    )