"""Forwarding entrypoint for Streamlit."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    # Streamlit may put the app directory first on sys.path. That makes the
    # sibling app.py shadow the app package when app.py imports app.components.
    sys.path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != project_root
    ]
    sys.path.insert(0, str(project_root))
    app_path = Path(__file__).resolve().parent / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
